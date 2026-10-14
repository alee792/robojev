"""Several deciders at the same tier, each with its own responsibilities, behind one `decide()`.

The fast tier asks three questions per event (core/decision.py): `right_now` (what the arm does this
instant), `in_plan_fix` (whether the plan absorbs the event with a small, code-checked move) and
`route` (who handles it: the robot, a planner, the user), plus the progress and plan-check Nouls.
Each is a responsibility with its own criteria and its own gate, and today one model answers all of
them in one request (`core.combine.JevDecider`). A `Panel` sends each group to the decider named
for it, in parallel, and hands the merged answers to the same `combine()`: a safety-minded model
on `right_now`, a task-minded one on `in_plan_fix`, a cheaper or slower one on `route`, each with
its own prompt or latency budget. Nothing else changes: code still merges, the gates still apply,
and the priority order (STOP > safety filter > right-now > route > fix > plan) still outranks every
answer. That is the rule for adding agents at a tier: a responsibility is a question group; an agent
is whatever answers a set of groups; arbitration stays in code.

Kept single for now: `Panel({"*": backend})` is exactly `JevDecider(backend)`. The Nouls (done.*,
check.*) ride with the `route` group's decider unless `nouls=` names another.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping

from .core.combine import Ablation, Gates, combine
from .core.decision import build_questions, build_state
from .core.data import Event

GROUPS = ("right_now", "in_plan_fix", "route")


class Panel:
    """A decider whose question groups are answered by different backends (module doc).
    `backends`: group -> DecisionBackend, with "*" as the default for groups not named."""

    def __init__(self, backends: Mapping[str, object], gates: Gates | None = None, ablation: Ablation | None = None,
                 nouls: str = "route", name: str = "panel"):
        if "*" not in backends and any(g not in backends for g in GROUPS):
            missing = [g for g in GROUPS if g not in backends]
            raise ValueError(f"no backend for {missing} and no '*' default")
        self.backends = dict(backends)
        self.gates, self.abl, self.nouls, self.name = gates or Gates(), ablation or Ablation(), nouls, name
        self._pool = ThreadPoolExecutor(max_workers=len(GROUPS), thread_name_prefix="panel")

    def backend_for(self, group: str):
        return self.backends.get(group, self.backends.get("*"))

    def decide(self, event: Event, view) -> dict:
        groups = [g for g, on in (("right_now", self.abl.right_now), ("in_plan_fix", self.abl.in_plan_fix), ("route", self.abl.router)) if on]
        state = build_state(event, view)
        qs = build_questions(event, view, groups)
        # one request per distinct backend, carrying only its groups (and the Nouls with their owner)
        owner = {k: (k if k in GROUPS else self.nouls) for k in qs}
        by_backend: dict[int, tuple[object, dict]] = {}
        for k, g in owner.items():
            be = self.backend_for(g)
            by_backend.setdefault(id(be), (be, {}))[1][k] = qs[k]
        futures = [self._pool.submit(be.answer, state, part, (event, view)) for be, part in by_backend.values()]
        results = [f.result() for f in futures]
        answers: dict = {}
        meta: dict = {}
        latency, in_tok = 0.0, 0
        for (be, part), r in zip(by_backend.values(), results):
            got = r.get("answers")
            if got is None:                      # one member failing (a Jev error) fails the decision: combine() treats None as low confidence everywhere
                answers = None
                break
            answers.update({k: v for k, v in got.items() if k in part})
            latency = max(latency, float(r.get("latency_ms", 0.0)))
            in_tok += int(r.get("in_tok", 0))
            meta[getattr(be, "name", be.__class__.__name__)] = {"groups": sorted(part), **r.get("meta", {})}
        comb = combine(event, answers, view, self.gates, self.abl)
        return {"combined": comb, "latency_ms": latency, "request": {"state": state, "questions": qs}, "answers": answers,
                "in_tok": in_tok, "meta": {"panel": meta}, "jev": True}
