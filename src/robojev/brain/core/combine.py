"""Code combines the three answers (docs/v2.md "The decision", "Confidence decides escalation").

1. The right-now answer applies at once. Below the right-now gate it falls back to the cautious
   choice (table CAUTIOUS). While an LLM replans, it decides whether the step carries on or holds.
2. The in-plan fix applies only if the Router says stay_local and the weakest confidence among the
   route and the fix clears the stay_local gate.
3. Otherwise the fast or capable LLM replans. Each route has its own threshold, set by what a wrong
   answer costs: a low-confidence route moves one rung up the ladder
   (stay_local -> fast_llm -> capable_llm; ask_user -> capable_llm).
Code rules on top: a plan-check "yes" (or an unsure one) sends the plan back to the LLM; a failed
step or an unmet plan with no in-plan fix cannot stay local; a Jev error is treated as low confidence
everywhere.

A new plan (`plan_arrived`) is routed by the plan check alone. docs/v2.md: "a new plan is presumed
good: it goes back to the LLM only when one of these specific questions says yes, not on a general
doubt". The generic route question did the opposite in the live runs (docs/live-results.md, H4): on
185 new plans the oracle said to keep, it sent back 91 with Jev (49%) and 123 with the Decisions API
(66%), unsure that a fresh plan "still holds as written" or confidently asking for a replan, while
the plan-check Nouls flagged none of them and alone caught 59/60 planted bad plans with Jev. So on
plan_arrived the route is stay_local unless a plan-check Noul flags a problem (then fast_llm, with
the problems attached), with code-certain confidence; a route or in-plan-fix answer in the request
is ignored for that event (core/decision.py no longer asks them). The replan-loop limit (rules.py,
3 rejections since the user last spoke) stays as the backstop. The offline replay of the recorded
answers under this rule: experiments/replay_plan_route.py.

A routine step_done is kept local on the decider's choice alone. The stay_local gate guards what a
wrong LOCAL answer would do: an in-plan fix applied on a guess. On a step that finished normally
with no fix and no plan-check problem, keeping local does nothing but start the next planned step,
which the plan already says to do; a wrong keep costs a later replan (plan_done_unmet, a failed
step), not a wrong move. The live spelling battery (docs/live-results.md) had Jev answer stay_local
on such steps at 0.3-0.77, and the gate turned 28 of them into planner calls (the replan-loop limit
counts only plans sent back, so nothing capped them). So `routine(event, fix, problems)` says when
the choice stands as given; the loop's enforce_gate (rules.py) applies the same rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .decision import build_questions, build_state
from .data import Combined, Event

# Low-confidence right-now answers fall back to the cautious choice. `resume` falls back to carry_on,
# i.e. stay paused/holding (the paused-idle event re-asks later); `back_off` falls back to pause
# (no motion around a hand on a guess).
CAUTIOUS = {"carry_on": "hold", "re_target": "hold", "resume": "carry_on", "hold": "hold", "pause": "pause", "back_off": "pause"}


@dataclass
class Gates:
    right_now: float = 0.5
    stay_local: float = 0.8      # the in-plan fix / keeping it local (e12v2: 0.8 caught 98% of wrong answers, 0.7 89%)
    fast_llm: float = 0.3        # below: capable LLM
    capable_llm: float = 0.0
    ask_user: float = 0.5        # below: capable LLM instead of bothering the user
    check: float = 0.5           # plan check / progress Nouls, on |2p - 1|

    def as_dict(self):
        return asdict(self)


@dataclass
class Ablation:
    right_now: bool = True
    in_plan_fix: bool = True
    router: bool = True

    @property
    def name(self) -> str:
        off = [n for n, on in (("right-now", self.right_now), ("in-plan-fix", self.in_plan_fix), ("router", self.router)) if not on]
        return "jev" + "".join(f"-no-{n}" for n in off)


def noul_conf(p: float) -> float:
    return abs(2 * p - 1)


def routine(event_kind: str, fix: str, problems: list | None) -> bool:
    """A step finished normally and nothing local is being changed (module doc): a stay_local answer
    stands at any confidence."""
    return event_kind == "step_done" and fix == "none" and not problems


def _ch(answers: dict, key: str) -> tuple[str | None, float]:
    a = (answers or {}).get(key) or {}
    return a.get("choice"), float(a.get("confidence") or 0.0)


def combine(event: Event, answers: dict | None, view, gates: Gates, abl: Ablation = Ablation()) -> Combined:
    hand = view.hand is not None
    if answers is None:
        return Combined("pause" if hand else "hold", "none", "fast_llm", "jev_error", conf={})
    conf = {}
    rn, c_rn = _ch(answers, "right_now") if abl.right_now else ("carry_on", 1.0)
    rn = rn or "carry_on"
    rn_raw = rn
    conf["right_now"] = c_rn
    if c_rn < gates.right_now:
        rn = CAUTIOUS.get(rn, "hold")
        if rn == "hold" and hand:
            rn = "pause"
    # not asked (ablated, or "none" was the only option code offered) = none, certain
    fix, c_fix = _ch(answers, "in_plan_fix") if abl.in_plan_fix and "in_plan_fix" in answers else ("none", 1.0)
    fix = fix or "none"
    conf["in_plan_fix"] = c_fix
    problems, reason = [], ""
    for k, a in answers.items():
        if k.startswith("check."):
            p = float(a.get("noul", 0.0))
            conf[k] = noul_conf(p)
            if p >= 0.5:
                problems.append(f"{view.name(k[6:])} is something the task asks about but the plan leaves out")
            elif noul_conf(p) < gates.check:
                problems.append(f"{view.name(k[6:])} may be something the task asks about that the plan leaves out (unsure)")
    done = {}
    for k, a in answers.items():
        if k.startswith("done."):
            p = float(a.get("noul", 0.0))
            conf[k] = noul_conf(p)
            done[k[5:]] = p >= 0.5 and noul_conf(p) >= gates.check
    if event.kind == "plan_arrived":
        # a new plan is presumed good: the plan check alone sends it back (module doc); any route or
        # fix answer is ignored, and the confidences are code's (certain), so no gate escalates it
        fix, c_fix = "none", 1.0
        conf["in_plan_fix"] = c_fix
        route, c_route = ("fast_llm" if problems else "stay_local"), 1.0
        reason = "plan check" if problems else "new plan: the plan check found nothing"
    elif abl.router:
        route, c_route = _ch(answers, "route")
        route = route or "fast_llm"
    else:
        route, c_route = ("stay_local" if fix != "none" or event.kind == "step_done" else "fast_llm"), 1.0

    conf["route"] = c_route
    route_raw = route
    if problems and route not in ("fast_llm", "capable_llm"):
        route, reason = "fast_llm", "plan check"
    if route == "stay_local":
        w = min(c_route, c_fix)
        if w < gates.stay_local and not routine(event.kind, fix, problems):
            route, reason = "fast_llm", "low confidence"
        elif w < gates.stay_local:
            reason = "routine: the step finished and nothing local changes"

        elif fix == "none" and event.kind in ("step_failed", "plan_done_unmet"):
            route, reason = "fast_llm", "nothing local fixes it"
    elif route == "fast_llm" and c_route < gates.fast_llm:
        route, reason = "capable_llm", "low confidence"
    elif route == "ask_user" and c_route < gates.ask_user:
        route, reason = "capable_llm", "low confidence"
    applied_fix = fix if route == "stay_local" else "none"
    return Combined(rn, applied_fix, route, reason or route, right_now_raw=rn_raw, route_raw=route_raw, fix_raw=fix,
                    conf=conf, problems=problems, done=done)


class JevDecider:
    """Decision = one request (three groups + progress/plan-check Nouls) + combine()."""

    def __init__(self, backend, gates: Gates | None = None, ablation: Ablation | None = None):
        self.backend, self.gates, self.abl = backend, gates or Gates(), ablation or Ablation()
        self.name = self.abl.name

    def decide(self, event: Event, view) -> dict:
        groups = [g for g, on in (("right_now", self.abl.right_now), ("in_plan_fix", self.abl.in_plan_fix), ("route", self.abl.router)) if on]
        state = build_state(event, view)
        qs = build_questions(event, view, groups)
        r = self.backend.answer(state, qs, ref=(event, view))
        answers = r.get("answers")
        comb = combine(event, answers, view, self.gates, self.abl)
        return {"combined": comb, "latency_ms": r.get("latency_ms", 0.0), "request": {"state": state, "questions": qs},
                "answers": answers, "in_tok": r.get("in_tok", 0), "meta": r.get("meta", {}), "jev": True}
