"""The planner client: e12v2's plan and replan requests, built from the connected robot's manifest.

Requests carry e12v2's shape (instructions, a JSON input, a strict schema; see e12v2.core.planner)
so any e12v2 LLM backend can send them; what changes is that the instructions describe this robot
and the step schema lists only its skills (schema.py). An answer is checked in three layers, and an
invalid one gets one retry with the errors, as in e12v2:
  1. shape and manifest: every step's skill is plannable on this arm, with its required arguments;
  2. translated to e12v2's step names (steps.py), e12v2's diff checks and generic validator.
The result handed back to the loop is in e12v2's names, ready for its plan functions.

Calls block (seconds, for a real model): the loop runs them in a thread and never waits on them.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from e12v2.core import plan as P
from e12v2.core.data import WorldState
from e12v2.core.planner import LLMRequest, PlanResult, plan_view, with_errors

from ..protocol import Manifest
from ..protocol import WorldState as RobotWorld
from . import schema
from .steps import diff_to_core, missing_args, plan_to_core, plannable

REQUEST_NAMES = {"plan": "plan", "replan": "plan_diff"}


@dataclass(frozen=True)
class PlanSnapshot:
    """The plan a replan changes, as backends may read it from `ref["episode"]` (e12v2's convention:
    its mock LLM writes a diff against `.plan` and `.queue`). A copy: backends run in other threads."""
    plan: dict
    queue: list[str]


class ManifestPlanner:
    """`backends`: route ("fast_llm", "capable_llm") -> an e12v2 LLMBackend; a missing route uses fast_llm's."""

    def __init__(self, backends: Mapping[str, object], manifest: Manifest, arm: str, retry: bool = True):
        if "fast_llm" not in backends:
            raise ValueError("the planner needs a fast_llm backend")
        self.backends, self.retry, self.arm = dict(backends), retry, arm
        self.skills = {s.name: s for s in plannable(manifest, arm)}
        self.instructions = schema.instructions(manifest, arm)

    # ---------------------------------------------------------------- schemas (for the request, and for tests)
    def step_schema(self, ids: dict) -> dict:
        return schema.step_schema(list(self.skills.values()), ids)

    def plan_schema(self, ids: dict) -> dict:
        return schema.plan_schema(list(self.skills.values()), ids)

    def diff_schema(self, ids: dict) -> dict:
        return schema.diff_schema(list(self.skills.values()), ids)

    # ---------------------------------------------------------------- requests
    def plan(self, task: str, user_messages: list[str], world: RobotWorld, core: WorldState, ref=None,
             route: str = "fast_llm") -> PlanResult:
        """The first plan of an episode (full plan), from the world as it is now."""
        body = {"task": task, "user_messages": user_messages, "world": schema.world_view(world, self.arm)}
        req = self._request("plan", body, self.plan_schema(P.schema_ids(core)))

        def check(out: dict) -> list[str]:
            return self.step_errors(out["steps"]) or P.validate(plan_to_core(out), core, core.arm.holding)

        out, attempts = self._ask(route, req, check, ref)
        return PlanResult("plan", route, out is not None, plan=plan_to_core(out) if out else None, attempts=attempts)

    def replan(self, route: str, task: str, user_messages: list[str], world: RobotWorld, core: WorldState, plan: dict,
               queue: list[str], status: dict, arm_now: dict, why: str, ref=None) -> PlanResult:
        """A decision handed an event to an LLM: only what changes (a diff against `plan`)."""
        body = {"task": task, "user_messages": user_messages, "world": schema.world_view(world, self.arm),
                "current_plan": plan_view(plan, queue, status, core), "arm_now": arm_now, "why": why}
        req = self._request("replan", body, self.diff_schema(P.schema_ids(core)))

        def check(out: dict) -> list[str]:
            errs = self.step_errors(out["new_steps"])
            if errs:
                return errs
            d = diff_to_core(out)
            errs = P.diff_errors(plan, d)
            if errs:
                return errs
            newp, q = P.apply_diff(plan, d)
            return P.validate(newp, core, core.arm.holding, q)

        out, attempts = self._ask(route, req, check, ref)
        return PlanResult("replan", route, out is not None, diff=diff_to_core(out) if out else None, attempts=attempts)

    def step_errors(self, steps: list[dict]) -> list[str]:
        """Steps (catalog names) this arm cannot run as written: an unadvertised skill, a missing argument."""
        errs = []
        for s in steps:
            spec = self.skills.get(s["skill"])
            if spec is None:
                errs.append(f"step {s['id']}: this robot has no skill {s['skill']}; use only {', '.join(self.skills)}")
                continue
            errs += [f"step {s['id']}: {s['skill']} needs {a}" for a in missing_args(s, spec)]
        return errs

    # ---------------------------------------------------------------- one request, one retry
    def _request(self, kind: str, body: dict, sch: dict) -> LLMRequest:
        return LLMRequest(kind, REQUEST_NAMES[kind], self.instructions, json.dumps({"request": kind, **body}), sch)

    def _ask(self, route: str, req: LLMRequest, check: Callable[[dict], list[str]], ref) -> tuple[dict | None, list[dict]]:
        """Send `req`; validate; on errors retry once with them (e12v2's rule). A failed call is not
        retried. Attempts are recorded in e12v2's shape, so its metrics read them."""
        be = self.backends.get(route) or self.backends["fast_llm"]
        attempts: list[dict] = []
        for attempt in range(2 if self.retry else 1):
            r = be.call(req, ref=(ref, attempt))
            out, errs = None, []
            if r.error is not None:
                errs = [f"call failed: {r.error}"]
            else:
                try:
                    out = json.loads(r.raw or "")
                    errs = check(out)
                except json.JSONDecodeError as e:
                    errs = [f"not valid JSON: {e}"]
                except (KeyError, TypeError, AttributeError) as e:     # model output is untrusted: wrong shape is an error, not a crash
                    errs = [f"not the requested shape: {e!r}"]
            attempts.append({"route": route, "kind": req.kind, "latency_ms": r.latency_ms, "in_tok": r.in_tok,
                             "out_tok": r.out_tok, "model": getattr(r, "model", ""), "errors": errs, "raw": r.raw})
            if not errs:
                return out, attempts
            if r.error is not None:
                break
            req = with_errors(req, r.raw, errs)
        return None, attempts
