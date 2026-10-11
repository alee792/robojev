"""The planner client: e12v2's plan and replan requests, built from the connected robot's manifest.

Requests carry e12v2's shape (instructions, a JSON input, a strict schema; see e12v2.core.planner)
so any e12v2 LLM backend can send them; what changes is that the instructions describe this robot
and the step schema lists only its skills (schema.py). An answer is checked in three layers, and an
invalid one gets one retry with the errors, as in e12v2:
  1. shape and manifest: every step's skill is plannable on this arm, with its required arguments;
  2. translated to e12v2's step names (steps.py), e12v2's diff checks and generic validator.
The result handed back to the loop is in e12v2's names, ready for its plan functions.

Several arms (one planner for all of them, docs/widowx-pair.md "Crawl"): every step names its arm
(schema.py), layer 1 also checks the skill is plannable on that arm, and layer 2 runs e12v2's
validator with no one arm's holding rule, then arms.py's checks: each arm holding an object puts it
somewhere first, and no step goes to an arm that cannot reach its object or destination. The
instructions then carry what each arm reaches now, so they are built per request.

Calls block (seconds, for a real model): the loop runs them in a thread and never waits on them.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .core import plan as P
from .core.data import WorldState
from .core.planner import LLMRequest, PlanResult, plan_view, with_errors

from ..protocol import Manifest
from ..protocol import WorldState as RobotWorld
from . import schema
from .arms import holding_errors, reach_errors
from .steps import diff_to_core, missing_args, plan_to_core, plannable, plannable_on

REQUEST_NAMES = {"plan": "plan", "replan": "plan_diff"}


@dataclass(frozen=True)
class PlanSnapshot:
    """The plan a replan changes, as backends may read it from `ref["episode"]` (e12v2's convention:
    its mock LLM writes a diff against `.plan` and `.queue`). A copy: backends run in other threads."""
    plan: dict
    queue: list[str]


class ManifestPlanner:
    """`backends`: route ("fast_llm", "capable_llm") -> an e12v2 LLMBackend; a missing route uses fast_llm's.
    `arms`: the arm, or arms, plans are for (several: every step names one)."""

    def __init__(self, backends: Mapping[str, object], manifest: Manifest, arms: str | Sequence[str], retry: bool = True):
        if "fast_llm" not in backends:
            raise ValueError("the planner needs a fast_llm backend")
        arms = (arms,) if isinstance(arms, str) else tuple(arms)
        self.backends, self.retry, self.manifest = dict(backends), retry, manifest
        self.arms, self.arm, self.multi = arms, arms[0], len(arms) > 1
        self.specs = {a.id: a for a in manifest.arms if a.id in arms}
        self.skills = {s.name: s for s in plannable_on(manifest, arms)}
        self.on_arm = {a: {s.name for s in plannable(manifest, a)} for a in arms}
        self.instructions = None if self.multi else schema.instructions(manifest, self.arm)

    # ---------------------------------------------------------------- schemas (for the request, and for tests)
    @property
    def _arm_enum(self) -> tuple[str, ...]:
        return self.arms if self.multi else ()

    def step_schema(self, ids: dict) -> dict:
        return schema.step_schema(list(self.skills.values()), ids, self._arm_enum)

    def plan_schema(self, ids: dict) -> dict:
        return schema.plan_schema(list(self.skills.values()), ids, self._arm_enum)

    def diff_schema(self, ids: dict) -> dict:
        return schema.diff_schema(list(self.skills.values()), ids, self._arm_enum)

    def _world(self, world: RobotWorld) -> dict:
        return schema.world_view_arms(world, self.specs) if self.multi else schema.world_view(world, self.arm)

    def check(self, plan: dict, core: WorldState, world: RobotWorld, queue: list[str] | None = None) -> list[str]:
        """Errors in a plan in e12v2's names (with `queue`: the steps still to run, in order). One arm:
        e12v2's validator with that arm's holding rule. Several: the validator, then the holding rule
        per arm and reach (arms.py)."""
        if not self.multi:
            return P.validate(plan, core, core.arm.holding, queue)
        errs = P.validate(plan, core, None, queue)
        if errs:
            return errs
        byid = P.steps_by_id(plan)
        order = [byid[sid] for sid in (queue if queue is not None else [s["id"] for s in plan["steps"]])]
        return (holding_errors(order, {a: world.arms[a].holding for a in self.arms}) +
                reach_errors(order, world, self.specs))

    # ---------------------------------------------------------------- requests
    def plan(self, task: str, user_messages: list[str], world: RobotWorld, core: WorldState, ref=None,
             route: str = "fast_llm") -> PlanResult:
        """The first plan of an episode (full plan), from the world as it is now."""
        body = {"task": task, "user_messages": user_messages, "world": self._world(world)}
        req = self._request("plan", body, self.plan_schema(P.schema_ids(core)), world)

        def check(out: dict) -> list[str]:
            return self.step_errors(out["steps"]) or self.check(plan_to_core(out), core, world)

        out, attempts = self._ask(route, req, check, ref)
        return PlanResult("plan", route, out is not None, plan=plan_to_core(out) if out else None, attempts=attempts)

    def replan(self, route: str, task: str, user_messages: list[str], world: RobotWorld, core: WorldState, plan: dict,
               queue: list[str], status: dict, arm_now: dict, why: str, ref=None) -> PlanResult:
        """A decision handed an event to an LLM: only what changes (a diff against `plan`)."""
        body = {"task": task, "user_messages": user_messages, "world": self._world(world),
                "current_plan": plan_view(plan, queue, status, core), "arm_now": arm_now, "why": why}
        req = self._request("replan", body, self.diff_schema(P.schema_ids(core)), world)

        def check(out: dict) -> list[str]:
            errs = self.step_errors(out["new_steps"])
            if errs:
                return errs
            d = diff_to_core(out)
            errs = P.diff_errors(plan, d)
            if errs:
                return errs
            newp, q = P.apply_diff(plan, d)
            return self.check(newp, core, world, q)

        out, attempts = self._ask(route, req, check, ref)
        return PlanResult("replan", route, out is not None, diff=diff_to_core(out) if out else None, attempts=attempts)

    def step_errors(self, steps: list[dict]) -> list[str]:
        """Steps (catalog names) that cannot run as written: an unadvertised skill, a missing argument;
        with several arms, a step naming no arm of this robot or one without the skill."""
        errs = []
        for s in steps:
            spec = self.skills.get(s["skill"])
            if spec is None:
                errs.append(f"step {s['id']}: this robot has no skill {s['skill']}; use only {', '.join(self.skills)}")
                continue
            if self.multi:
                arm = s.get("arm")
                if arm not in self.on_arm:
                    errs.append(f"step {s['id']}: arm must be one of {', '.join(self.arms)}, not {arm!r}")
                    continue
                if s["skill"] not in self.on_arm[arm]:
                    errs.append(f"step {s['id']}: {arm} has no skill {s['skill']}; it can use {', '.join(sorted(self.on_arm[arm]))}")
                    continue
            errs += [f"step {s['id']}: {s['skill']} needs {a}" for a in missing_args(s, spec)]
        return errs

    # ---------------------------------------------------------------- one request, one retry
    def _request(self, kind: str, body: dict, sch: dict, world: RobotWorld) -> LLMRequest:
        text = self.instructions if not self.multi else schema.instructions(self.manifest, None, world)
        return LLMRequest(kind, REQUEST_NAMES[kind], text, json.dumps({"request": kind, **body}), sch)

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
