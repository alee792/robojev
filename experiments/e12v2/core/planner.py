"""The Planner: an LLM with strict structured output writes the plan, and replans with a diff.

Every request shares one instruction text (instructions + skill list + plan rules + what each request
kind asks for), so the prompt prefix is identical across calls and can be cached; everything that
varies (task, world, current plan, why) goes in `input`. Object ids and places are schema enums built
from the world state. Output is validated by the generic validator (core/plan.py); an invalid answer
gets one retry with the errors. A failed call is not retried.

Request kinds:
  plan    the first plan of an episode (full plan)
  replan  a decision handed the event to an LLM: return only what changes (a diff)
  react   the always-LLM control: the LLM also picks the right-now reaction, and a diff if needed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import plan as P
from .data import RIGHT_NOW, WorldState

_SKILLS = "\n".join(f"- {v}" for v in P.SKILLS.values())
_RELS = "\n".join(f"  - {k}: {v}" for k, v in P.RELATIONS.items())
_CONS = "\n".join(f"  - {k}: {v}" for k, v in P.CONSTRAINT_KINDS.items())

INSTRUCTIONS = f"""You plan for a robot arm working on a table. The input's `world` lists every object (id, name, colour, number, size, where it is, x/y in cm: x grows to the robot's right, y grows away from the robot) and every place (tray slots that hold one object each, bins that hold any number, "table" for a free spot on the table, "person" for handing an object to the person, and "tray" meaning any tray slot, for conditions and constraints only).

The arm's skills are motor capabilities only; they do not know the task:
{_SKILLS}

A plan has:
- steps: skill calls with their objects filled in, run in order. Put an object into a one-object place only when that place is empty at that point in the plan; move whatever is in the way first (to "table" if it has nowhere else to go). Do not move objects the task does not ask about unless they are in the way.
- done_when: one condition per fact that must hold when the task is done, e.g. "block_5 is in tray_slot_3". Use a relation code can check:
{_RELS}
- constraints: rules that hold for the whole task (e.g. "never touch the blue cup", "keep the marker out of the left bin"):
{_CONS}
  Steps must never break a constraint. If an object that must be picked up is so close to a dont_touch object that the gripper would touch it, push the object away from the dont_touch object first.
- If `world.arm.holding` names an object, the first step must put that object somewhere.

The input's `request` says what to return:
- "plan": the full plan for `task` (and anything in `user_messages`) from the world as it is now. Step ids s1, s2, ...; condition ids d1, d2, ...; constraint ids c1, c2, ....
- "replan": the robot is partway through `current_plan` when `why` happened. Work out the final arrangement now wanted (the task as the user's latest messages have changed it) and return ONLY what changes: new steps (new ids n1, n2, ... not used before), the full order of the steps still to run by id (kept and new; a step not listed will not run; a step already done can be listed again to run it again), and the done conditions and constraints to drop or add. Blocks already moved can be moved again. `arm_now` says what the arm is doing; if it is carrying on, keep its current step first when it is still right.
- "react": as "replan", and also choose what the arm should do right now (right_now) about `why`: carry_on (no change), hold (stop at a safe point while the plan changes), pause (stop until told to go on: a hand coming toward the gripper or in its path, or the user asks to wait), back_off (move away from a hand very close to the gripper, then stop), re_target (aim at where the current step's object is now), resume (start again after a pause or hold). Set change_plan to false and leave the plan fields empty when the plan needs no change.

Work the arrangement out carefully: the robot follows the plan exactly."""


@dataclass
class LLMRequest:          # same fields as e13_branches.planner.LLMRequest, so E13's OpenAI backend can send it
    kind: str
    name: str
    instructions: str
    input: str
    schema: dict


@dataclass
class PlanResult:
    kind: str
    route: str
    ok: bool
    plan: dict | None = None          # full plan (kind "plan")
    diff: dict | None = None          # replan / react
    right_now: str | None = None      # react
    change_plan: bool = True
    attempts: list = field(default_factory=list)
    t_request: int = 0

    @property
    def latency_ms(self) -> float:
        return sum(a["latency_ms"] for a in self.attempts)

    @property
    def in_tok(self) -> int:
        return sum(a["in_tok"] for a in self.attempts)

    @property
    def out_tok(self) -> int:
        return sum(a["out_tok"] for a in self.attempts)


def world_view(state: WorldState, names: dict) -> dict:
    return {
        "objects": [{"id": o.id, "name": o.name, "colour": o.colour, "number": o.number, "size": o.size,
                     "where": o.where, "x": round(o.x), "y": round(o.y)} for o in state.objects.values()],
        "places": [{"id": p.id, "name": p.name, "kind": p.kind, "holds": "one object" if p.capacity == 1 else "any number"}
                   for p in state.places.values()],
        "arm": {"holding": state.arm.holding or "nothing"},
    }


def plan_view(plan: dict, queue: list, status: dict, state: WorldState) -> dict:
    return {
        "reading": plan.get("reading", ""),
        "steps": [{**s, "status": status.get(s["id"], "dropped")} for s in plan["steps"]
                  if status.get(s["id"]) in ("done", "running", "failed") or s["id"] in queue],
        "pending_order": list(queue),
        "done_when": [{**c, "true_now": P.measure(c, state)} for c in plan["done_when"]],
        "constraints": plan["constraints"],
    }


def _req(kind: str, body: dict, schema: dict) -> LLMRequest:
    return LLMRequest(kind, {"plan": "plan", "replan": "plan_diff", "react": "reaction"}[kind], INSTRUCTIONS,
                      json.dumps({"request": kind, **body}), schema)


def plan_request(task: str, user_messages: list, state: WorldState, names: dict) -> LLMRequest:
    return _req("plan", {"task": task, "user_messages": user_messages, "world": world_view(state, names)},
                P.plan_schema(P.schema_ids(state)))


def replan_request(kind: str, task: str, user_messages: list, state: WorldState, names: dict, plan: dict, queue: list,
                   status: dict, arm_now: dict, why: str) -> LLMRequest:
    body = {"task": task, "user_messages": user_messages, "world": world_view(state, names),
            "current_plan": plan_view(plan, queue, status, state), "arm_now": arm_now, "why": why}
    ids = P.schema_ids(state)
    return _req(kind, body, P.react_schema(ids, RIGHT_NOW) if kind == "react" else P.diff_schema(ids))


def with_errors(req: LLMRequest, raw: str | None, errors: list) -> LLMRequest:
    extra = ("\n\nYour previous answer was:\n" + (raw or "(no output)") +
             "\n\nIt was rejected for these reasons; fix them and answer again:\n- " + "\n- ".join(errors))
    return LLMRequest(req.kind, req.name, req.instructions, req.input + extra, req.schema)


class PlannerClient:
    """Sends plan/replan/react requests to the backend for a route, validates, retries once."""

    def __init__(self, backends: dict, retry: bool = True):
        self.backends, self.retry = backends, retry     # route -> LLMBackend ("fast_llm", "capable_llm")

    def _run(self, route: str, req: LLMRequest, check, ref) -> tuple[dict | None, list]:
        be = self.backends.get(route) or self.backends["fast_llm"]
        attempts = []
        for attempt in range(2 if self.retry else 1):
            r = be.call(req, ref=(ref, attempt))
            out, errs = None, []
            if r.error is not None:
                errs = [f"call failed: {r.error}"]
            else:
                try:
                    out = json.loads(r.raw or "")
                except json.JSONDecodeError as e:
                    errs = [f"not valid JSON: {e}"]
                if out is not None:
                    errs = check(out)
            attempts.append({"route": route, "kind": req.kind, "latency_ms": r.latency_ms, "in_tok": r.in_tok, "out_tok": r.out_tok,
                             "cached_tok": getattr(r, "cached_tok", 0) or 0, "reasoning_tok": getattr(r, "reasoning_tok", 0) or 0,
                             "model": r.model, "errors": errs, "raw": r.raw})
            if not errs:
                return out, attempts
            if r.error is not None:
                return None, attempts
            req = with_errors(req, r.raw, errs)
        return None, attempts

    def plan(self, task: str, user_messages: list, state: WorldState, names: dict, ref=None, route: str = "fast_llm") -> PlanResult:
        req = plan_request(task, user_messages, state, names)
        out, att = self._run(route, req, lambda p: P.validate(p, state, state.arm.holding), ref)
        return PlanResult("plan", route, out is not None, plan=out, attempts=att)

    def replan(self, kind: str, route: str, task: str, user_messages: list, state: WorldState, names: dict, plan: dict,
               queue: list, status: dict, arm_now: dict, why: str, ref=None) -> PlanResult:
        req = replan_request(kind, task, user_messages, state, names, plan, queue, status, arm_now, why)

        def check(d):
            if kind == "react" and not d.get("change_plan", True):
                return []
            errs = P.diff_errors(plan, d)
            if errs:
                return errs
            newp, q = P.apply_diff(plan, d)
            return P.validate(newp, state, state.arm.holding, q)

        out, att = self._run(route, req, check, ref)
        if out is None:
            return PlanResult(kind, route, False, attempts=att)
        return PlanResult(kind, route, True, diff=out, right_now=out.get("right_now"), change_plan=out.get("change_plan", True),
                          attempts=att)
