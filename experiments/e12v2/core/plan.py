"""The plan: skill-call steps, done conditions and constraints (docs/v2.md "The plan").

A full plan (the first one of an episode) is
  {"reading": str,
   "steps":       [{"id", "skill", "object", "target", "direction"}],   # run in this order
   "done_when":   [{"id", "text", "object", "relation", "target"}],
   "constraints": [{"id", "text", "kind", "object", "place"}]}
A replan returns only what changes (a diff):
  {"reading": str, "new_steps": [...], "pending_order": [step ids still to run, in order; kept and new],
   "drop_done_when": [ids], "new_done_when": [...], "drop_constraints": [ids], "new_constraints": [...]}
Object ids and places are schema enums built from the world state. Strict JSON schema: every property
required, no extra keys, "none" where a field does not apply.

The validator is generic: it replays the steps on a symbolic copy of the world (where each object is)
and checks shapes, capacities, stacks, constraints, and that the measurable done conditions come out
true. Nothing here knows what any task means.
"""

from __future__ import annotations

import copy

from .data import WorldState

SKILLS = {
    "move_object": "move_object(object, target place): pick the object up, carry it and put it down in the place (a tray slot, a bin, or \"table\" for a free spot on the table).",
    "hand_over": "hand_over(object): pick the object up and hand it to the person (target \"person\"); done when the person has taken it.",
    "stack_on": "stack_on(object, target object): pick the object up and set it on top of the target object.",
    "push": "push(object, direction): slide the object about 6 cm in a direction (left, right, toward_robot, away_from_robot; the robot's point of view) with one fingertip, without grasping it.",
    "survey": "survey(): lift the gripper and look over the table again.",
    "hold": "hold(): stay still for about a second.",
}
DIRECTIONS = ("left", "right", "toward_robot", "away_from_robot")
RELATIONS = {
    "in": "the object is in the target place (a slot, a bin, \"tray\" for any tray slot, \"table\", or \"person\" = the person is holding it)",
    "on": "the object sits directly on top of the target object",
    "not_in": "the object is not in the target place",
    "other": "anything else (then object and target may be \"none\"; the text says it all)",
}
CONSTRAINT_KINDS = {
    "dont_touch": "the robot must never touch or move the object",
    "keep_out_of": "the object must never be put into the place (\"tray\" = any tray slot); if it gets there, it must be taken out",
}
NONE = "none"


# ---------------------------------------------------------------- ids for the schema enums


def schema_ids(state: WorldState) -> dict:
    return {"objects": sorted(state.objects), "places": sorted(state.places)}


def _step_schema(ids: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["id", "skill", "object", "target", "direction"],
        "properties": {
            "id": {"type": "string", "description": "short unique id, e.g. s1 (new ids in a replan: n1, n2, ...)"},
            "skill": {"type": "string", "enum": list(SKILLS)},
            "object": {"type": "string", "enum": ids["objects"] + [NONE]},
            "target": {"type": "string", "enum": ids["places"] + ids["objects"] + [NONE],
                       "description": "a place for move_object, \"person\" for hand_over, an object for stack_on, else none"},
            "direction": {"type": "string", "enum": list(DIRECTIONS) + [NONE], "description": "push only, else none"},
        },
    }


def _cond_schema(ids: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["id", "text", "object", "relation", "target"],
        "properties": {
            "id": {"type": "string", "description": "short unique id, e.g. d1"},
            "text": {"type": "string", "description": "the condition as a plain statement, e.g. \"block_5 is in tray_slot_3\""},
            "object": {"type": "string", "enum": ids["objects"] + [NONE]},
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "target": {"type": "string", "enum": ids["places"] + ids["objects"] + [NONE]},
        },
    }


def _constraint_schema(ids: dict) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["id", "text", "kind", "object", "place"],
        "properties": {
            "id": {"type": "string", "description": "short unique id, e.g. c1"},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": list(CONSTRAINT_KINDS)},
            "object": {"type": "string", "enum": ids["objects"]},
            "place": {"type": "string", "enum": ids["places"] + [NONE], "description": "keep_out_of only, else none"},
        },
    }


def plan_schema(ids: dict) -> dict:
    arr = lambda items, d: {"type": "array", "items": items, "description": d}  # noqa: E731
    return {
        "type": "object", "additionalProperties": False,
        "required": ["reading", "steps", "done_when", "constraints"],
        "properties": {
            "reading": {"type": "string", "description": "one sentence: the final arrangement the task asks for"},
            "steps": arr(_step_schema(ids), "the steps, in the order to run them"),
            "done_when": arr(_cond_schema(ids), "one condition per fact that must hold when the task is done"),
            "constraints": arr(_constraint_schema(ids), "rules that hold for the whole task; empty if none"),
        },
    }


def diff_schema(ids: dict) -> dict:
    arr = lambda items, d: {"type": "array", "items": items, "description": d}  # noqa: E731
    s = {"type": "string"}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["reading", "new_steps", "pending_order", "drop_done_when", "new_done_when", "drop_constraints", "new_constraints"],
        "properties": {
            "reading": {"type": "string", "description": "one sentence: what changes and the final arrangement now wanted"},
            "new_steps": arr(_step_schema(ids), "only steps that are not already in the plan"),
            "pending_order": arr(s, "ids of every step still to run, in order (kept steps and new ones); a step not listed will not run"),
            "drop_done_when": arr(s, "ids of done conditions that no longer apply"),
            "new_done_when": arr(_cond_schema(ids), "only conditions that are not already in the plan"),
            "drop_constraints": arr(s, "ids of constraints that no longer apply"),
            "new_constraints": arr(_constraint_schema(ids), "only constraints that are not already in the plan"),
        },
    }


def react_schema(ids: dict, right_now: tuple) -> dict:
    """The always-LLM control: the LLM also picks the reaction; the plan diff is applied if change_plan."""
    d = diff_schema(ids)
    d["required"] = ["right_now", "change_plan"] + d["required"]
    d["properties"] = {"right_now": {"type": "string", "enum": list(right_now)},
                       "change_plan": {"type": "boolean"}, **d["properties"]}
    return d


# ---------------------------------------------------------------- reading a plan


def steps_by_id(plan: dict) -> dict:
    return {s["id"]: s for s in plan["steps"]}


def step_text(step: dict, names: dict | None = None) -> str:
    n = (names or {}).get
    sk, o, t = step["skill"], step["object"], step["target"]
    on, tn = n(o, o), n(t, t)
    if sk == "move_object":
        return f"move_object({on} → {tn})"
    if sk == "stack_on":
        return f"stack_on({on} on {tn})"
    if sk == "hand_over":
        return f"hand_over({on} to the person)"
    if sk == "push":
        return f"push({on} {step['direction']})"
    return f"{sk}()"


def step_goal(step: dict) -> tuple | None:
    """(object, where it ends up) for a step that moves an object."""
    sk = step["skill"]
    if sk == "move_object":
        return step["object"], step["target"]
    if sk == "stack_on":
        return step["object"], f"on:{step['target']}"
    if sk == "hand_over":
        return step["object"], "person"
    return None


def mentioned(plan: dict) -> set:
    ids = set()
    for s in plan["steps"]:
        ids |= {s["object"], s["target"]}
    for c in plan["done_when"]:
        ids |= {c["object"], c["target"]}
    for c in plan["constraints"]:
        ids.add(c["object"])
    return ids - {NONE}


def measure(cond: dict, state: WorldState) -> bool | None:
    """Code checks what it can measure; None means ask Jev (relation "other")."""
    rel, o, t = cond["relation"], cond["object"], cond["target"]
    if rel == "other" or o not in state.objects:
        return None
    if rel == "on":
        return state.objects[o].where == f"on:{t}"
    inside = state.in_place(o, t) if t in state.places else state.objects[o].where == f"on:{t}"
    return inside if rel == "in" else not inside


def forbidden(plan: dict) -> set:
    return {c["object"] for c in plan["constraints"] if c["kind"] == "dont_touch"}


def keep_out(plan: dict) -> list[tuple[str, str]]:
    return [(c["object"], c["place"]) for c in plan["constraints"] if c["kind"] == "keep_out_of"]


# ---------------------------------------------------------------- validation


class _Sym:
    """A symbolic world: where each object is, with capacities, groups and stacks."""

    def __init__(self, state: WorldState, holding: str | None):
        self.state = state
        self.where = {o.id: o.where for o in state.objects.values()}
        if holding:
            self.where[holding] = "gripper"

    def occupant(self, place: str, but: str) -> str | None:
        return next((o for o, w in self.where.items() if w == place and o != but), None)

    def covered(self, oid: str, but: str) -> str | None:
        return next((o for o, w in self.where.items() if w == f"on:{oid}" and o != but), None)

    def inside(self, oid: str, place: str) -> bool:
        p = self.state.places.get(place)
        w = self.where.get(oid)
        if p is not None and p.kind == "group":
            return w in p.members
        return w == place

    def measure(self, cond: dict) -> bool | None:
        rel, o, t = cond["relation"], cond["object"], cond["target"]
        if rel == "other" or o not in self.where:
            return None
        if rel == "on":
            return self.where[o] == f"on:{t}"
        inside = self.inside(o, t) if t in self.state.places else self.where[o] == f"on:{t}"
        return inside if rel == "in" else not inside


def _check_step(s: dict, st: WorldState, errs: list):
    sk, o, t, d = s["skill"], s["object"], s["target"], s["direction"]
    objs, places = st.objects, st.places
    if sk not in SKILLS:
        errs.append(f"step {s['id']}: unknown skill {sk}")
        return False
    if sk in ("survey", "hold"):
        if o != NONE or t != NONE:
            errs.append(f"step {s['id']}: {sk} takes no object or target")
        return True
    if o not in objs:
        errs.append(f"step {s['id']}: unknown object {o}")
        return False
    if sk == "move_object" and not (t in places and places[t].kind in ("slot", "bin", "table")):
        errs.append(f"step {s['id']}: move_object needs a slot, a bin or table as target, not {t}")
        return False
    if sk == "stack_on" and (t not in objs or t == o):
        errs.append(f"step {s['id']}: stack_on needs another object as target, not {t}")
        return False
    if sk == "hand_over" and t not in ("person", NONE):
        errs.append(f"step {s['id']}: hand_over's target is the person")
        return False
    if sk == "push" and d not in DIRECTIONS:
        errs.append(f"step {s['id']}: push needs a direction")
        return False
    return True


def validate(plan: dict, state: WorldState, holding: str | None = None, queue: list | None = None) -> list[str]:
    """Errors in a full plan (or, with `queue`, in the plan that results from a diff). Empty = valid."""
    errs: list[str] = []
    objs, places = state.objects, state.places
    ids = [s["id"] for s in plan["steps"]]
    if len(set(ids)) != len(ids):
        errs.append("step ids must be unique")
    for key in ("done_when", "constraints"):
        cids = [c["id"] for c in plan[key]]
        if len(set(cids)) != len(cids):
            errs.append(f"{key} ids must be unique")
    if not plan["done_when"]:
        errs.append("the plan needs at least one done condition")
    for c in plan["constraints"]:
        if c["object"] not in objs:
            errs.append(f"constraint {c['id']}: unknown object {c['object']}")
        if c["kind"] == "keep_out_of" and c["place"] not in places:
            errs.append(f"constraint {c['id']}: keep_out_of needs a place")
    for c in plan["done_when"]:
        if c["relation"] != "other" and c["object"] not in objs:
            errs.append(f"done condition {c['id']}: unknown object {c['object']}")
        if c["relation"] != "other" and c["target"] not in places and c["target"] not in objs:
            errs.append(f"done condition {c['id']}: unknown target {c['target']}")
    if errs:
        return errs
    by_id = steps_by_id(plan)
    order = queue if queue is not None else ids
    missing = [i for i in order if i not in by_id]
    if missing:
        return [f"pending_order names steps that do not exist: {', '.join(missing)}"]
    forb = forbidden(plan)
    kout = keep_out(plan)
    sym = _Sym(state, holding)
    if holding and order:
        first = by_id[order[0]]
        if first["object"] != holding or first["skill"] not in ("move_object", "stack_on", "hand_over"):
            errs.append(f"the arm is holding {holding}: the first step must put it somewhere (move_object, stack_on or hand_over of {holding})")
    elif holding and not order:
        errs.append(f"the arm is holding {holding}: the plan needs a step that puts it somewhere")
    for sid in order:
        s = by_id[sid]
        if not _check_step(s, state, errs):
            continue
        o, t = s["object"], s["target"]
        if s["skill"] in ("survey", "hold"):
            continue
        if o in forb:
            errs.append(f"step {sid}: {o} must not be touched (constraint)")
            continue
        top = sym.covered(o, but=NONE)
        if top and s["skill"] != "push":
            errs.append(f"step {sid}: {o} has {top} on top of it at that point")
        if s["skill"] == "move_object":
            p = places[t]
            if p.capacity == 1:
                occ = sym.occupant(t, but=o)
                if occ:
                    errs.append(f"step {sid}: {t} is still occupied by {occ} when this step puts {o} there")
            for ko, kp in kout:
                if ko == o and (t == kp or t in getattr(places.get(kp), "members", ())):
                    errs.append(f"step {sid}: puts {o} into {t}, which a constraint forbids")
            sym.where[o] = t
        elif s["skill"] == "stack_on":
            occ = sym.covered(t, but=o)
            if occ:
                errs.append(f"step {sid}: {t} already has {occ} on top of it")
            if sym.where.get(t) in ("person", "gripper"):
                errs.append(f"step {sid}: {t} is not on the table")
            sym.where[o] = f"on:{t}"
        elif s["skill"] == "hand_over":
            sym.where[o] = "person"
    for c in plan["done_when"]:
        v = sym.measure(c)
        if v is False:
            errs.append(f"after the plan, done condition {c['id']} ({c['text']}) would still be false")
    for ko, kp in kout:
        if sym.inside(ko, kp):
            errs.append(f"after the plan, {ko} would still be in {kp}, which a constraint forbids")
    return errs


# ---------------------------------------------------------------- diffs


def apply_diff(plan: dict, diff: dict) -> tuple[dict, list]:
    """-> (new plan, new queue). Steps not in pending_order stay in the plan's history but won't run."""
    p = copy.deepcopy(plan)
    p["reading"] = diff.get("reading") or p.get("reading", "")
    p["steps"] = p["steps"] + [dict(s) for s in diff["new_steps"]]
    drop = set(diff["drop_done_when"])
    p["done_when"] = [c for c in p["done_when"] if c["id"] not in drop] + [dict(c) for c in diff["new_done_when"]]
    dropc = set(diff["drop_constraints"])
    p["constraints"] = [c for c in p["constraints"] if c["id"] not in dropc] + [dict(c) for c in diff["new_constraints"]]
    return p, list(diff["pending_order"])


def diff_errors(plan: dict, diff: dict) -> list[str]:
    old = {s["id"] for s in plan["steps"]}
    errs = [f"new step id {s['id']} is already used" for s in diff["new_steps"] if s["id"] in old]
    for key, drop in (("done_when", "drop_done_when"), ("constraints", "drop_constraints")):
        have = {c["id"] for c in plan[key]}
        errs += [f"{drop}: no {key} entry {i}" for i in diff[drop] if i not in have]
        errs += [f"new {key} id {c['id']} is already used" for c in diff["new_" + key] if c["id"] in have]
    return errs


def _sig(s: dict) -> tuple:
    return (s["skill"], s["object"], s["target"], s["direction"])


def make_diff(plan: dict, queue: list, new_plan: dict, reading: str = "") -> dict:
    """The smallest diff that turns (plan, queue) into new_plan's steps / conditions / constraints.
    Used by the mocks (and handy for tests); an LLM writes its own."""
    by_id = steps_by_id(plan)
    reuse = {}
    for sid in queue:
        reuse.setdefault(_sig(by_id[sid]), []).append(sid)
    used = {s["id"] for s in plan["steps"]}
    new_steps, order, k = [], [], 1
    for s in new_plan["steps"]:
        cand = reuse.get(_sig(s))
        if cand:
            order.append(cand.pop(0))
            continue
        while f"n{k}" in used:
            k += 1
        nid = f"n{k}"
        used.add(nid)
        new_steps.append({**s, "id": nid})
        order.append(nid)

    def cdiff(key, sig):
        old = {sig(c): c["id"] for c in plan[key]}
        keep = {sig(c) for c in new_plan[key]}
        have = {c["id"] for c in plan[key]}
        drop = [cid for sg, cid in old.items() if sg not in keep]
        new, j = [], 1
        for c in new_plan[key]:
            if sig(c) in old:
                continue
            while f"{key[0]}{j}x" in have:
                j += 1
            nid = f"{key[0]}{j}x"
            have.add(nid)
            new.append({**c, "id": nid})
        return drop, new

    dd, nd = cdiff("done_when", lambda c: (c["object"], c["relation"], c["target"]))
    dc, nc = cdiff("constraints", lambda c: (c["kind"], c["object"], c["place"]))
    return {"reading": reading or new_plan.get("reading", ""), "new_steps": new_steps, "pending_order": order,
            "drop_done_when": dd, "new_done_when": nd, "drop_constraints": dc, "new_constraints": nc}


class StepListFormat:
    """The step-list plan format (the only one built). See core/interfaces.PlanFormat."""
    name = "steps"

    def schema(self, ids):
        return plan_schema(ids)

    def diff_schema(self, ids):
        return diff_schema(ids)

    def validate(self, plan, state, holding=None, queue=None):
        return validate(plan, state, holding, queue)

    def apply_diff(self, plan, diff):
        return apply_diff(plan, diff)
