"""The planner's side of the manifest: what the robot is and which steps it can be asked for.

The step format stays e12v2's flat one ({id, skill, object, target, direction}), so its validator,
diffs, decision text and mocks work unchanged; what changes is that every enum and description in the
step schema comes from manifest().skills for the arm being driven. A skill the server does not
advertise is not in the `skill` enum, so a strict-output planner cannot write it; a planner that does
anyway is rejected by `manifest_errors` and retried. A field no skill uses collapses to "none".

The plan rules (done conditions, constraints, replans as diffs) are e12v2's instruction text, reused
from "A plan has:" on; the head (what the robot is, how the world is described) is written here,
because e12v2's head described its cm, x-right/y-away toy frame and a fixed skill list.
"""
from __future__ import annotations

import json

from e12v2.core import plan as P
from e12v2.core.data import WorldState
from e12v2.core.planner import INSTRUCTIONS, LLMRequest, PlannerClient, PlanResult, plan_view

from ..protocol import Manifest, SkillSpec
from ..protocol import WorldState as PWorld

STEP_FIELDS = ("object", "target", "direction")     # a skill's args must be a subset: the step format carries these


def arm_skills(m: Manifest, arm: str) -> list[SkillSpec]:
    return [s for s in m.skills if not s.arms or arm in s.arms]


def skill_args(s: SkillSpec) -> list[str]:
    return list(s.args_schema.get("properties", {}))


def check_manifest(m: Manifest):
    for s in m.skills:
        extra = [a for a in skill_args(s) if a not in STEP_FIELDS]
        if extra:
            raise ValueError(f"skill {s.name}: args {extra} have no field in the step format ({', '.join(STEP_FIELDS)})")


def step_args(step: dict, s: SkillSpec) -> dict:
    """A plan step -> the protocol's args for start(): only the fields the skill declares."""
    return {k: step[k] for k in skill_args(s)}


def describe_robot(m: Manifest) -> str:
    """The manifest as a few plain sentences for the planner's prompt."""
    n = len(m.arms)
    lines = [f"The robot ({m.robot}) has {n} arm{'s' if n != 1 else ''}: {', '.join(a.id for a in m.arms)}. "
             "Positions are in metres in each arm's base frame (+x forward, +y left, +z up)."]
    for a in m.arms:
        (x0, x1), (y0, y1), (z0, z1) = a.workspace
        g = a.gripper
        grip = ("no gripper" if g is None else
                f"a gripper that opens to {g.max_width * 100:.1f} cm and can grasp" if g.can_grasp else
                "an end effector that cannot grasp: it can only touch and push")
        lines.append(f"{a.id} has {grip}. It reaches x {x0:.2f} to {x1:.2f} m, y {y0:.2f} to {y1:.2f} m, "
                     f"z {z0:.2f} to {z1:.2f} m, and travels {a.travel_z:.2f} m up between places.")
    lines.append("Its skills, the only steps a plan may use:")
    for s in m.skills:
        who = f" (arms: {', '.join(s.arms)})" if s.arms else ""
        lines.append(f"- {s.name}({', '.join(skill_args(s))}): {s.description}{who}")
    return "\n".join(lines)


def instructions(m: Manifest) -> str:
    head = ("You plan for a robot working on a table.\n\n" + describe_robot(m) + "\n\n"
            "The input's `world` lists every object (id, name, the label read off it, size in m, where it is, x/y in m) "
            "and every place (tray slots that hold one object each, bins that hold any number, \"table\" for a free spot on "
            "the table, \"person\" for handing an object to the person, and a group such as \"tray\" meaning any of its slots, "
            "for conditions and constraints only). The skills are motor capabilities only; they do not know the task.\n\n")
    i = INSTRUCTIONS.index("A plan has:")
    return head + INSTRUCTIONS[i:]


def step_schema(m: Manifest, arm: str, ids: dict) -> dict:
    skills = arm_skills(m, arm)

    def users(f):
        return [(s.name, s.args_schema["properties"][f]) for s in skills if f in skill_args(s)]

    def desc(f):
        u = users(f)
        return "; ".join(f"{n}: {p.get('description', f)}" for n, p in u) + ("; else none" if u else "no skill uses it: none")

    target = ids["places"] + ids["objects"] if users("target") else []
    dirs = sorted({d for _, p in users("direction") for d in p.get("enum", [])})
    return {
        "type": "object", "additionalProperties": False,
        "required": ["id", "skill", *STEP_FIELDS],
        "properties": {
            "id": {"type": "string", "description": "short unique id, e.g. s1 (new ids in a replan: n1, n2, ...)"},
            "skill": {"type": "string", "enum": [s.name for s in skills]},
            "object": {"type": "string", "enum": (ids["objects"] if users("object") else []) + [P.NONE], "description": desc("object")},
            "target": {"type": "string", "enum": target + [P.NONE], "description": desc("target")},
            "direction": {"type": "string", "enum": dirs + [P.NONE], "description": desc("direction")},
        },
    }


def plan_schema(m: Manifest, arm: str, ids: dict) -> dict:
    s = P.plan_schema(ids)
    s["properties"]["steps"]["items"] = step_schema(m, arm, ids)
    return s


def diff_schema(m: Manifest, arm: str, ids: dict) -> dict:
    s = P.diff_schema(ids)
    s["properties"]["new_steps"]["items"] = step_schema(m, arm, ids)
    return s


def world_view(ws: PWorld, arm: str) -> dict:
    """What the planner reads of the world: protocol-native, metres."""
    w = lambda x: "gripper" if x == f"gripper:{arm}" else x  # noqa: E731
    return {
        "objects": [{"id": o.id, "name": o.name, "label": o.label, "size_m": o.size, "where": w(o.where),
                     "x": round(o.x, 3), "y": round(o.y, 3)} for o in ws.objects.values()],
        "places": [{"id": p.id, "name": p.name, "kind": p.kind, "holds": "one object" if p.capacity == 1 else "any number"}
                   for p in ws.places.values()],
        "arm": {"holding": ws.arms[arm].holding or "nothing"},
    }


class ManifestPlanner(PlannerClient):
    """e12v2's PlannerClient (backends per route, validate, one retry with the errors), with requests
    built from the manifest. `world` is the protocol state the planner reads; `core` the cm adapter the
    validator replays steps on."""

    def __init__(self, backends: dict, manifest: Manifest, arm: str, retry: bool = True):
        super().__init__(backends, retry)
        check_manifest(manifest)
        self.m, self.arm = manifest, arm
        self.skills = {s.name: s for s in arm_skills(manifest, arm)}
        self.instructions = instructions(manifest)

    def manifest_errors(self, plan: dict, steps: list | None = None) -> list[str]:
        errs = []
        for s in steps if steps is not None else plan["steps"]:
            sk = self.skills.get(s["skill"])
            if sk is None:
                errs.append(f"step {s['id']}: {self.m.robot} has no skill {s['skill']}")
                continue
            errs += [f"step {s['id']}: {s['skill']} needs {a}" for a in sk.args_schema.get("required", []) if s.get(a, P.NONE) == P.NONE]
        return errs

    def _req(self, kind: str, body: dict, schema: dict) -> LLMRequest:
        return LLMRequest(kind, {"plan": "plan", "replan": "plan_diff"}[kind], self.instructions,
                          json.dumps({"request": kind, **body}), schema)

    def plan(self, task: str, user_messages: list, world: PWorld, core: WorldState, ref=None, route: str = "fast_llm") -> PlanResult:
        ids = P.schema_ids(core)
        req = self._req("plan", {"task": task, "user_messages": user_messages, "world": world_view(world, self.arm)},
                        plan_schema(self.m, self.arm, ids))
        check = lambda p: self.manifest_errors(p) or P.validate(p, core, core.arm.holding)  # noqa: E731
        out, att = self._run(route, req, check, ref)
        return PlanResult("plan", route, out is not None, plan=out, attempts=att)

    def replan(self, route: str, task: str, user_messages: list, world: PWorld, core: WorldState, plan: dict, queue: list,
               status: dict, arm_now: dict, why: str, ref=None) -> PlanResult:
        body = {"task": task, "user_messages": user_messages, "world": world_view(world, self.arm),
                "current_plan": plan_view(plan, queue, status, core), "arm_now": arm_now, "why": why}
        req = self._req("replan", body, diff_schema(self.m, self.arm, P.schema_ids(core)))

        def check(d):
            errs = P.diff_errors(plan, d) or self.manifest_errors(plan, d["new_steps"])
            if errs:
                return errs
            newp, q = P.apply_diff(plan, d)
            return P.validate(newp, core, core.arm.holding, q)

        out, att = self._run(route, req, check, ref)
        return PlanResult("replan", route, out is not None, diff=out, attempts=att)
