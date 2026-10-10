"""What the planner is shown: the robot described in the brain's own words, and a step schema
generated from the manifest.

The manifest decides *which* skills exist on the arm; the catalog decides what is *said* about them.
Free text from the server (robot name, arm descriptions, skill descriptions) never reaches the
prompt: identifiers pass only if they look like identifiers, numbers are formatted here, and skill
descriptions are the brain's own catalog copies ("Manifest text is data", protocol.RobotServer).

The step schema is e12v2's step with every enum and description regenerated: `skill` lists only the
plannable skills on the arm, so a strict-output planner cannot write a skill the robot does not
advertise; `object`, `target` and `direction` are "none"-only when no such skill uses them; each
argument without an e12v2 slot (push's distance, hold's seconds) is a nullable field of its own.
The rest of the plan (done conditions, constraints, the diff shape) is e12v2's.
"""
from __future__ import annotations

import re

from e12v2.core import plan as P
from e12v2.core.planner import INSTRUCTIONS

from ..protocol import ArmSpec, Manifest, SkillSpec
from ..protocol import WorldState as RobotWorld
from .steps import CORE_FIELDS, args_of, extra_fields, field_of, plannable

MAX_DESCRIPTION_LINES = 10
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,39}$")
_RULES = INSTRUCTIONS[INSTRUCTIONS.index("A plan has:"):]   # e12v2's plan rules; its head described a cm toy frame


def ident(text: str, fallback: str) -> str:
    """`text` if it is a plain identifier, else `fallback`: a server-supplied id is never free text."""
    return text if _IDENT.match(text or "") else fallback


def _signature(s: SkillSpec) -> str:
    return f"{s.name}({', '.join(args_of(s))})"


def _arm_line(a: ArmSpec, label: str) -> str:
    (x0, x1), (y0, y1), (z0, z1) = a.workspace
    g = a.gripper
    grip = ("no gripper" if g is None else
            f"a gripper that opens to {g.max_width * 100:.1f} cm and can grasp" if g.can_grasp else
            "an end effector that cannot grasp; it can only touch and push")
    return (f"{label}: {grip}; it reaches x {x0:.2f} to {x1:.2f} m, y {y0:.2f} to {y1:.2f} m, z {z0:.2f} to {z1:.2f} m, "
            f"and travels {a.travel_z:.2f} m up between places.")


def describe_robot(m: Manifest, arm: str | None = None) -> str:
    """The robot for the planner's prompt, in at most MAX_DESCRIPTION_LINES lines: how many arms, the
    driven arm's gripper and reach, and its skills (one line each, or one line of signatures when
    that would run over)."""
    a = next(x for x in m.arms if arm is None or x.id == arm)
    label = ident(a.id, "the arm")
    n = len(m.arms)
    skills = plannable(m, a.id)
    head = [f"The robot has {n} arm{'s' if n != 1 else ''}; you plan for {label}. "
            "Positions are in metres in the arm's base frame: +x away from the robot, +y to its left, +z up.",
            _arm_line(a, label)]
    each = ["Its skills, the only steps a plan may use:"] + [f"- {_signature(s)}: {s.description}" for s in skills]
    if len(head) + len(each) > MAX_DESCRIPTION_LINES:
        each = ["Its skills, the only steps a plan may use: " + ", ".join(_signature(s) for s in skills) + "."]
    return "\n".join(head + each)


def _fields_text(skills: list[SkillSpec]) -> str:
    """How each skill's arguments sit in a step."""
    by_field: dict[str, list[str]] = {}
    for s in skills:
        for a in args_of(s):
            by_field.setdefault(field_of(a), [])
            if a not in by_field[field_of(a)]:
                by_field[field_of(a)].append(a)
    carried = "; ".join(f"`{f}` carries {' or '.join(args)}" for f, args in by_field.items())
    fields = ", ".join(("id", "skill") + CORE_FIELDS + tuple(extra_fields(skills)))
    return (f"A step is {{{fields}}}" + (f": {carried}" if carried else "") +
            '. A field the step\'s skill does not take is "none" (null for numbers).')


def instructions(m: Manifest, arm: str) -> str:
    """The planner's whole instruction text: identical across calls (cacheable), varying only by robot."""
    skills = plannable(m, arm)
    return ("You plan for a robot working on a table.\n\n" + describe_robot(m, arm) + "\n\n"
            "The input's `world` lists every object (id, name, the label read off it, size in m, where it is, x/y in m) "
            "and every place (tray slots that hold one object each, bins that hold any number, \"table\" for a free spot on "
            "the table, \"person\" for handing an object to the person, and a group such as \"tray\" meaning any of its "
            "slots, for conditions and constraints only). The skills are motor capabilities only; they do not know the task.\n\n"
            + _fields_text(skills) + "\n\n" + _RULES)


def step_schema(skills: list[SkillSpec], ids: dict) -> dict:
    """One plan step, strict: every property required, no extra keys."""
    def users(f: str) -> list[tuple[str, str, dict]]:
        return [(s.name, a, s.args_schema["properties"][a]) for s in skills for a in args_of(s) if field_of(a) == f]

    def desc(f: str, empty: str) -> str:
        u = users(f)
        said = "; ".join(f"{n}: its {a}" + (f" ({p['description']})" if p.get("description") else "") for n, a, p in u)
        return f"{said}; else {empty}" if u else f"no skill uses it: always {empty}"

    dirs = sorted({d for _, _, p in users("direction") for d in p.get("enum", [])})
    props = {
        "id": {"type": "string", "description": "short unique id, e.g. s1 (new ids in a replan: n1, n2, ...)"},
        "skill": {"type": "string", "enum": [s.name for s in skills]},
        "object": {"type": "string", "enum": (ids["objects"] if users("object") else []) + [P.NONE],
                   "description": desc("object", P.NONE)},
        "target": {"type": "string", "enum": (ids["places"] + ids["objects"] if users("target") else []) + [P.NONE],
                   "description": desc("target", P.NONE)},
        "direction": {"type": "string", "enum": dirs + [P.NONE], "description": desc("direction", P.NONE)},
    }
    for f in extra_fields(skills):
        spec = users(f)[0][2]
        bounds = "".join(f"; {k} {spec[k]}" for k in ("minimum", "maximum") if k in spec)
        props[f] = {"type": [spec.get("type", "string"), "null"], "description": desc(f, "null") + bounds}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def plan_schema(skills: list[SkillSpec], ids: dict) -> dict:
    s = P.plan_schema(ids)
    s["properties"]["steps"]["items"] = step_schema(skills, ids)
    return s


def diff_schema(skills: list[SkillSpec], ids: dict) -> dict:
    s = P.diff_schema(ids)
    s["properties"]["new_steps"]["items"] = step_schema(skills, ids)
    return s


def world_view(ws: RobotWorld, arm: str) -> dict:
    """What the planner reads of the world: protocol-native, in metres, the driven arm's gripper as "gripper"."""
    def where(w: str) -> str:
        return "gripper" if w == f"gripper:{arm}" else w
    return {
        "objects": [{"id": o.id, "name": o.name, "label": o.label, "size_m": o.size, "where": where(o.where),
                     "x": round(o.x, 3), "y": round(o.y, 3)} for o in ws.objects.values()],
        "places": [{"id": p.id, "name": p.name, "kind": p.kind, "holds": "one object" if p.capacity == 1 else "any number"}
                   for p in ws.places.values()],
        "arm": {"holding": ws.arms[arm].holding or "nothing"},
    }
