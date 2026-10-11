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

More than one arm (the brain driving them all): every step gains an `arm` field, an enum of the
manifest's arm ids, and the description says each arm's box, what each reaches in the world as it
is now (the one line that depends on the world, built per request), and how steps on different arms
share the table. With one arm (or one arm chosen from several) nothing of this appears: schema,
description and world view are exactly the one-arm ones.
"""
from __future__ import annotations

import re

from .core import plan as P
from .core.planner import INSTRUCTIONS

from collections.abc import Mapping, Sequence

from ..protocol import ArmSpec, Manifest, SkillSpec
from ..protocol import WorldState as RobotWorld
from .arms import Reach, reaches
from .steps import CORE_FIELDS, args_of, extra_fields, field_of, plannable, plannable_on

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


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def reach_text(m: Manifest, ws: RobotWorld) -> str:
    """What each arm reaches now, by id: the loose objects in its box, the groups (tray, shared zone)
    whose every member is in it, and slots or bins in it that belong to no such group. Then the
    shared-zone rule, if the world has a zone."""
    parts = []
    for a in m.arms:
        groups = [p for p in ws.places.values() if p.members and all(
            q in ws.places and reaches(a, ws.places[q].x, ws.places[q].y) for q in p.members)]
        grouped = {q for g in groups for q in g.members}
        singles = [p for p in ws.places.values() if p.kind in ("slot", "bin") and p.id not in grouped
                   and not any(p.id in g.members for g in ws.places.values()) and reaches(a, p.x, p.y)]
        loose = [o for o in ws.objects.values() if (o.where == "table" or o.where.startswith("on:")) and reaches(a, o.x, o.y)]
        got = [x for x in (ident(o.id, "") for o in loose + groups + singles) if x]
        parts.append(f"{ident(a.id, 'an arm')} reaches {_join(got) if got else 'nothing listed in the world'}")
    text = "Right now " + "; ".join(parts) + " (every arm reaches the table for a free spot)."
    zone = next((p for p in ws.places.values() if p.kind == "zone"), None)
    if zone is not None:
        text += f" Only one arm may be inside {ident(zone.id, 'the shared zone')} at a time: the robot makes the other wait."
    return text


def _describe_arms(m: Manifest, ws: RobotWorld | None) -> str:
    """Several arms, all planned for: each arm's box, what each reaches now, how their steps run."""
    labels = [ident(a.id, f"arm {i}") for i, a in enumerate(m.arms)]
    frames = {a.base_frame for a in m.arms}
    frame = (f"in the `{ident(m.arms[0].base_frame, 'shared')}` frame the arms share" if len(frames) == 1 else
             "each in its own arm's base frame")
    head = [f"The robot has {len(m.arms)} arms ({_join(labels)}); you plan for all of them, and every step names the arm "
            f"that runs it. Positions are in metres, {frame}; each arm reaches a box in it."]
    head += [_arm_line(a, label) for a, label in zip(m.arms, labels)]
    if ws is not None:
        head.append(reach_text(m, ws))
    head.append("Steps on different arms run at the same time unless they use the same object or one-object place; "
                "an arm runs its own steps in order. Give each step to an arm that reaches its object and where it goes; "
                "to move an object from one arm's reach to another's, one arm puts it in a slot both reach and the other "
                "takes it from there.")
    skills = plannable_on(m, [a.id for a in m.arms])

    def only(s: SkillSpec) -> str:
        on = [ident(a.id, "an arm") for a in m.arms if s in plannable(m, a.id)]
        return "" if len(on) == len(m.arms) else f" ({_join(on)} only)"
    each = ["Its skills, the only steps a plan may use:"] + [f"- {_signature(s)}{only(s)}: {s.description}" for s in skills]
    if len(head) + len(each) > MAX_DESCRIPTION_LINES:
        each = ["Its skills, the only steps a plan may use: " + ", ".join(_signature(s) + only(s) for s in skills) + "."]
    return "\n".join(head + each)


def describe_robot(m: Manifest, arm: str | None = None, world: RobotWorld | None = None) -> str:
    """The robot for the planner's prompt, in at most MAX_DESCRIPTION_LINES lines: how many arms, the
    driven arm's gripper and reach, and its skills (one line each, or one line of signatures when
    that would run over). Several arms with no `arm` chosen: every arm is planned for, and the text
    gives each arm's box, what each reaches in `world` (if given) and how their steps share the table."""
    if arm is None and len(m.arms) > 1:
        return _describe_arms(m, world)
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


def _fields_text(skills: list[SkillSpec], arms: Sequence[str] = ()) -> str:
    """How each skill's arguments sit in a step (and, for several arms, which arm runs it)."""
    by_field: dict[str, list[str]] = {}
    for s in skills:
        for a in args_of(s):
            by_field.setdefault(field_of(a), [])
            if a not in by_field[field_of(a)]:
                by_field[field_of(a)].append(a)
    carried = "; ".join(f"`{f}` carries {' or '.join(args)}" for f, args in by_field.items())
    if arms:
        carried = f"`arm` is the arm that runs the step ({' or '.join(arms)}); " + carried
    fields = ", ".join(("id",) + (("arm",) if arms else ()) + ("skill",) + CORE_FIELDS + tuple(extra_fields(skills)))
    return (f"A step is {{{fields}}}" + (f": {carried}" if carried else "") +
            '. A field the step\'s skill does not take is "none" (null for numbers).')


def instructions(m: Manifest, arm: str | None, world: RobotWorld | None = None) -> str:
    """The planner's whole instruction text: identical across calls (cacheable), varying only by robot.
    With several arms and no `arm` chosen it plans for all of them, and the reach line varies with
    `world` (it changes only when an object changes whose reach it is in)."""
    if arm is None and len(m.arms) > 1:
        return _instructions_arms(m, world)
    skills = plannable(m, arm)
    return ("You plan for a robot working on a table.\n\n" + describe_robot(m, arm) + "\n\n"
            "The input's `world` lists every object (id, name, the label read off it, size in m, where it is, x/y in m) "
            "and every place (tray slots that hold one object each, bins that hold any number, \"table\" for a free spot on "
            "the table, \"person\" for handing an object to the person, and a group such as \"tray\" meaning any of its "
            "slots, for conditions and constraints only). The skills are motor capabilities only; they do not know the task.\n\n"
            + _fields_text(skills) + "\n\n" + _RULES)


def _instructions_arms(m: Manifest, world: RobotWorld | None) -> str:
    arms = [a.id for a in m.arms]
    skills = plannable_on(m, arms)
    return ("You plan for a robot working on a table.\n\n" + describe_robot(m, None, world) + "\n\n"
            "The input's `world` lists every object (id, name, the label read off it, size in m, where it is: a place id, "
            "\"table\", \"on:<object id>\" or \"gripper:<arm id>\", x/y in m, and the arms that reach it now) and every place "
            "(tray slots and shared-zone slots that hold one object each, bins that hold any number, \"table\" for a free spot "
            "on the table, \"person\" for handing an object to the person, and groups such as \"tray\" or the shared zone "
            "meaning any of their slots, for conditions and constraints only), with the arms that reach each place. "
            "`world.arms` says what each arm holds. The skills are motor capabilities only; they do not know the task.\n\n"
            + _fields_text(skills, arms) + "\n\n" + _RULES +
            "\n\nWith several arms: an arm holding an object (`world.arms.<id>.holding`) must put it somewhere with its "
            "first step; `arm_now` in a replan says what each arm is doing.")


def step_schema(skills: list[SkillSpec], ids: dict, arms: Sequence[str] = ()) -> dict:
    """One plan step, strict: every property required, no extra keys. `arms` (several): the step's
    `arm` field, right after its id."""
    def users(f: str) -> list[tuple[str, str, dict]]:
        return [(s.name, a, s.args_schema["properties"][a]) for s in skills for a in args_of(s) if field_of(a) == f]

    def desc(f: str, empty: str) -> str:
        u = users(f)
        said = "; ".join(f"{n}: its {a}" + (f" ({p['description']})" if p.get("description") else "") for n, a, p in u)
        return f"{said}; else {empty}" if u else f"no skill uses it: always {empty}"

    dirs = sorted({d for _, _, p in users("direction") for d in p.get("enum", [])})
    props = {
        "id": {"type": "string", "description": "short unique id, e.g. s1 (new ids in a replan: n1, n2, ...)"},
        **({"arm": {"type": "string", "enum": list(arms),
                    "description": "the arm that runs this step: one that reaches its object and where it goes"}} if arms else {}),
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


def plan_schema(skills: list[SkillSpec], ids: dict, arms: Sequence[str] = ()) -> dict:
    s = P.plan_schema(ids)
    s["properties"]["steps"]["items"] = step_schema(skills, ids, arms)
    return s


def diff_schema(skills: list[SkillSpec], ids: dict, arms: Sequence[str] = ()) -> dict:
    s = P.diff_schema(ids)
    s["properties"]["new_steps"]["items"] = step_schema(skills, ids, arms)
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


def world_view_arms(ws: RobotWorld, specs: Mapping[str, ArmSpec]) -> dict:
    """world_view for several arms: grippers keep their arm ("gripper:arm_1"), every object and place
    says which arms reach it now, and `arms` says what each holds."""
    r = Reach(ws, specs)
    return {
        "objects": [{"id": o.id, "name": o.name, "label": o.label, "size_m": o.size, "where": o.where,
                     "x": round(o.x, 3), "y": round(o.y, 3), "reached_by": r.object_arms(o.id)} for o in ws.objects.values()],
        "places": [{"id": p.id, "name": p.name, "kind": p.kind, "holds": "one object" if p.capacity == 1 else "any number",
                    "reached_by": r.place_arms(p.id) if not p.members else
                    [a for a in specs if all(a in r.place_arms(q) for q in p.members)]}
                   for p in ws.places.values()],
        "arms": {a: {"holding": ws.arms[a].holding or "nothing"} for a in specs},
    }
