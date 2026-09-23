"""E13 system under test, generic part: the Plan schema, its validator and branch lookup.

Nothing here knows what any task means. A plan is whatever the LLM returned:
  {"task_reading": str,
   "parameters": [{"name", "about", "values": [{"value", "meaning"}], "default"}],
   "branches":   [{"settings": [{"parameter", "value"}], "goal": [{"block", "destination"}]}]}
The scene supplies the block ids and the destinations, and says which destinations hold one block
(`scene_view["destinations"]`). Code only checks the plan's shape and consistency. Must not import
the oracle.
"""

from __future__ import annotations

MAX_BRANCHES = 4
MAX_PARAMETERS = 3
MAX_VALUES = 4


def destinations(scene_view: dict) -> tuple[list[str], list[str]]:
    d = scene_view["destinations"]
    return list(d["one_block_each"]), list(d["any_number_of_blocks"])


def block_ids(scene_view: dict) -> list[str]:
    return [b["id"] for b in scene_view["blocks"]]


def plan_schema(scene_view: dict) -> dict:
    """Strict JSON schema (OpenAI structured outputs: every property required, no extra keys).
    Block ids and destinations are enums taken from the scene; everything else is generic."""
    one, many = destinations(scene_view)
    s = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["task_reading", "parameters", "branches"],
        "properties": {
            "task_reading": {**s, "description": "One sentence: the final arrangement the task asks for."},
            "parameters": {
                "type": "array",
                "description": f"0-{MAX_PARAMETERS} settings of this task the user might change while the robot works.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "about", "values", "default"],
                    "properties": {
                        "name": {**s, "description": "short snake_case name"},
                        "about": {**s, "description": "what this setting controls, in plain words"},
                        "values": {
                            "type": "array",
                            "description": f"2-{MAX_VALUES} values, each named by its meaning",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["value", "meaning"],
                                "properties": {"value": {**s, "description": "snake_case, named by meaning"},
                                               "meaning": {**s, "description": "one line a user's words can be matched against"}},
                            },
                        },
                        "default": {**s, "description": "the value the task as given asks for"},
                    },
                },
            },
            "branches": {
                "type": "array",
                "description": f"1-{MAX_BRANCHES} branches: the default combination first, then the likeliest alternatives.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["settings", "goal"],
                    "properties": {
                        "settings": {
                            "type": "array",
                            "description": "one entry per parameter",
                            "items": {"type": "object", "additionalProperties": False, "required": ["parameter", "value"],
                                      "properties": {"parameter": s, "value": s}},
                        },
                        "goal": {
                            "type": "array",
                            "description": "one entry per block: where it must end up",
                            "items": {"type": "object", "additionalProperties": False, "required": ["block", "destination"],
                                      "properties": {"block": {"type": "string", "enum": block_ids(scene_view)},
                                                     "destination": {"type": "string", "enum": one + many}}},
                        },
                    },
                },
            },
        },
    }


def settings_of(branch: dict) -> dict[str, str]:
    return {s["parameter"]: s["value"] for s in branch["settings"]}


def default_settings(plan: dict) -> dict[str, str]:
    return {p["name"]: p["default"] for p in plan["parameters"]}


def combo_key(plan: dict, settings: dict[str, str]) -> str:
    """Canonical key for a combination, in the plan's parameter order ('' when the plan has none)."""
    return "|".join(f"{p['name']}={settings.get(p['name'], '?')}" for p in plan["parameters"])


def branch_keys(plan: dict) -> list[str]:
    return [combo_key(plan, settings_of(b)) for b in plan["branches"]]


def goal_of(branch: dict) -> dict[str, str]:
    return {g["block"]: g["destination"] for g in branch["goal"]}


def branch_goals(plan: dict) -> dict[str, dict[str, str]]:
    return {combo_key(plan, settings_of(b)): goal_of(b) for b in plan["branches"]}


def lite(plan: dict) -> dict:
    """The plan without goals: enough to decide a branch from Jev's answers (kept in the log)."""
    return {"task_reading": plan.get("task_reading", ""), "parameters": plan["parameters"],
            "branches": [{"settings": b["settings"]} for b in plan["branches"]]}


def validate(plan, scene_view: dict) -> list[str]:
    """Every problem found, as short sentences the LLM can act on. Empty list = valid."""
    if not isinstance(plan, dict):
        return ["the plan is not a JSON object"]
    errs: list[str] = []
    params = plan.get("parameters") or []
    branches = plan.get("branches") or []
    if len(params) > MAX_PARAMETERS:
        errs.append(f"{len(params)} parameters; at most {MAX_PARAMETERS}")
    names = [p.get("name") for p in params]
    if len(set(names)) != len(names):
        errs.append("parameter names repeat")
    allowed = {}
    for p in params:
        vals = [v.get("value") for v in p.get("values") or []]
        if not 2 <= len(vals) <= MAX_VALUES:
            errs.append(f"parameter {p.get('name')!r} has {len(vals)} values; needs 2-{MAX_VALUES}")
        if len(set(vals)) != len(vals):
            errs.append(f"parameter {p.get('name')!r} repeats a value")
        if p.get("default") not in vals:
            errs.append(f"parameter {p.get('name')!r}: default {p.get('default')!r} is not one of its values")
        allowed[p.get("name")] = set(vals)
    if not 1 <= len(branches) <= MAX_BRANCHES:
        errs.append(f"{len(branches)} branches; needs 1-{MAX_BRANCHES}")
    ids = block_ids(scene_view)
    one, many = destinations(scene_view)
    keys = []
    for i, b in enumerate(branches):
        where = f"branch {i + 1}"
        st = b.get("settings") or []
        got = [s.get("parameter") for s in st]
        if sorted(got, key=str) != sorted(names, key=str):
            errs.append(f"{where}: settings must name each parameter exactly once ({names}), got {got}")
        for s in st:
            if s.get("parameter") in allowed and s.get("value") not in allowed[s.get("parameter")]:
                errs.append(f"{where}: {s.get('value')!r} is not a value of {s.get('parameter')!r}")
        keys.append(combo_key(plan, {s.get("parameter"): s.get("value") for s in st}))
        goal = b.get("goal") or []
        blocks = [g.get("block") for g in goal]
        missing = [x for x in ids if x not in blocks]
        extra = sorted({x for x in blocks if x not in ids}, key=str)
        dup = sorted({x for x in blocks if blocks.count(x) > 1}, key=str)
        if missing:
            errs.append(f"{where}: no destination for {', '.join(missing)}")
        if dup:
            errs.append(f"{where}: more than one destination for {', '.join(dup)}")
        if extra:
            errs.append(f"{where}: unknown blocks {extra}")
        dests = [g.get("destination") for g in goal]
        bad = sorted({d for d in dests if d not in one and d not in many}, key=str)
        if bad:
            errs.append(f"{where}: unknown destinations {bad}")
        shared = sorted({d for d in dests if d in one and dests.count(d) > 1})
        if shared:
            errs.append(f"{where}: more than one block in {', '.join(shared)} (each holds one block)")
    if len(set(keys)) != len(keys):
        errs.append("two branches have the same settings")
    if params and combo_key(plan, {p.get("name"): p.get("default") for p in params}) not in keys:
        errs.append("no branch has every parameter at its default")
    return errs


def default_branch(plan: dict) -> dict:
    key = combo_key(plan, default_settings(plan))
    return next(b for b in plan["branches"] if combo_key(plan, settings_of(b)) == key)
