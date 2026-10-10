"""A plan step <-> a protocol skill call: the one place the brain translates between vocabularies.

Inside the brain, plans use e12v2's flat step format ({id, skill, object, target, direction}),
because e12v2's validator, diffs, done checks and decision text read that format and are reused
unchanged. Outside it (the planner's schema, the LLM's answers, every start()) the catalog's names
and arguments are used. Two things differ, and are translated only here:

- Names. e12v2 calls the catalog's `pick_and_place` "move_object"; every other standard skill has
  the same name in both.
- Argument slots. The catalog's `place` (pick_and_place) and `onto` (stack_on) both ride in e12v2's
  `target`; `object` and `direction` keep their names. Any other argument (push's `distance`,
  hold's `seconds`) gets a step field of its own name, which e12v2's code carries along and ignores.

Only standard skills are plannable. e12v2's validator replays each step's effect on a symbolic world
and knows nothing of an extension's effects ("widowx.wiggle_free" would be rejected there as an
unknown skill), so extensions wait for a validator that takes effects from somewhere other than code.
"""
from __future__ import annotations

from collections.abc import Iterable

from e12v2.core.plan import NONE

from .. import catalog
from ..protocol import Manifest, SkillSpec

CORE_NAME = {"pick_and_place": "move_object", "stack_on": "stack_on", "push": "push",
             "hand_over": "hand_over", "survey": "survey", "hold": "hold"}
CATALOG_NAME = {core: cat for cat, core in CORE_NAME.items()}
CORE_FIELDS = ("object", "target", "direction")       # e12v2's own step fields, besides id and skill
SLOT = {"object": "object", "place": "target", "onto": "target", "direction": "direction"}


def field_of(arg: str) -> str:
    """The step field a catalog argument rides in."""
    return SLOT.get(arg, arg)


def args_of(spec: SkillSpec) -> list[str]:
    return list(spec.args_schema.get("properties", {}))


def plannable(manifest: Manifest, arm: str) -> list[SkillSpec]:
    """The skills a plan may use on `arm`: advertised for it and standard. Specs are the brain's own
    copies from the catalog (identical to the manifest's, which connect() checked): what the planner
    is told about a skill never comes from the server's text."""
    return [catalog.BY_NAME[s.name] for s in manifest.skills
            if s.name in CORE_NAME and (not s.arms or arm in s.arms)]


def extra_fields(skills: Iterable[SkillSpec]) -> list[str]:
    """Step fields beyond e12v2's, one per argument that has no e12v2 slot, in a stable order."""
    return sorted({field_of(a) for s in skills for a in args_of(s)} - set(CORE_FIELDS))


def to_core(step: dict) -> dict:
    """A step as the planner wrote it (catalog names) -> as the brain keeps it (e12v2 names)."""
    return {**step, "skill": CORE_NAME.get(step["skill"], step["skill"])}


def to_catalog(step: dict) -> dict:
    """A step as the brain keeps it -> as the planner reads and writes it."""
    return {**step, "skill": CATALOG_NAME.get(step["skill"], step["skill"])}


def plan_to_core(plan: dict) -> dict:
    return {**plan, "steps": [to_core(s) for s in plan["steps"]]}


def diff_to_core(diff: dict) -> dict:
    return {**diff, "new_steps": [to_core(s) for s in diff["new_steps"]]}


def plan_to_catalog(plan: dict) -> dict:
    return {**plan, "steps": [to_catalog(s) for s in plan["steps"]]}


def call_args(step: dict, spec: SkillSpec) -> dict:
    """A step -> the arguments start() takes: the skill's own argument names, filled from the step's
    fields. An argument the step leaves empty ("none" or null) takes the schema's default, or is left
    out (the server then refuses a required one with a literal reason)."""
    out = {}
    for a, p in spec.args_schema.get("properties", {}).items():
        v = step.get(field_of(a))
        if v is None or v == NONE:
            if "default" not in p:
                continue
            v = p["default"]
        out[a] = v
    return out


def missing_args(step: dict, spec: SkillSpec) -> list[str]:
    have = call_args(step, spec)
    return [a for a in spec.args_schema.get("required", []) if a not in have]
