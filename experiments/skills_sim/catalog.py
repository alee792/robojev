"""The standard skill catalog: fixed names and argument shapes every robot server offers where it can.

MCP gives discovery; it says nothing about what tools are called. Shared names are what make
planner prompts, the decision loop's expectations, evaluations and, above all, recorded training
data transfer between robots: a `pick_and_place` on a WidowX, a Panda and a YAM is one dataset.

- Mandatory on every server: the control tools (hold, pause, resume, stop, heartbeat) and the
  resources (manifest, world, status). That is the safety contract.
- Standard skills: the specs below. A server advertises those it can do, unchanged.
- Extensions: robot-specific skills under a prefix ("widowx.wiggle_free"), with their own schemas.
  The planner sees them through the manifest; only standard skills get shared prompts and evals.
- Failure reasons: `REASONS` are the standard codes; the literal text the decision loop reads goes
  alongside, it does not replace them.

Arguments name objects and places by their world-state ids.
"""
from __future__ import annotations

from .protocol import SkillSpec

CATALOG_VERSION = "0.1"


def _obj(desc="object id from the world state"):
    return {"type": "string", "description": desc}


PICK_AND_PLACE = SkillSpec(
    "pick_and_place", "Pick up an object and put it down at a place (a slot, a bin, a free spot on the table).",
    {"type": "object", "properties": {"object": _obj(), "place": {"type": "string", "description": "place id"}},
     "required": ["object", "place"], "additionalProperties": False})

STACK_ON = SkillSpec(
    "stack_on", "Pick up an object and put it down on top of another object.",
    {"type": "object", "properties": {"object": _obj(), "onto": _obj("object id to stack on")},
     "required": ["object", "onto"], "additionalProperties": False})

PUSH = SkillSpec(
    "push", "Slide an object along the table with the closed gripper, without picking it up.",
    {"type": "object", "properties": {
        "object": _obj(),
        "direction": {"type": "string", "enum": ["left", "right", "toward_robot", "away_from_robot"]},
        "distance": {"type": "number", "description": "metres", "minimum": 0.01, "maximum": 0.3}},
     "required": ["object", "direction", "distance"], "additionalProperties": False})

HAND_OVER = SkillSpec(
    "hand_over", "Pick up an object, hold it out toward the person, and release it when their open hand is beneath it.",
    {"type": "object", "properties": {"object": _obj()}, "required": ["object"], "additionalProperties": False})

SURVEY = SkillSpec(
    "survey", "Lift the gripper clear of the table so the cameras see everything.",
    {"type": "object", "properties": {}, "additionalProperties": False})

HOLD = SkillSpec(
    "hold", "Stay still for a moment (a planned pause, e.g. to let a person finish).",
    {"type": "object", "properties": {"seconds": {"type": "number", "minimum": 0.5, "maximum": 10, "default": 1.0}},
     "additionalProperties": False})

STANDARD = (PICK_AND_PLACE, STACK_ON, PUSH, HAND_OVER, SURVEY, HOLD)
BY_NAME = {s.name: s for s in STANDARD}

# Mandatory on every server, whatever skills it offers.
CONTROL_TOOLS = ("hold", "pause", "resume", "retarget", "stop", "heartbeat")
RESOURCES = ("manifest", "world", "status")

# Standard failure codes. A SkillStatus.reason is "<code>: <literal text>".
REASONS = ("unreachable", "grasp_failed", "dropped", "blocked", "stalled", "timeout", "not_in_view", "precondition")


def is_standard(name: str) -> bool:
    return name in BY_NAME


def is_extension(name: str) -> bool:
    return "." in name and not is_standard(name)


def check_manifest_skills(skills) -> list[str]:
    """Problems with a server's advertised skills: a standard name with a changed schema, or a
    non-standard name without a prefix."""
    problems = []
    for s in skills:
        std = BY_NAME.get(s.name)
        if std is not None and s.args_schema != std.args_schema:
            problems.append(f"{s.name}: schema differs from the catalog")
        elif std is None and not is_extension(s.name):
            problems.append(f"{s.name}: not a standard skill and not prefixed as an extension")
    return problems
