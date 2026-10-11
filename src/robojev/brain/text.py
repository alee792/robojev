"""Literal text for events, as the decision model reads it (docs/v2.md "Writing Jev questions": say
exactly what happened, name things as people do). Same wording as e12v2's harness, so its prompts
and mocks read these events as they read its own.
"""
from __future__ import annotations

from collections.abc import Callable

BAND = {"very close": "very close to", "near": "near", "mid-range": "mid-range from", "far": "far from"}

Name = Callable[[str], str]


def where_text(where: str, name: Name) -> str:
    if where.startswith("on:"):
        return f"on top of {name(where[3:])}"
    return {"table": "on the table", "gripper": "in the gripper", "person": "held by the person"}.get(where, f"in {name(where)}")


def hand_text(now: tuple | None, before: tuple | None) -> str:
    """A hand transition from core.changes: `now` is (band, approaching, in_path, held_out) or None."""
    if now is None:
        return "the person's hand has gone out of view"
    band, approaching, in_path, held_out = now
    bits = [BAND[band] + " the gripper", "coming closer" if approaching else "not coming closer",
            "in the arm's path" if in_path else "not in the arm's path"]
    if held_out:
        bits.append("held out still, palm up, as if to take something")
    return ("a person's hand appeared: " if before is None else "the person's hand is now ") + ", ".join(bits)


def change_text(ch: dict, name: Name) -> str:
    """One change the robot didn't cause (core.changes.ChangeDetector.update), in words."""
    w = ch["what"]
    if w == "hand":
        return hand_text(ch["now"], ch["before"])
    o = name(ch["object"])
    if w == "object_moved":
        fr, to = where_text(ch["from"], name), where_text(ch["to"], name)
        if ch["from"] != ch["to"]:
            return f"{o} was moved by someone else (not the robot): it was {fr}, now it is {to}"
        return f"{o} was moved by someone else (not the robot): still {to}, about {ch['moved_cm']} cm from where it was"
    if w == "object_gone":
        return f"{o} is no longer in view (it was {where_text(ch['from'], name)})"
    return f"{o} appeared, {where_text(ch['to'], name)}"


def between_steps(step_text: str, finished: bool, holding: str | None) -> str:
    held = f"holding {holding}" if holding else "the gripper is empty"
    return f"between steps: {step_text} just {'finished' if finished else 'stopped'}; {held}"
