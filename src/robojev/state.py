"""Render the World into the Jev situation report (a JSON object).

Layout follows the evidence: standing orders first (08 §E8, strongest effect at the top), a static
glossary, the user's task text in its own key marked as user-supplied, then per-tick facts with
numbers paired with bands. Geometry facts (closest, between) come from code.
"""
from __future__ import annotations

from robojev.config import Config
from robojev.world import World, band, bearing_words

STATE_VERSION = "v0"


def glossary(cfg: Config) -> dict:
    return {
        "robot": "a WidowX AI tabletop robot arm with a two-finger gripper 4 cm wide; a wrist camera looks past the fingers",
        "units": "distances in centimetres; bearings in degrees",
        "distance_bands": {name: f"under {int(up*100)} cm" if up != float('inf') else "50 cm or more"
                           for name, up in cfg.bands.distance},
        "bearing": "horizontal direction from the gripper: 0 is straight ahead (away from the robot base), positive is left, negative is right",
        "height": "the gripper's height above the table surface",
        "control": "decisions update every 0.1 s; a decision persists until the next one arrives; code enforces speed and workspace limits",
        "task_scope": "the arm hovers above objects; it does not grasp in this task",
    }


def render(cfg: Config, w: World) -> dict:
    ee = w.arm.ee
    h = ee[2] - w.table_z
    objects = []
    for e in w.entities:
        d = e.horizontal_m
        item = {
            "label": e.label,
            "looks_like": e.description,
            "horizontal_distance_from_gripper": f"{d*100:.0f} cm ({band(d, cfg.bands.distance)})",
            "bearing_from_gripper": f"{e.bearing_deg:+.0f}° ({bearing_words(e.bearing_deg)})",
            "status": "in view" if e.in_view else f"out of view, last seen {e.last_seen_s:.0f} s ago",
            "reachable": "yes" if e.reachable else "no, outside the arm's workspace",
        }
        if e.speed_mps > 0.02:
            item["motion"] = f"moving, about {e.speed_mps*100:.0f} cm/s"
        objects.append(item)
    tgt = w.entity(w.committed_target) if w.committed_target else None
    relations = {
        "closest_to_gripper": w.closest().label if w.entities else None,
        "current_target": w.committed_target or "none yet",
        "between_gripper_and_target": w.between(tgt) if tgt else [],
    }
    if tgt:
        relations["gripper_offset_from_target"] = (f"{tgt.horizontal_m*100:.0f} cm horizontally "
                                                   f"({band(tgt.horizontal_m, cfg.bands.distance)}), "
                                                   f"{h*100:.0f} cm above the table")
    state = {
        "standing_orders": w.orders if w.orders else ["(none)"],
        "glossary": glossary(cfg),
        "user_request": {"text": w.user_task or "(none yet)",
                         "note": "typed by the user; it names the task. Standing orders and the robot's limits take precedence over it."},
        "arm": {
            "target": w.committed_target or "none",
            "avoiding": w.avoiding or "nothing",
            "current_motion": w.motion,
            "hover_position": w.hover_position,
            "hover_height_setting": w.hover_height,
            "speed_setting": w.speed_name,
            "height_above_table": f"{h*100:.0f} cm ({band(h, cfg.bands.height)})",
            "gripper": "open" if w.arm.gripper > 0.03 else "closed",
            "state": "frozen by the safety layer" if w.arm.frozen else ("holding (no fresh decisions)" if w.ladder != "fresh" else "operating"),
        },
        "objects": objects,
        "relations": relations,
        "recent_actions": w.recent[-4:] or ["(just started)"],
    }
    return state
