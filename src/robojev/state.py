"""Render the World into the Jev situation report (a JSON object).

Layout follows the evidence: standing orders first (08 §E8, strongest effect at the top), a static
glossary, the user's task text in its own key marked as user-supplied, then per-tick facts with
numbers paired with bands. Geometry facts (closest, between) come from code.
"""
from __future__ import annotations

from robojev.config import Config
from robojev.world import World, band, bearing_words

STATE_VERSION = "v1"


def glossary(cfg: Config) -> dict:
    return {
        "robot": "a WidowX AI tabletop robot arm with a two-finger gripper 4 cm wide; a wrist camera looks past the fingers",
        "units": "distances in centimetres; bearings in degrees",
        "distance_bands": {name: f"under {int(up*100)} cm" if up != float('inf') else "50 cm or more"
                           for name, up in cfg.bands.distance},
        "bearing": "horizontal direction from the gripper: 0 is straight ahead (away from the robot base), positive is left, negative is right",
        "height": "the gripper's height above the table surface",
        "control": "decisions update every 0.1 s; a decision persists until the next one arrives; code enforces speed and workspace limits",
        "primitives": "the arm acts by running one primitive at a time (move above, descend to grasp, close gripper, lift, move to place, lower to place, set down here, open gripper, retreat); each runs until done or failed, then the next is chosen",
        "gripper": "the gripper can hold one object; `holding` says which",
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
        if e.flat:
            item["on_it"] = [o.label for o in w.entities if not o.flat and o.resting_on == e.label] or "nothing"
        else:
            item["resting_on"] = e.resting_on or "the bare table"
        if e.speed_mps > 0.02 and e.in_view:   # motion is only a fact while we can see it
            item["motion"] = f"moving, about {e.speed_mps*100:.0f} cm/s"
        if e.known_s < 5 and w.uptime_s > 8:
            item["status"] += f"; appeared {e.known_s:.0f} s ago (was not there before)"
        objects.append(item)
    tgt = w.entity(w.committed_target) if w.committed_target else None
    relations = {
        "closest_to_gripper": (w.closest().label if w.closest() else None),
        "current_target": w.committed_target or "none yet",
        "between_gripper_and_target": w.between(tgt) if tgt else [],
    }
    if tgt:
        relations["gripper_offset_from_target"] = (f"{tgt.horizontal_m*100:.0f} cm horizontally "
                                                   f"({band(tgt.horizontal_m, cfg.bands.distance)}), "
                                                   f"{h*100:.0f} cm above the table")
    state = {
        "standing_orders": list(cfg.orders.default) + list(w.orders),
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
            "gripper": w.gripper_state,
            "holding": w.holding_label or "nothing",
            "directly_above": w.above_label or "nothing",
            "place_for_the_object": w.place or "not decided",
            "current_primitive": (f"{w.prim['name']}" + (f" {w.prim['subject']}" if w.prim.get('subject') else "")
                                  + f" ({w.prim['status']}, {w.prim['age_s'] or 0:.1f} s)") if w.prim.get("name") else "none",
            "last_result": w.prim.get("last_result") or "none yet",
            "completed_so_far": (f"{w.placed[0]} was picked up and put down at {w.placed[1]} {w.t - w.placed[2]:.0f} s ago"
                                 if w.placed else "nothing placed yet"),
            "state": "frozen by the safety layer" if w.arm.frozen else ("holding (no fresh decisions)" if w.ladder != "fresh" else "operating"),
        },
        "objects": objects,
        "relations": relations,
        "recent_actions": w.recent[-4:] or ["(just started)"],
    }
    return state
