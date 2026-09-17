"""The primitive library: the closed action vocabulary Jev sequences at run time.

Each primitive is a parameterised routine code can execute and verify. Code offers a primitive as
an option only when its preconditions hold, resolves the pick into a goal (xyz + gripper), and
reports done / failed with a reason. New verbs in natural language become sequences of these,
chosen tick by tick; no primitive is written per task.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from robojev.config import Config
from robojev.world import World

SAFETY = {"hold": "stay where it is", "back_off": "move horizontally away from the current target",
          "rise_away": "go up, away from the table and everything on it"}

PLACE_RELATIONS = {"left_of": (0.0, 1.0), "right_of": (0.0, -1.0), "in_front_of": (-1.0, 0.0), "behind": (1.0, 0.0)}
PLACE_WORDS = {"left_of": "to the robot's left of", "right_of": "to the robot's right of",
               "in_front_of": "in front of (between the robot and)", "behind": "behind (further from the robot than)"}


@dataclass
class Prim:
    key: str            # option key as offered to Jev, e.g. "move_above:white cup-like object C"
    name: str
    subject: str | None
    text: str           # criteria text


def _above(w: World, label: str, tol: float = 0.025) -> bool:
    e = w.entity(label)
    return e is not None and math.hypot(w.arm.ee[0] - e.xyz[0], w.arm.ee[1] - e.xyz[1]) < tol


def carry_z(cfg: Config, w: World) -> float:
    tallest = max([e.height_m for e in w.entities] + [0.0])
    return min(cfg.workspace.z[1], w.table_z + max(cfg.motion.carry_height, tallest + cfg.motion.object_clearance + 0.05))


def hover_z(cfg: Config, w: World) -> float:
    tallest = max([e.height_m for e in w.entities] + [0.0])
    return min(cfg.workspace.z[1], w.table_z + max(cfg.motion.hover_heights["high"], tallest + cfg.motion.object_clearance))


def grasp_z(cfg: Config, w: World, label: str) -> float:
    e = w.entity(label)
    h = e.height_m if e else 0.08
    return max(cfg.workspace.z[0], w.table_z + h * cfg.motion.grasp_fraction)


def resolve_place(cfg: Config, w: World, place: str | None, origin_xy) -> tuple[float, float] | None:
    """Turn a place pick into xy in base frame, or None if it cannot be resolved."""
    if not place or place == "unspecified":
        return None
    if place == "where_it_was":
        return tuple(origin_xy) if origin_xy else None
    rel, _, label = place.partition(":")
    e = w.entity(label)
    if e is None or rel not in PLACE_RELATIONS:
        return None
    dx, dy = PLACE_RELATIONS[rel]
    gap = cfg.motion.place_gap + e.width_m / 2
    x, y = e.xyz[0] + dx * gap, e.xyz[1] + dy * gap
    # keep the spot inside the box; a spot that had to move more than 6 cm is not that place any more
    cx, cy, _ = cfg.workspace.clamp((x, y, cfg.workspace.z[0] + 0.001))
    if math.hypot(cx - x, cy - y) > 0.06:
        return None
    return (cx, cy)


def offered(cfg: Config, w: World, brain) -> list[Prim]:
    """Primitives whose preconditions hold right now. Safety picks are always included."""
    out = []
    holding = w.holding_label
    gripper_open = w.gripper_state == "open"
    height = w.arm.ee[2] - w.table_z
    place_xy = resolve_place(cfg, w, brain.place, brain.origin_xy)
    for e in w.entities:
        if not e.reachable:
            continue
        if holding is None:
            out.append(Prim(f"move_above:{e.label}", "move_above", e.label, f"move to hover above {e.label}"))
            if gripper_open and _above(w, e.label):
                out.append(Prim(f"descend_to_grasp:{e.label}", "descend_to_grasp", e.label,
                                f"lower the open gripper down around {e.label}, ready to grasp it"))
    if holding is None and gripper_open and w.above_label and height < (w.entity(w.above_label).height_m + 0.03 if w.entity(w.above_label) else 0.0):
        out.append(Prim("close_gripper", "close_gripper", w.above_label, f"close the gripper to grasp {w.above_label}"))
    if holding is not None:
        if height < cfg.motion.carry_height - 0.02:
            out.append(Prim("lift", "lift", holding, f"raise {holding} up to carrying height"))
        if place_xy is not None and math.hypot(w.arm.ee[0] - place_xy[0], w.arm.ee[1] - place_xy[1]) > 0.02:
            out.append(Prim("move_to_place", "move_to_place", holding, f"carry {holding} to the place the user asked for ({brain.place})"))
        if place_xy is not None and math.hypot(w.arm.ee[0] - place_xy[0], w.arm.ee[1] - place_xy[1]) <= 0.03:
            out.append(Prim("lower_to_place", "lower_to_place", holding, f"lower {holding} onto the table at the place"))
    if w.gripper_state != "open":
        out.append(Prim("open_gripper", "open_gripper", holding, "open the gripper" + (f", releasing {holding}" if holding else "")))
    if height < cfg.motion.safe_height - 0.02:
        out.append(Prim("retreat", "retreat", None, "rise straight up to a safe height, e.g. after releasing an object"))
    for k, v in SAFETY.items():
        out.append(Prim(k, k, None, v))
    return out


def goal_for(cfg: Config, w: World, brain, now: float):
    """(goal_xyz, gripper or None, done: bool, fail: str | None, reason) for the brain's current primitive."""
    name, subj = brain.prim, brain.prim_subject
    sp = w.arm.setpoint
    ee = w.arm.ee
    tz = w.table_z
    age = now - (brain.prim_started_t or now)
    def near(goal, xy_tol=0.015, z_tol=0.015):
        return math.hypot(ee[0] - goal[0], ee[1] - goal[1]) < xy_tol and abs(ee[2] - goal[2]) < z_tol
    if name == "move_above":
        e = w.entity(subj)
        if e is None:
            return sp, None, False, f"{subj} is no longer known", "move_above: lost subject"
        goal = cfg.workspace.clamp((e.xyz[0], e.xyz[1], hover_z(cfg, w)))
        return goal, None, near(goal), None, f"move_above {subj}"
    if name == "descend_to_grasp":
        e = w.entity(subj)
        if e is None:
            return sp, None, False, f"{subj} is no longer known", "descend: lost subject"
        goal = cfg.workspace.clamp((e.xyz[0], e.xyz[1], grasp_z(cfg, w, subj)))
        return goal, 0.04, near(goal, 0.02, 0.01), None, f"descend_to_grasp {subj}"
    if name == "close_gripper":
        done = age > cfg.motion.gripper_settle_s
        fail = None
        if done and w.holding_label is None:
            fail = "closed on nothing"
        return sp, 0.0, done, fail, "close_gripper"
    if name == "lift":
        goal = (sp[0], sp[1], carry_z(cfg, w))
        return goal, None, near(goal), None, f"lift {subj}"
    if name == "move_to_place":
        xy = resolve_place(cfg, w, brain.place, brain.origin_xy)
        if xy is None:
            return sp, None, False, "place cannot be resolved", "move_to_place: no place"
        goal = cfg.workspace.clamp((xy[0], xy[1], carry_z(cfg, w)))
        return goal, None, near(goal), None, f"move_to_place {brain.place}"
    if name == "lower_to_place":
        held = w.entity(w.holding_label) if w.holding_label else None
        h = held.height_m if held else 0.08
        goal = (sp[0], sp[1], max(cfg.workspace.z[0], tz + h * cfg.motion.grasp_fraction + 0.01))
        return goal, None, near(goal, 0.02, 0.01), None, "lower_to_place"
    if name == "open_gripper":
        done = age > cfg.motion.gripper_settle_s
        return sp, 0.04, done, None, "open_gripper"
    if name == "retreat":
        goal = (sp[0], sp[1], min(cfg.workspace.z[1], tz + cfg.motion.safe_height))
        return goal, None, near(goal), None, "retreat"
    if name == "rise_away":
        goal = (sp[0], sp[1], min(cfg.workspace.z[1], tz + cfg.motion.safe_height))
        return goal, None, near(goal), None, "rise_away"
    if name == "back_off":
        tgt = w.entity(brain.target) if brain.target else None
        if tgt is None:
            return sp, None, True, None, "back_off (no target)"
        dx, dy = ee[0] - tgt.xyz[0], ee[1] - tgt.xyz[1]
        L = math.hypot(dx, dy); ux, uy = (dx / L, dy / L) if L > 1e-6 else (-1.0, 0.0)
        goal = cfg.workspace.clamp((ee[0] + ux * cfg.motion.back_off_distance, ee[1] + uy * cfg.motion.back_off_distance, max(sp[2], hover_z(cfg, w))))
        return goal, None, near(goal), None, "back_off"
    return sp, None, True, None, "hold"
