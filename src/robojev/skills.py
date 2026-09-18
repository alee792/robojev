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
SHIFTS = {"shift_left": (0.0, 1.0), "shift_right": (0.0, -1.0), "shift_away": (1.0, 0.0), "shift_closer": (-1.0, 0.0)}
SHIFT_WORDS = {"shift_left": "a short way to the robot's left of where it was picked up", "shift_right": "a short way to the robot's right of where it was picked up",
               "shift_away": "a short way further from the robot than where it was picked up", "shift_closer": "a short way closer to the robot than where it was picked up"}
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
    # the box's top edge is at the envelope's limit; keep 2 cm inside it so goals stay reachable
    tallest = max([e.height_m for e in w.entities if e.label != w.holding_label] + [0.0])
    return min(cfg.workspace.z[1] - 0.02, w.table_z + max(cfg.motion.carry_height, tallest + cfg.motion.object_clearance + 0.03))


def hover_z(cfg: Config, w: World) -> float:
    tallest = max([e.height_m for e in w.entities] + [0.0])
    return min(cfg.workspace.z[1] - 0.02, w.table_z + max(cfg.motion.hover_heights["high"], tallest + cfg.motion.object_clearance))


def side_grasp(cfg: Config, e) -> bool:
    """Objects too wide to drop the fingers around from above are grasped from the side, wrist level."""
    return e is not None and e.width_m >= cfg.motion.side_grasp_min_width and e.height_m >= 0.05


def side_z(cfg: Config, w: World) -> float:
    return max(cfg.workspace.z[0], w.table_z + cfg.motion.side_grasp_height)


def _level(w: World, tol: float = 0.12, cfg: Config | None = None) -> bool:
    """Wrist at the side-grasp pitch (the setpoint has arrived there)."""
    hi = cfg.motion.side_pitch if cfg else 0.5
    lo = min(cfg.motion.side_pitch_advance, cfg.motion.side_pitch_far) if cfg else 0.25
    return w.arm.pitch is not None and (lo - tol) < w.arm.pitch < (hi + tol)


def _behind(w: World, e, max_back: float, dy_tol: float = 0.03) -> bool:
    """EE is behind e (toward the robot), within max_back of its near edge and roughly on its centre line."""
    back = e.xyz[0] - w.arm.ee[0]
    return abs(e.xyz[1] - w.arm.ee[1]) < dy_tol and 0.0 < back < e.width_m / 2 + max_back


def grasp_z(cfg: Config, w: World, label: str) -> float:
    e = w.entity(label)
    h = e.height_m if e else 0.08
    return max(cfg.workspace.z[0], w.table_z + h * cfg.motion.grasp_fraction)


def resolve_place(cfg: Config, w: World, place: str | None, origin_xy, last_set_down_xy=None) -> tuple[float, float] | None:
    """Turn a place pick into xy in base frame, or None if it cannot be resolved."""
    if not place or place == "unspecified":
        return None
    if place == "where_it_was":
        return tuple(origin_xy) if origin_xy else None
    if place == "where_it_was_set_down":
        return tuple(last_set_down_xy) if last_set_down_xy else None
    if place == "somewhere_else":
        if not origin_xy:
            return None
        import random
        ws = cfg.workspace
        rng = random.Random(int(origin_xy[0] * 1000) ^ int(origin_xy[1] * 1000) ^ int(w.t * 10))
        best = None
        for _ in range(200):
            (x0, x1), (y0, y1) = cfg.motion.place_region
            x = rng.uniform(x0, x1); y = rng.uniform(y0, y1)
            d_origin = math.hypot(x - origin_xy[0], y - origin_xy[1])
            d_obj = min([math.hypot(x - e.xyz[0], y - e.xyz[1]) for e in w.entities if e.label != w.holding_label] + [9.0])
            if d_origin < 0.15 or d_obj < 0.12:
                continue
            cx0, cy0 = (x0 + x1) / 2, 0.0
            score = min(d_origin, 0.25) + min(d_obj, 0.20) - 0.8 * math.hypot(x - cx0, y - cy0)
            if best is None or score > best[0]:
                best = (score, x, y)
        return (best[1], best[2]) if best else None
    rel2, _, label2 = place.partition(":")
    if rel2 in ("on", "off") and label2:
        m = w.entity(label2)
        if m is None or not m.flat or not m.footprint:
            return None
        fp = m.footprint
        others = [e for e in w.entities if not e.flat and e.label != w.holding_label]
        def free(x, y, d=0.10):
            return all(math.hypot(x - o.xyz[0], y - o.xyz[1]) >= d for o in others)
        cands = []
        if rel2 == "on":
            for i in range(9):
                for j in range(9):
                    x = fp[0] + 0.05 + (fp[1] - fp[0] - 0.10) * i / 8; y = fp[2] + 0.05 + (fp[3] - fp[2] - 0.10) * j / 8
                    cands.append((x, y))
            cx, cy = (fp[0] + fp[1]) / 2, (fp[2] + fp[3]) / 2
            cands.sort(key=lambda c: math.hypot(c[0] - cx, c[1] - cy))   # the middle first
        else:
            (x0, x1), (y0, y1) = cfg.motion.place_region
            here = origin_xy or (w.arm.ee[0], w.arm.ee[1])
            for i in range(14):
                for j in range(14):
                    x = x0 + (x1 - x0) * i / 13; y = y0 + (y1 - y0) * j / 13
                    if fp[0] - 0.06 <= x <= fp[1] + 0.06 and fp[2] - 0.06 <= y <= fp[3] + 0.06:
                        continue          # not on the mat, and clear of its edge
                    cands.append((x, y))
            cands.sort(key=lambda c: math.hypot(c[0] - here[0], c[1] - here[1]))   # the nearest spot off the mat
        for x, y in cands:
            if cfg.workspace.contains((x, y, cfg.workspace.z[0] + 0.001), margin=0.0) and free(x, y):
                return (x, y)
        return None
    if place in SHIFTS:
        if not origin_xy:
            return None
        dx, dy = SHIFTS[place]
        x, y = origin_xy[0] + dx * cfg.motion.shift_distance, origin_xy[1] + dy * cfg.motion.shift_distance
        cx, cy, _ = cfg.workspace.clamp((x, y, cfg.workspace.z[0] + 0.001))
        return (cx, cy) if math.hypot(cx - origin_xy[0], cy - origin_xy[1]) > 0.06 else None
    rel, _, label = place.partition(":")
    e = w.entity(label)
    if e is None or rel not in PLACE_RELATIONS:
        return None
    dx, dy = PLACE_RELATIONS[rel]
    gap = cfg.motion.place_gap + e.width_m / 2
    x, y = e.xyz[0] + dx * gap, e.xyz[1] + dy * gap
    # keep the spot inside the box; a spot that had to move more than 6 cm is not that place any more
    cx, cy, _ = cfg.workspace.clamp((x, y, cfg.workspace.z[0] + 0.001))
    if math.hypot(cx - x, cy - y) > 0.08:
        return None
    return (cx, cy)


def offered(cfg: Config, w: World, brain) -> list[Prim]:
    """Primitives whose preconditions hold right now. Safety picks are always included."""
    out = []
    holding = w.holding_label
    gripper_open = w.gripper_state == "open"
    height = w.arm.ee[2] - w.table_z
    place_xy = (brain.place_xy if (brain.prim in ("move_to_place", "lower_to_place") and brain.place_xy) else None) \
        or resolve_place(cfg, w, brain.place, brain.origin_xy, getattr(brain, 'last_set_down_xy', None))
    for e in w.entities:
        if not e.reachable or e.flat:
            continue
        if holding is None and side_grasp(cfg, e):
            out.append(Prim(f"approach_side:{e.label}", "approach_side", e.label,
                            f"come down level with the table and line up behind {e.label}, ready to reach it from the side"))
            if gripper_open and _level(w, cfg=cfg) and _behind(w, e, cfg.motion.side_standoff + 0.03) and height < cfg.motion.side_grasp_height + 0.03:
                out.append(Prim(f"advance_to_grasp:{e.label}", "advance_to_grasp", e.label,
                                f"slide the open gripper forward around {e.label} from the side, ready to grasp it"))
        elif holding is None:
            out.append(Prim(f"move_above:{e.label}", "move_above", e.label, f"move to hover above {e.label}"))
            if gripper_open and _above(w, e.label):
                out.append(Prim(f"descend_to_grasp:{e.label}", "descend_to_grasp", e.label,
                                f"lower the open gripper down around {e.label}, ready to grasp it"))
    if holding is None and gripper_open and w.above_label and height < (w.entity(w.above_label).height_m + 0.03 if w.entity(w.above_label) else 0.0) \
            and (not side_grasp(cfg, w.entity(w.above_label)) or (_level(w, cfg=cfg) and height < cfg.motion.side_grasp_height + 0.03)):
        out.append(Prim("close_gripper", "close_gripper", w.above_label, f"close the gripper to grasp {w.above_label}"))
    if holding is not None:
        if height < cfg.motion.carry_height - 0.02:
            out.append(Prim("lift", "lift", holding, f"raise {holding} up to carrying height"))
        if place_xy is not None and math.hypot(w.arm.ee[0] - place_xy[0], w.arm.ee[1] - place_xy[1]) > 0.02:
            out.append(Prim("move_to_place", "move_to_place", holding, f"carry {holding} to the place the user asked for ({brain.place})"))
        if place_xy is not None and math.hypot(w.arm.ee[0] - place_xy[0], w.arm.ee[1] - place_xy[1]) <= 0.03:
            out.append(Prim("lower_to_place", "lower_to_place", holding, f"lower {holding} onto the table at the place"))
    if holding is not None:
        held = w.entity(holding)
        dz = brain.grasp_dz if getattr(brain, 'grasp_dz', None) is not None else (held.height_m if held else 0.08) * cfg.motion.grasp_fraction
        low = height < dz + 0.04
        if not low:
            out.append(Prim("set_down_here", "set_down_here", holding, f"lower {holding} onto the table right here (e.g. to abort or if the place cannot be reached)"))
        else:
            out.append(Prim("open_gripper", "open_gripper", holding, f"open the gripper, releasing {holding} onto the table"))
    elif w.gripper_state != "open":
        out.append(Prim("open_gripper", "open_gripper", None, "open the gripper (it is holding nothing)"))
    if height < cfg.motion.safe_height - 0.02:
        out.append(Prim("retreat", "retreat", None, "rise straight up to a safe height, e.g. after releasing an object"))
    at_survey = (math.hypot(w.arm.ee[0] - cfg.motion.hover_start[0], w.arm.ee[1] - cfg.motion.hover_start[1]) < 0.03
                 and height > cfg.motion.safe_height - 0.03 and w.arm.pitch is not None and abs(w.arm.pitch - cfg.motion.hover_pitch) < 0.1)
    if holding is None and not at_survey:
        out.append(Prim("survey", "survey", None, "go back to the survey pose, where the camera sees the whole table: use it when the target is gone, cannot be found, or the scene needs a fresh look"))
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
        brain.pitch = cfg.motion.down_orientation[1]
        goal = cfg.workspace.clamp((min(e.xyz[0], cfg.motion.down_reach_x), e.xyz[1], hover_z(cfg, w)))
        return goal, None, near(goal), None, f"move_above {subj}"
    if name == "descend_to_grasp":
        e = w.entity(subj)
        if e is None:
            return sp, None, False, f"{subj} is no longer known", "descend: lost subject"
        brain.pitch = cfg.motion.down_orientation[1]
        goal = cfg.workspace.clamp((min(e.xyz[0], cfg.motion.down_reach_x), e.xyz[1], grasp_z(cfg, w, subj)))
        return goal, 0.04, near(goal, 0.02, 0.01), None, f"descend_to_grasp {subj}"
    if name == "approach_side":
        e = w.entity(subj)
        if e is None:
            return sp, None, False, f"{subj} is no longer known", "approach_side: lost subject"
        brain.pitch = cfg.motion.side_pitch
        goal = cfg.workspace.clamp((e.xyz[0] - (e.width_m / 2 + cfg.motion.side_standoff), e.xyz[1], side_z(cfg, w)))
        return goal, 0.04, near(goal) and _level(w, 0.05, cfg), None, f"approach_side {subj}"
    if name == "advance_to_grasp":
        e = w.entity(subj)
        if e is None:
            return sp, None, False, f"{subj} is no longer known", "advance: lost subject"
        brain.pitch = cfg.motion.side_pitch_far if e.xyz[0] > cfg.motion.side_far_x else cfg.motion.side_pitch_advance
        goal = cfg.workspace.clamp((e.xyz[0] - cfg.motion.side_grasp_depth, e.xyz[1], side_z(cfg, w)))
        if brain.advance_f0 is None and age > 1.0:
            brain.advance_f0 = float(w.arm.ext_force[0])
        fx = float(w.arm.ext_force[0]) - (brain.advance_f0 if brain.advance_f0 is not None else float(w.arm.ext_force[0]))
        close_enough = (e.xyz[0] - ee[0]) < e.width_m / 2 + 0.04   # contact is only plausible near the object
        if fx > cfg.motion.advance_push_n and age > 1.3 and close_enough:
            return sp, 0.04, False, f"the fingers are pushing {subj} (F_x +{fx:.0f} N)", "advance: pushing"
        return goal, 0.04, near(goal, 0.012, 0.01), None, f"advance_to_grasp {subj}"
    if name == "close_gripper":
        done = age > cfg.motion.gripper_settle_s
        fail = None
        if done and w.holding_label is None:
            fail = "closed on nothing"
        e = w.entity(subj) if subj else None
        target = max(0.026, min(0.035, ((e.width_m if e else 0.0) - cfg.motion.grip_squeeze) / 2))   # per-carriage travel: the pads stop on the object before reaching it; floor 5.2 cm gap (a low width estimate closed to 2.2 cm on the cup: real run 13)
        return sp, target, done, fail, "close_gripper"
    if name == "lift":
        goal = (sp[0], sp[1], carry_z(cfg, w))
        return goal, None, near(goal), None, f"lift {subj}"
    if name == "move_to_place":
        xy = brain.place_xy or resolve_place(cfg, w, brain.place, brain.origin_xy, getattr(brain, 'last_set_down_xy', None))
        if xy is None:
            return sp, None, False, "place cannot be resolved", "move_to_place: no place"
        goal = cfg.workspace.clamp((xy[0], xy[1], carry_z(cfg, w)))
        return goal, None, near(goal), None, f"move_to_place {brain.place}"
    if name == "set_down_here":
        held = w.entity(w.holding_label) if w.holding_label else None
        h = held.height_m if held else 0.08
        dz = brain.grasp_dz if brain.grasp_dz is not None else h * cfg.motion.grasp_fraction
        goal = (sp[0], sp[1], max(cfg.workspace.z[0], tz + dz + 0.005))
        return goal, None, near(goal, 0.02, 0.01), None, "set_down_here"
    if name == "lower_to_place":
        held = w.entity(w.holding_label) if w.holding_label else None
        h = held.height_m if held else 0.08
        dz = brain.grasp_dz if brain.grasp_dz is not None else h * cfg.motion.grasp_fraction
        goal = (sp[0], sp[1], max(cfg.workspace.z[0], tz + dz + 0.005))
        return goal, None, near(goal, 0.02, 0.01), None, "lower_to_place"
    if name == "open_gripper":
        done = age > cfg.motion.gripper_settle_s
        return sp, 0.04, done, None, "open_gripper"
    if name == "retreat":
        if _level(w, cfg=cfg):
            # wrist level with the open fingers around something: back out before rising, or the
            # fingers lift the object's rim
            for e in w.entities:
                if e.label != w.holding_label and abs(e.xyz[1] - ee[1]) < e.width_m / 2 + 0.02 and -(e.width_m / 2 + 0.02) < e.xyz[0] - ee[0] < e.width_m / 2 + 0.05:
                    goal = cfg.workspace.clamp((e.xyz[0] - (e.width_m / 2 + cfg.motion.side_standoff), sp[1], sp[2]))
                    if near(goal, 0.02, 0.05):
                        break   # backed out as far as the box allows: rise rather than chase the next
                        # object into the workspace edge (it stalled there: sim mat run 4)
                    return goal, None, False, None, f"retreat: backing out from {e.label}"
        goal = (sp[0], sp[1], min(cfg.workspace.z[1], tz + cfg.motion.safe_height))
        return goal, None, near(goal), None, "retreat"
    if name == "survey":
        brain.pitch = cfg.motion.hover_pitch
        goal = cfg.workspace.clamp((cfg.motion.hover_start[0], cfg.motion.hover_start[1], tz + cfg.motion.safe_height))
        return goal, None, near(goal, 0.03, 0.03) and w.arm.pitch is not None and abs(w.arm.pitch - cfg.motion.hover_pitch) < 0.05, None, "survey"
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
