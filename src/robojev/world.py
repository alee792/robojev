"""World model: what code knows this tick. Built from the arm snapshot and the tracker; all
geometry (distances, bearings, what is closest, what is between) is computed here, in code."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from robojev.arm import ArmSnapshot
from robojev.config import Config
from robojev.perception.memory import Entity


def band(value: float, table) -> str:
    for name, upper in table:
        if value < upper:
            return name
    return table[-1][0]


def bearing_words(deg: float) -> str:
    a = abs(deg)
    if a <= 10:
        return "dead ahead"
    side = "left" if deg > 0 else "right"
    if a <= 60:
        return f"ahead {side}"
    if a <= 120:
        return side
    return f"behind {side}"


@dataclass
class EntityView:
    id: str
    label: str
    description: str
    xyz: tuple[float, float, float]      # footprint centre, base frame
    horizontal_m: float                  # from the EE, horizontal
    bearing_deg: float                   # from the EE, +left, 0 = +x (away from base)
    height_m: float
    in_view: bool
    last_seen_s: float
    speed_mps: float
    reachable: bool
    width_m: float = 0.08


@dataclass
class World:
    t: float
    arm: ArmSnapshot
    entities: list[EntityView]
    table_z: float
    user_task: str
    orders: list[str]
    committed_target: str | None
    motion: str
    hover_position: str
    hover_height: str
    speed_name: str
    ladder: str                           # "fresh" | "hold" | "rise"
    recent: list[str] = field(default_factory=list)
    avoiding: str | None = None
    gripper_state: str = "open"          # open | closed | closed on something | moving
    holding_label: str | None = None
    above_label: str | None = None
    prim: dict = field(default_factory=dict)   # current primitive: name, subject, status, age, last_result
    place: str | None = None

    def entity(self, label: str) -> EntityView | None:
        for e in self.entities:
            if e.label == label:
                return e
        return None

    def closest(self) -> EntityView | None:
        return min(self.entities, key=lambda e: e.horizontal_m) if self.entities else None

    def between(self, target: EntityView, corridor: float = 0.06) -> list[str]:
        """Entities whose footprint lies within `corridor` of the straight EE->target segment."""
        ee = np.array(self.arm.ee[:2]); tg = np.array(target.xyz[:2])
        seg = tg - ee; L = np.linalg.norm(seg)
        if L < 1e-6:
            return []
        out = []
        for e in self.entities:
            if e.label == target.label:
                continue
            p = np.array(e.xyz[:2]) - ee
            s = float(p @ seg / L)
            if 0.02 < s < L - 0.02 and abs(float(np.cross(seg / L, p))) < corridor + e.xyz[2] * 0:
                out.append(e.label)
        return out


def build_world(cfg: Config, arm: ArmSnapshot, entities: list[Entity], in_view_fn, table_z: float,
                user_task: str, orders: list[str], brain_state: dict, now: float | None = None) -> World:
    now = now or time.time()
    ee = np.asarray(arm.ee)
    views = []
    for e in entities:
        dx, dy = e.xyz[0] - ee[0], e.xyz[1] - ee[1]
        horiz = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        reach = cfg.workspace.contains((e.xyz[0], e.xyz[1], cfg.workspace.z[0] + 0.001), margin=0.0)
        views.append(EntityView(e.id, e.label(), e.describe(), (float(e.xyz[0]), float(e.xyz[1]), float(e.xyz[2])),
                                horiz, bearing, e.height, in_view_fn(e, now), now - e.last_seen, e.velocity(), reach, e.width))
    views.sort(key=lambda v: v.horizontal_m)
    # phase facts, all from code
    w_ = arm.gripper
    if arm.holding:
        gripper_state = "closed on something"
    elif w_ > 0.03:
        gripper_state = "open"
    elif w_ < 0.008:
        gripper_state = "closed"
    else:
        gripper_state = "moving" if arm.gripper_goal is not None and abs(arm.gripper_goal - w_) > 0.006 else "partly open"
    nearest = min(views, key=lambda v: v.horizontal_m) if views else None
    above = nearest.label if nearest and nearest.horizontal_m < 0.025 else None
    holding_label = None
    if arm.holding:
        # the held object is the one riding with the gripper (its track sits under the EE)
        holding_label = brain_state.get("held") or (nearest.label if nearest and nearest.horizontal_m < 0.05 else "an object")
    return World(now, arm, views, table_z, user_task, orders, brain_state.get("target"), brain_state.get("motion", "hold"),
                 brain_state.get("hover_position", "directly_above"), brain_state.get("hover_height", "high"),
                 brain_state.get("speed_name", "slow"), brain_state.get("ladder", "fresh"), brain_state.get("recent", []),
                 brain_state.get("avoid"), gripper_state, holding_label, above,
                 {"name": brain_state.get("prim"), "subject": brain_state.get("prim_subject"), "status": brain_state.get("prim_status"),
                  "age_s": brain_state.get("prim_age"), "last_result": brain_state.get("last_result")},
                 brain_state.get("place"))
