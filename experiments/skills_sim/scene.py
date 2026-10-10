"""The K1 scene: the WidowX AI follower from trossen_arm_mujoco on a table, free-body blocks and a
tray of slots. Physics, not kinematic attachment: the fingers' collision boxes hold the blocks.

Base frame as on the real arm: +x forward, +y left, +z up, metres; the table is at z = TABLE_Z.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIM_DIR = Path(os.environ.get("TROSSEN_ARM_MUJOCO_DIR", Path.home() / "trossen_arm_mujoco"))
FOLLOWER_XML = SIM_DIR / "trossen_arm_mujoco/assets/wxai/wxai_follower.xml"
TABLE_Z = -0.02          # table surface in base frame (the base plate sits 2 cm proud, as in the bay)


@dataclass
class Block:
    name: str
    xy: tuple[float, float]
    yaw: float = 0.0                  # rad about +z
    size: float = 0.04                # edge, m
    mass: float = 0.03                # kg (a 4 cm wooden cube)
    rgba: tuple = (0.85, 0.25, 0.2, 1)


@dataclass
class Tray:
    """A row of square slots marked on the table (visual only: placing is judged by position)."""
    origin: tuple[float, float] = (0.36, 0.12)     # centre of slot 0
    step: tuple[float, float] = (0.0, -0.06)       # slot i is at origin + i * step
    n: int = 5
    slot: float = 0.05

    def slot_xy(self, i: int) -> tuple[float, float]:
        return (self.origin[0] + i * self.step[0], self.origin[1] + i * self.step[1])


@dataclass
class SceneSpec:
    blocks: list[Block] = field(default_factory=list)
    tray: Tray | None = field(default_factory=Tray)


def build(spec: SceneSpec):
    """Compile the scene. Returns the MjModel."""
    import mujoco
    if not FOLLOWER_XML.exists():
        raise FileNotFoundError(f"{FOLLOWER_XML} (set TROSSEN_ARM_MUJOCO_DIR to a trossen_arm_mujoco clone)")
    s = mujoco.MjSpec.from_file(str(FOLLOWER_XML))
    s.modelname = "robojev_k1"
    s.option.timestep = 0.002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    s.option.impratio = 10.0           # stiffer friction: blocks don't creep out of the grip
    w = s.worldbody
    w.add_light(pos=[0.3, 0, 1.2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, 0.01], pos=[0.3, 0, TABLE_Z - 0.01],
               rgba=[0.75, 0.6, 0.4, 1], friction=[0.8, 0.01, 0.001])
    if spec.tray:
        t = spec.tray
        for i in range(t.n):
            x, y = t.slot_xy(i)
            w.add_geom(name=f"slot_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t.slot / 2, t.slot / 2, 0.0005],
                       pos=[x, y, TABLE_Z + 0.0005], rgba=[0.2, 0.2, 0.25, 1], contype=0, conaffinity=0)
    for b in spec.blocks:
        h = b.size / 2
        body = w.add_body(name=b.name, pos=[b.xy[0], b.xy[1], TABLE_Z + h],
                          quat=[np.cos(b.yaw / 2), 0, 0, np.sin(b.yaw / 2)])
        body.add_freejoint(name=b.name + "_free")
        body.add_geom(name=b.name + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=b.mass,
                      rgba=list(b.rgba), friction=[1.0, 0.01, 0.001], condim=4)
    # finger pads: high friction, torsional resistance, so a held block doesn't spin or slip
    for g in s.geoms:
        if g.parent.name in ("left_carriage_link", "right_carriage_link") or "carriage" in (g.parent.name or ""):
            if g.contype or g.conaffinity:
                g.friction = [1.5, 0.02, 0.002]
                g.condim = 4
    w.add_camera(name="overhead", pos=[0.33, 0.0, 0.80], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    return s.compile()
