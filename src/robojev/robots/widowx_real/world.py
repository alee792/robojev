"""Where the blocks are, for a server with no ground truth. `WorldSource` is what the real server
reads; `StaticWorld` is day one's implementation (typed-in positions, kept current by the server's
own grasp/place bookkeeping); `PerceptionWorld` is the P1 stub.

The bookkeeping (`StaticWorld.track`, called every policy tick with what the arm reports):
  - a block the gripper holds follows the EE: its centre `tip_height` below the fingertips plus half
    its size, exactly where the sim's pinch puts it;
  - a block that stops being held settles on the block centred within ON_TOL_XY under it ("on:<id>",
    the sim's rule) or on the table; anything that was on a block being lifted drops to the table;
  - a closed gripper (width under CLOSED_W) whose pads overlap an unheld block at its height shoves
    it so the pads' outer face (`finger_t` outside the fingertips) just touches it: a push without
    physics, enough for the push policy's progress check.
`where` is computed from these positions as the sim does (slot radius, "on:", "table"), so
conformance's judge reads the same words. None of it is observed: it is what the arm was asked to
do, trusted. The judge says so.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from robojev.protocol import HandObs, ObjectObs, PlaceObs
from robojev.robots.widowx_sim import scene, skills

ON_TOL_XY = 0.015       # a block centred within this over another, resting on it, is "on:" it (sim's rule)
GRASP_RADIUS = 0.02     # a block centred within this of the EE, at the fingers' height, is between the fingers
CLOSED_W = 0.02         # a gripper narrower than this pushes with its pads (a block is never under 4 cm)


class WorldSource(Protocol):
    """What the real server reads and what it tells. Also satisfies base.WorldView."""
    table_z: float
    places: dict[str, PlaceObs]
    hands: list[HandObs]

    def ids(self) -> Iterable[str]: ...
    def pose(self, oid: str) -> skills.ObjPose | None: ...
    def poses(self) -> dict[str, list[float]]:
        """{id: [x, y, z, yaw]} for the recorder"""
    def where(self, oid: str, held_by: dict[str, str]) -> str: ...
    def place_xy(self, pid: str, held_by: dict[str, str]) -> tuple[float, float] | None: ...
    def occupant(self, pid: str, held_by: dict[str, str]) -> str | None: ...
    def objects(self, held_by: dict[str, str]) -> dict[str, ObjectObs]: ...
    def under(self, ee: tuple[float, float, float]) -> str | None:
        """the block between the fingers at this EE pose, if any"""
    def track(self, ee: tuple[float, float, float], held: str | None, width: float, p: skills.Params) -> None:
        """one tick of bookkeeping from what the arm reports; a perceiving world ignores it"""


@dataclass
class _Block:
    id: str
    name: str
    label: str | None
    x: float
    y: float
    z: float
    yaw: float
    size: float
    on: str | None = None       # resting on this block
    carried: bool = False       # following the EE


class StaticWorld:
    """Typed-in block positions (scene.Block: name, xy, yaw, size) and a tray, kept current by
    `track`. `hands` is whatever `set_hands` was last given: day one has no hand perception."""

    def __init__(self, blocks: Iterable[scene.Block], tray: scene.Tray | None = scene.Tray(), table_z: float = scene.TABLE_Z):
        self.table_z = table_z
        self.places: dict[str, PlaceObs] = tray.places() if tray else {}
        self.hands: list[HandObs] = []
        self._blocks: dict[str, _Block] = {}
        for b in blocks:
            tail = b.name.rsplit("_", 1)[-1]
            self._blocks[b.name] = _Block(b.name, b.name.replace("_", " "), tail if len(tail) == 1 or tail.isdigit() else None,
                                          b.xy[0], b.xy[1], table_z + b.size / 2, b.yaw, b.size)

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticWorld":
        blocks, tray, table_z = load_blocks(path)
        return cls(blocks, tray, table_z)

    # -- reads ---------------------------------------------------------------------------------------
    def ids(self) -> Iterable[str]:
        return self._blocks.keys()

    def pose(self, oid: str) -> skills.ObjPose | None:
        b = self._blocks.get(oid)
        return None if b is None else skills.ObjPose(b.x, b.y, b.z, b.yaw, b.size)

    def poses(self) -> dict[str, list[float]]:
        return {oid: [b.x, b.y, b.z, b.yaw] for oid, b in self._blocks.items()}

    def where(self, oid: str, held_by: dict[str, str]) -> str:
        if oid in held_by:
            return f"gripper:{held_by[oid]}"
        b = self._blocks[oid]
        if b.on is not None:
            return f"on:{b.on}"
        for pl in self.places.values():
            if pl.kind == "slot" and math.hypot(b.x - pl.x, b.y - pl.y) <= pl.radius:
                return pl.id
        return "table"

    def objects(self, held_by: dict[str, str]) -> dict[str, ObjectObs]:
        return {oid: ObjectObs(oid, b.name, b.x, b.y, b.z, self.where(oid, held_by), label=b.label, size=b.size)
                for oid, b in self._blocks.items()}

    def place_xy(self, pid: str, held_by: dict[str, str]) -> tuple[float, float] | None:
        """A slot's centre, or a group's first free slot (the sim's rule)."""
        pl = self.places.get(pid)
        if pl is None:
            return None
        if pl.kind == "group":
            taken = {self.where(oid, held_by) for oid in self._blocks}
            free = next((m for m in pl.members if m not in taken), None)
            return self.place_xy(free, held_by) if free else None
        return (pl.x, pl.y)

    def occupant(self, pid: str, held_by: dict[str, str]) -> str | None:
        return next((oid for oid in self._blocks if self.where(oid, held_by) == pid), None)

    def under(self, ee: tuple[float, float, float]) -> str | None:
        """The block whose centre is within GRASP_RADIUS of the EE in xy with the fingertips between
        its bottom (1 cm of slack) and its top; the nearest if several. A carried block qualifies: it
        is what the fingers rest on."""
        near = [b for b in self._blocks.values()
                if math.hypot(b.x - ee[0], b.y - ee[1]) <= GRASP_RADIUS
                and b.z - b.size / 2 - 0.01 <= ee[2] <= b.z + b.size / 2]
        b = min(near, key=lambda b: math.hypot(b.x - ee[0], b.y - ee[1]), default=None)
        return None if b is None else b.id

    # -- bookkeeping ---------------------------------------------------------------------------------
    def track(self, ee: tuple[float, float, float], held: str | None, width: float, p: skills.Params) -> None:
        h = self._blocks.get(held) if held else None
        if h is not None:
            lifted = not h.carried
            h.carried, h.on = True, None
            if lifted:                                           # just lifted: whatever sat on it drops
                for o in self._blocks.values():
                    if o.on == h.id:
                        o.on = None
                        self._settle(o)
            h.x, h.y = ee[0], ee[1]
            h.z = ee[2] - p.tip_height + h.size / 2
        for b in self._blocks.values():
            if b is h:
                continue
            if b.carried:                                        # released, or dropped
                b.carried = False
                self._settle(b)
            elif width < CLOSED_W and ee[2] < b.z + b.size / 2:  # the closed pads are at its height
                d = math.hypot(b.x - ee[0], b.y - ee[1])
                reach = b.size / 2 + p.finger_t
                if 1e-9 < d < reach:
                    k = (reach - d) / d
                    dx, dy = (b.x - ee[0]) * k, (b.y - ee[1]) * k
                    self._shift(b, dx, dy)

    def _shift(self, b: _Block, dx: float, dy: float) -> None:
        b.x, b.y = b.x + dx, b.y + dy
        for o in self._blocks.values():                          # a stack moves with its base
            if o.on == b.id:
                self._shift(o, dx, dy)

    def _settle(self, b: _Block) -> None:
        """Rest on the highest other block centred within ON_TOL_XY below, else on the table."""
        base = max((o for o in self._blocks.values() if o is not b and not o.carried
                    and math.hypot(o.x - b.x, o.y - b.y) <= ON_TOL_XY and o.z < b.z),
                   key=lambda o: o.z, default=None)
        if base is None:
            b.on, b.z = None, self.table_z + b.size / 2
        else:
            b.on, b.z = base.id, base.z + base.size / 2 + b.size / 2

    # -- not protocol ----------------------------------------------------------------------------------
    def set_hands(self, hands: list[HandObs]) -> None:
        self.hands = list(hands)

    def move(self, oid: str, x: float, y: float, yaw: float | None = None) -> None:
        """A person moved a block on the table (tests, the day-one script between trials)."""
        b = self._blocks[oid]
        b.x, b.y, b.on, b.carried = x, y, None, False
        if yaw is not None:
            b.yaw = yaw
        self._settle(b)


class PerceptionWorld:
    """Block positions from the scene camera. Not built: that is spike P1 (docs/harness-spikes.md),
    which runs on the Mac with rendering. Day one runs `StaticWorld` with typed-in positions."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("PerceptionWorld: block positions from the scene camera are spike P1 "
                                  "(docs/harness-spikes.md); day one uses StaticWorld with typed-in positions")


def load_blocks(path: str | Path) -> tuple[list[scene.Block], scene.Tray | None, float]:
    """The `--blocks` file: {"table_z": -0.02, "blocks": [{"name": "block_1", "xy": [0.30, -0.08],
    "yaw_deg": 0, "size": 0.04}, ...], "tray": {"origin": [0.36, 0.12], "step": [0, -0.08], "n": 5,
    "slot": 0.05}}. `tray` may be null (no places) or absent (the sim's default tray); `table_z`
    defaults to the sim's -0.02; yaw is typed in degrees because a person reads a protractor."""
    d = json.loads(Path(path).read_text())
    blocks = [scene.Block(b["name"], (float(b["xy"][0]), float(b["xy"][1])), yaw=math.radians(float(b.get("yaw_deg", 0.0))),
                          size=float(b.get("size", 0.04))) for b in d["blocks"]]
    if "tray" not in d:
        tray: scene.Tray | None = scene.Tray()
    elif d["tray"] is None:
        tray = None
    else:
        t = d["tray"]
        tray = scene.Tray(origin=tuple(t.get("origin", (0.36, 0.12))), step=tuple(t.get("step", (0.0, -0.08))),
                          n=int(t.get("n", 5)), slot=float(t.get("slot", 0.05)))
    return blocks, tray, float(d.get("table_z", scene.TABLE_Z))

