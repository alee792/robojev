"""The text world: blocks (number, colour, size) on a table, a tray of one-block slots and/or bins, the
arm, a person's hand, and perception with optional noise. Implements core's World interface.

Frame and constants come from E12 (e12_blocksworld/world.py): a 100 x 70 cm table seen from above,
the robot base at (50, 0) facing +y, so smaller x is the robot's LEFT; the tray along y = 55; 10
ticks per second; the gripper travels at TRAVEL_Z and grasps at LOW_Z. E12's Hand (reach in, stay,
withdraw) is reused for the person's hand, with a `purpose`: "reach" (into the workspace) or "take"
(held out still, palm up, for a handover).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from e12_blocksworld.world import (ABOVE_CM, BIN_R, FINGER_CM, FLOOR_CM, HAND_CONTACT_CM, HAND_SLOW_CM, LOW_CM, LOW_Z, SLOT_PITCH,
                                   TICKS_PER_S, TIP_CM, TRAVEL_Z, TRAY_Y, XY_SPEED, Hand, Noise)

from ..core.data import ArmObs, Command, Geometry, HandObs, Obj, Place, WorldState, dxy

BLOCK_CM = 4.0            # block height: a block on another sits this much higher
HOME = (50.0, 12.0, TRAVEL_Z)
BIN_Y_WITH_TRAY = 66.0

GEOMETRY = Geometry(travel_z=TRAVEL_Z, low_z=LOW_Z, low_cm=LOW_CM, finger_cm=FINGER_CM, tip_cm=TIP_CM,
                    hand_stop_cm=FLOOR_CM, hand_slow_cm=HAND_SLOW_CM, slow_speed=XY_SPEED["slow"],
                    workspace=(3.0, 97.0, 3.0, 68.0, LOW_Z - 1, TRAVEL_Z + 15), ticks_per_s=TICKS_PER_S)


@dataclass
class Block:
    id: str
    colour: str
    x: float
    y: float
    number: int | None = None
    size: str = "small"
    where: str = "table"
    z: float = LOW_Z

    @property
    def name(self) -> str:
        return f"block {self.number}" if self.number is not None else f"the {self.colour} block"


class PersonHand(Hand):
    def __init__(self, start, target, stay_ticks=30, speed=4.0, z=TRAVEL_Z, purpose="reach"):
        super().__init__(start=start, target=target, stay_ticks=stay_ticks, speed=speed, z=z)
        self.purpose = purpose
        self.took: str | None = None


@dataclass
class Arm:
    x: float = HOME[0]
    y: float = HOME[1]
    z: float = HOME[2]
    holding: str | None = None
    intent: dict | None = None     # set by the running sim skill (for the scripted person's triggers only)


class SimWorld:
    geometry = GEOMETRY

    def __init__(self, blocks: list[Block], n_slots: int = 0, bins: tuple = (), noise: Noise | None = None, seed: int = 0):
        self.blocks = {b.id: b for b in blocks}
        self.places: dict[str, Place] = {}
        for k in range(1, n_slots + 1):
            x = 50 + SLOT_PITCH * (k - (n_slots + 1) / 2)
            role = " (the leftmost slot)" if k == 1 else " (the rightmost slot)" if k == n_slots else ""
            self.places[f"tray_slot_{k}"] = Place(f"tray_slot_{k}", f"tray slot {k}{role}", "slot", x, TRAY_Y, 1, radius=ABOVE_CM + 0.5)
        if n_slots:
            self.places["tray"] = Place("tray", "the tray (any of its slots)", "group", 50.0, TRAY_Y, 0,
                                        members=tuple(f"tray_slot_{k}" for k in range(1, n_slots + 1)), radius=0)
        by = BIN_Y_WITH_TRAY if n_slots else TRAY_Y
        for i, c in enumerate(bins):
            x = 50 + 25 * (i - (len(bins) - 1) / 2)
            self.places[f"{c}_bin"] = Place(f"{c}_bin", f"the {c} bin", "bin", x, by, 0, radius=BIN_R)
        self.places["table"] = Place("table", "a free spot on the table", "table", 50.0, 25.0, 0, radius=0)
        self.places["person"] = Place("person", "the person's hand", "person", 50.0, 70.0, 0, radius=0)
        self.arm = Arm()
        self.hand: PersonHand | None = None
        self.person = None                      # sim.person.Person, set by the scenario
        self.t = 0
        self.noise = noise or Noise()
        self.rng = random.Random(seed)
        self.misread: dict[str, int] = {}
        for b in self.blocks.values():          # the VLM reads each number once; a misread sticks
            if b.number is not None and self.rng.random() < self.noise.misread:
                self.misread[b.id] = self.rng.choice([n for n in range(1, 13) if n != b.number])
        self.last_seen: dict[str, tuple] = {}
        self._restack()

    # ---------------------------------------------------------------- World interface
    def names(self) -> dict:
        d = {p.id: p.name for p in self.places.values()}
        for b in self.blocks.values():
            n = self.misread.get(b.id)
            d[b.id] = f"block {n}" if n is not None else b.name
        return d

    def advance(self, t: int) -> None:
        self.t = t
        if self.person is not None:
            self.person.act(self, t)
        h = self.hand
        if h is not None:
            h.step()
            if h.purpose == "take" and h.phase == "stay" and self.arm.holding and h.took is None \
                    and dxy((self.arm.x, self.arm.y), (h.x, h.y)) <= 10.0:
                b = self.blocks[self.arm.holding]
                b.where, b.x, b.y, b.z = "person", h.x, h.y, h.z
                self.arm.holding, h.took = None, b.id
                h.waited = max(h.waited, h.stay_ticks - 10)   # withdraw a second later
            if h.phase == "gone":
                self.hand = None

    def state(self) -> WorldState:
        return self._observe(noisy=True)

    def truth(self) -> WorldState:
        return self._observe(noisy=False)

    def execute(self, cmd: Command) -> None:
        a = self.arm
        if cmd.kind == "move":
            nx, ny, nz = cmd.target
            dx, dy = nx - a.x, ny - a.y
            a.x, a.y, a.z = nx, ny, nz
            if a.holding:
                b = self.blocks[a.holding]
                b.x, b.y, b.z = a.x, a.y, a.z
            if cmd.drag and cmd.drag in self.blocks:
                b = self.blocks[cmd.drag]
                b.x, b.y = b.x + dx, b.y + dy
                b.where = self.locate(b.id, b.x, b.y, LOW_Z)
        elif cmd.kind == "grip" and a.holding is None:
            cands = [b for b in self.blocks.values() if b.where not in ("gripper", "person")
                     and dxy((a.x, a.y), (b.x, b.y)) <= ABOVE_CM and abs(b.z - (a.z - 0.0)) <= 3.0 and self.top_of(b.id) is None]
            if cands:
                b = min(cands, key=lambda b: abs(b.z - a.z))
                b.where = "gripper"
                a.holding = b.id
        elif cmd.kind == "release" and a.holding:
            b = self.blocks[a.holding]
            a.holding = None
            b.where = "gripper-released"
            b.x, b.y = a.x, a.y
            b.where = self.locate(b.id, a.x, a.y, a.z)
            self._restack()

    # ---------------------------------------------------------------- geometry
    def top_of(self, bid: str) -> str | None:
        return next((b.id for b in self.blocks.values() if b.where == f"on:{bid}"), None)

    def locate(self, bid: str, x: float, y: float, z: float) -> str:
        """Where an object set down at (x, y), from height z, ends up."""
        for b in self.blocks.values():
            if b.id != bid and b.where not in ("gripper", "person", "gripper-released") and dxy((x, y), (b.x, b.y)) <= ABOVE_CM + 0.5 \
                    and z >= b.z + BLOCK_CM - 1.5 and self.top_of(b.id) in (None, bid):
                return f"on:{b.id}"
        for p in self.places.values():
            if p.kind == "slot" and dxy((x, y), (p.x, p.y)) <= p.radius and not any(
                    o.where == p.id and o.id != bid for o in self.blocks.values()):
                return p.id
            if p.kind == "bin" and dxy((x, y), (p.x, p.y)) <= p.radius:
                return p.id
        return "table"

    def _restack(self):
        for _ in range(3):
            for b in self.blocks.values():
                if b.where.startswith("on:"):
                    base = self.blocks[b.where[3:]]
                    b.z = base.z + BLOCK_CM
                elif b.where not in ("gripper", "person"):
                    b.z = LOW_Z

    def place_xy(self, place_id: str, bid: str | None = None) -> tuple:
        p = self.places[place_id]
        if p.kind == "bin":
            k = sum(1 for b in self.blocks.values() if b.where == place_id and b.id != bid)
            ang = 2.1 * k
            return (p.x + 2.5 * math.cos(ang) * (k > 0), p.y + 2.5 * math.sin(ang) * (k > 0))
        return (p.x, p.y)

    def free_spot(self, near=(50.0, 30.0), clear: float = 9.0, exclude=()) -> tuple:
        others = [(b.x, b.y) for b in self.blocks.values() if b.where not in ("gripper", "person") and b.id not in exclude]
        fixtures = [(p.x, p.y) for p in self.places.values() if p.kind in ("slot", "bin")]
        best = None
        for x in range(8, 93, 4):
            for y in range(8, 45, 4):
                if any(dxy((x, y), o) < clear for o in others) or any(dxy((x, y), f) < 12 for f in fixtures):
                    continue
                d = dxy((x, y), near)
                if best is None or d < best[0]:
                    best = (d, (float(x), float(y)))
        return best[1] if best else (float(near[0]), float(near[1]))

    # ---------------------------------------------------------------- perception
    def _observe(self, noisy: bool) -> WorldState:
        n = self.noise if noisy else Noise()
        objs = {}
        names = self.names() if noisy else {b.id: b.name for b in self.blocks.values()}
        for b in self.blocks.values():
            x, y = b.x, b.y
            if b.where == "gripper":
                x, y = self.arm.x, self.arm.y
            elif noisy and n.drop and self.rng.random() < n.drop and b.id in self.last_seen:
                x, y = self.last_seen[b.id]          # dropped this frame: the tracker keeps the last position
            elif noisy and n.jitter_cm:
                x, y = x + self.rng.gauss(0, n.jitter_cm), y + self.rng.gauss(0, n.jitter_cm)
            if noisy:
                self.last_seen[b.id] = (x, y)
            num = self.misread.get(b.id, b.number) if noisy else b.number
            where = "gripper" if b.where == "gripper-released" else b.where
            objs[b.id] = Obj(b.id, names[b.id], b.colour, num, b.size, x, y, b.z, where)
        h = None
        if self.hand is not None:
            hd = self.hand
            h = HandObs(hd.x, hd.y, hd.z, hd.x - hd.px, hd.y - hd.py, held_out=hd.purpose == "take" and hd.phase == "stay")
        a = self.arm
        return WorldState(self.t, objs, dict(self.places), h, ArmObs(a.x, a.y, a.z, a.holding), GEOMETRY)
