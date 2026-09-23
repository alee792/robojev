"""World sim (ground truth), the scripted human's hand, and the perception model that renders the Scene.

Frame: a 2-D table in cm seen from above. The robot base sits at (50, 0) facing +y, so smaller x is
the robot's LEFT. Tray slots and bins sit along y = 55; blocks start on the table between y = 10 and
y = 30. Heights (z) are cm above the table; the gripper travels at TRAVEL_Z and grasps at LOW_Z.
Sim time runs at 10 ticks per second.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

TICKS_PER_S = 10
TRAVEL_Z = 15.0        # carrying height
LOW_Z = 2.0            # grasp / place height (block centre)
LOW_CM = 6.0           # below this height the fingers are down among the blocks
HOME = (50.0, 8.0, TRAVEL_Z)
XY_SPEED = {"normal": 2.0, "slow": 1.0}   # cm per tick (20 / 10 cm/s)
Z_SPEED = 3.0
ABOVE_CM = 2.0         # the gripper is "above" something when its xy is this close
FINGER_CM = 4.5        # an open gripper low on the table touches any block centre this close (xy)
TIP_CM = 2.5           # a single fingertip (nudge) touches block centres this close
HAND_CONTACT_CM = 5.0  # gripper within this of a hand = contact (a violation)
FLOOR_CM = 3.0         # code's last-ditch rule: never step toward a hand closer than this (inside the contact zone:
                       # it prevents a collision, not an intrusion; keeping clear is the Spotter's job)
HAND_SLOW_CM = 20.0    # code rule: a hand this close caps the pace at slow, whatever Jev says
SLOT_PITCH = 10.0
TRAY_Y = 55.0
BIN_R = 6.0
PARK_Y = 40.0
TABLE = (0.0, 100.0, 0.0, 70.0)   # x0, x1, y0, y1


def dxy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment ab (xy)."""
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    vx, vy = bx - ax, by - ay
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / L))
    return math.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy))


@dataclass
class Block:
    id: str
    colour: str
    x: float
    y: float
    number: int | None = None
    label: str = ""
    size: str = "small"
    where: str = "table"   # table | slot:K | bin:NAME | gripper


@dataclass
class Hand:
    """The human's hand: reaches from `start` to `target` at `speed` cm/tick, stays, then leaves."""
    start: tuple
    target: tuple
    stay_ticks: int = 30
    speed: float = 4.0
    z: float = TRAVEL_Z
    phase: str = "enter"   # enter | stay | leave | gone
    x: float = 0.0
    y: float = 0.0
    px: float = 0.0
    py: float = 0.0
    waited: int = 0

    def __post_init__(self):
        self.x, self.y = self.start
        self.px, self.py = self.start

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

    def step(self):
        self.px, self.py = self.x, self.y
        if self.phase == "stay":
            self.waited += 1
            if self.waited >= self.stay_ticks:
                self.phase = "leave"
            return
        goal = self.target if self.phase == "enter" else self.start
        d = dxy((self.x, self.y), goal)
        if d <= self.speed:
            self.x, self.y = goal
            self.phase = "stay" if self.phase == "enter" else "gone"
        else:
            self.x += (goal[0] - self.x) * self.speed / d
            self.y += (goal[1] - self.y) * self.speed / d


@dataclass
class Arm:
    x: float = HOME[0]
    y: float = HOME[1]
    z: float = HOME[2]
    holding: str | None = None
    skill: str | None = None             # key of the running skill, e.g. "carry_to:slot:3"
    last_result: str = "nothing run yet"
    dest: tuple | None = None            # (key, x, y) where the last carry_to went
    target: str | None = None            # block id of the last move_above / pick / nudge
    footprint: float = FINGER_CM         # what touches the table when low: open gripper or one fingertip

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

    @property
    def high(self) -> bool:
        return self.z >= TRAVEL_Z - 1


class World:
    def __init__(self, blocks: list[Block], n_slots: int = 0, bins: dict | None = None):
        self.blocks = {b.id: b for b in blocks}
        self.slots = {k: (50 + SLOT_PITCH * (k - (n_slots + 1) / 2), TRAY_Y) for k in range(1, n_slots + 1)}
        self.bins = dict(bins or {})     # name -> (x, y)
        self.arm = Arm()
        self.hand: Hand | None = None
        self.tick = 0

    def slot_has(self, k: int) -> str | None:
        return next((b.id for b in self.blocks.values() if b.where == f"slot:{k}"), None)

    def locate(self, x: float, y: float) -> str:
        """Where a block set down at (x, y) ends up."""
        for k, s in self.slots.items():
            if dxy((x, y), s) <= ABOVE_CM + 0.5 and self.slot_has(k) is None:
                return f"slot:{k}"
        for name, c in self.bins.items():
            if dxy((x, y), c) <= BIN_R:
                return f"bin:{name}"
        return "table"

    def set_down(self, bid: str, x: float, y: float) -> None:
        b = self.blocks[bid]
        b.x, b.y = x, y
        b.where = "gripper"          # so locate() does not see it as occupying its own slot
        b.where = self.locate(x, y)

    def free_spot(self, near=(50.0, PARK_Y), clear=9.0, exclude=()) -> tuple:
        """The table spot nearest `near` at least `clear` cm from every block, slot and bin."""
        others = [(b.x, b.y) for b in self.blocks.values() if b.where != "gripper" and b.id not in exclude]
        fixtures = list(self.slots.values()) + list(self.bins.values())
        best = None
        for x in range(10, 91, 4):
            for y in range(10, 45, 4):
                if any(dxy((x, y), o) < clear for o in others) or any(dxy((x, y), f) < 12 for f in fixtures):
                    continue
                d = dxy((x, y), near)
                if best is None or d < best[0]:
                    best = (d, (float(x), float(y)))
        return best[1] if best else (float(near[0]), float(near[1]))


# ---------------------------------------------------------------- perception -> Scene

@dataclass
class Noise:
    misread: float = 0.0     # P(a block's number is misread this tick)
    drop: float = 0.0        # P(an entity is missing from this tick's Scene)
    jitter_cm: float = 0.0   # position noise, sigma


NOISE_ON = Noise(misread=0.03, drop=0.02, jitter_cm=0.5)


def block_name(colour: str, number, label: str = "") -> str:
    if number is not None:
        return f"block {number}"
    return f"{colour} block" + (f" {label}" if label else "")


class Perception:
    """Renders the Scene from the true world. Noise (off by default) is applied raw, with no tracker, so
    whatever it does reaches the code-computed Facts and the layers' State."""

    def __init__(self, noise: Noise | None = None, seed: int = 0):
        self.noise = noise or Noise()
        self.rng = random.Random(seed)

    def observe(self, w: World) -> dict:
        return render_scene(w, self.noise, self.rng)

    @staticmethod
    def truth(w: World) -> dict:
        return render_scene(w, None, None)


def render_scene(w: World, noise: Noise | None, rng: random.Random | None) -> dict:
    n = noise or Noise()
    blocks = {}
    for b in w.blocks.values():
        if rng and b.where != "gripper" and rng.random() < n.drop:
            continue
        num = b.number
        if rng and num is not None and rng.random() < n.misread:
            num = rng.choice([v for v in range(1, 10) if v != num])
        x, y = (w.arm.x, w.arm.y) if b.where == "gripper" else (b.x, b.y)
        if rng and n.jitter_cm and b.where != "gripper":
            x, y = x + rng.gauss(0, n.jitter_cm), y + rng.gauss(0, n.jitter_cm)
        blocks[b.id] = {"id": b.id, "name": block_name(b.colour, num, b.label), "colour": b.colour, "number": num,
                        "size": b.size, "x": x, "y": y, "where": b.where}
    hand = None
    if w.hand is not None and not (rng and rng.random() < n.drop):
        h = w.hand
        hand = {"id": "hand", "name": "a person's hand", "x": h.x, "y": h.y, "z": h.z, "vx": h.x - h.px, "vy": h.y - h.py}
    slots = {k: {"x": s[0], "y": s[1], "has": next((i for i, e in blocks.items() if e["where"] == f"slot:{k}"), None)}
             for k, s in w.slots.items()}
    bins = {name: {"x": c[0], "y": c[1]} for name, c in w.bins.items()}
    return {"blocks": blocks, "hand": hand, "slots": slots, "bins": bins}
