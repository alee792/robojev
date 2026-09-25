"""Data shared by every part of the loop: world state, events, commands, decisions.

Units are cm and simulation ticks. A world state is what perception reports (never ground truth);
`where` is a place id ("tray_slot_3", "red_bin", "table", "person"), "on:<object id>" for a stack,
or "gripper".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- world state


@dataclass
class Obj:
    id: str
    name: str                 # how people and Jev refer to it: "block 5", "the blue cup"
    colour: str
    number: int | None
    size: str
    x: float
    y: float
    z: float
    where: str


@dataclass
class Place:
    id: str
    name: str                 # "tray slot 3 (the leftmost slot)", "the red bin"
    kind: str                 # slot | bin | table | person | group
    x: float
    y: float
    capacity: int             # 1 = one object; 0 = any number
    members: tuple = ()       # a group ("tray") lists its slots
    radius: float = 3.0       # an object whose centre is this close is "in" the place


@dataclass
class HandObs:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    held_out: bool            # palm up and still, held out as if to take something (perception flag)


@dataclass
class ArmObs:
    x: float
    y: float
    z: float
    holding: str | None


@dataclass
class Geometry:
    """Numbers generic code needs about this arm and table (the World supplies them)."""
    travel_z: float
    low_z: float
    low_cm: float             # below this height the fingers are down among the objects
    finger_cm: float          # an open gripper low on the table touches object centres this close
    tip_cm: float             # a single fingertip touches object centres this close
    hand_stop_cm: float       # the code floor: never step toward a hand closer than this
    hand_slow_cm: float       # a hand this close caps the speed at slow
    slow_speed: float         # cm per tick
    workspace: tuple          # x0, x1, y0, y1, z0, z1
    ticks_per_s: int


@dataclass
class WorldState:
    t: int
    objects: dict             # id -> Obj
    places: dict              # id -> Place
    hand: HandObs | None
    arm: ArmObs
    geometry: Geometry

    def obj(self, oid: str) -> Obj | None:
        return self.objects.get(oid)

    def occupant(self, place_id: str) -> str | None:
        return next((o.id for o in self.objects.values() if o.where == place_id), None)

    def in_place(self, oid: str, place_id: str) -> bool:
        o = self.objects.get(oid)
        if o is None:
            return False
        p = self.places.get(place_id)
        if p is not None and p.kind == "group":
            return o.where in p.members
        return o.where == place_id

    def top_of(self, oid: str) -> str | None:
        return next((o.id for o in self.objects.values() if o.where == f"on:{oid}"), None)


def dxy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    vx, vy = bx - ax, by - ay
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / L))
    return math.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy))


# ---------------------------------------------------------------- events

EVENT_KINDS = ("plan_arrived", "step_done", "step_failed", "scene_change", "user_text", "paused_idle", "plan_done_unmet")


@dataclass
class Event:
    id: int
    t: int
    kind: str                 # one of EVENT_KINDS
    text: str                 # literal description, as Jev reads it
    object: str | None = None  # the object the event is about, if any
    step_id: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def routine(self) -> bool:
        """A step that finished normally: the one event kind even the always-LLM control keeps local."""
        return self.kind == "step_done"


# ---------------------------------------------------------------- motion


@dataclass
class Command:
    kind: str                           # move | grip | release | wait
    target: tuple | None = None         # (x, y, z) for move
    speed: float | None = None          # cm per tick (xy); None = the skill's normal speed
    fingertip: bool = False             # a one-fingertip push: smaller footprint, drags `drag`
    drag: str | None = None             # object pushed along by a fingertip move


# ---------------------------------------------------------------- decisions

RIGHT_NOW = ("carry_on", "hold", "pause", "back_off", "re_target", "resume")
FIXES = ("none", "retry", "skip", "re_queue", "another_step_first")
ROUTES = ("stay_local", "fast_llm", "capable_llm", "ask_user")
LLM_ROUTES = ("fast_llm", "capable_llm")


@dataclass
class Combined:
    """What code makes of one decision (or of a control's reaction)."""
    right_now: str
    fix: str
    route: str
    reason: str = ""
    react: bool = False                     # always-LLM control: the LLM also picks the reaction
    right_now_raw: str | None = None        # before the cautious fallback
    route_raw: str | None = None            # before confidence escalation
    fix_raw: str | None = None
    conf: dict = field(default_factory=dict)  # group -> confidence used
    problems: list = field(default_factory=list)  # plan-check / progress findings passed to the LLM
    done: dict = field(default_factory=dict)      # done-condition id -> Jev's progress answer (unmeasurable ones)
