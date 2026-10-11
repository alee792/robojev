"""The robot protocol: what a brain can know and do about any robot, with no hardware in sight.

Shaped like MCP so it becomes MCP at K6 (docs/harness-spikes.md): each method below maps 1:1 onto
an MCP tool, resource or notification, noted in its docstring. Until then it is an in-process Python
interface. The brain is the client; a RobotServer per hardware (sim WidowX, real WidowX, Panda, a
stub) is the server. A brain that imports anything hardware-specific has broken the design.

Units: metres and seconds everywhere. Positions are in the robot's declared base frame.

Skill granularity is the protocol boundary: a skill is seconds long and interruptible; the policy
inside it (10-100 Hz) and the motor loop (~1 kHz) never cross this interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

# ------------------------------------------------------------------ manifest (MCP: initialize + tools/list)


@dataclass(frozen=True)
class GripperSpec:
    max_width: float                    # m between the pads fully open
    can_grasp: bool = True              # False for a handle-only end effector (a leader arm)


@dataclass(frozen=True)
class ArmSpec:
    id: str                             # "arm_0", "left", "follower": how plans refer to it
    base_frame: str                     # name of the frame positions are reported in
    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]   # (x0,x1),(y0,y1),(z0,z1)
    gripper: GripperSpec | None
    travel_z: float                     # height the arm travels at between places
    description: str = ""


@dataclass(frozen=True)
class SkillSpec:
    """One capability, as an MCP tool: name, what it does, JSON schema for its arguments.

    `args_schema` uses place and object *ids* from the world state. The planner's output schema is
    generated from the manifest's skill list, so a robot that doesn't advertise a skill is never
    asked to do it."""
    name: str
    description: str
    args_schema: dict
    arms: tuple[str, ...] = ()          # which arms can run it; () = any


@dataclass(frozen=True)
class Manifest:
    robot: str                          # "widowx-sim", "widowx-bay-1", "panda"
    arms: tuple[ArmSpec, ...]
    skills: tuple[SkillSpec, ...]
    units: str = "m,s"
    catalog: str = "0.1"                # catalog version the skill specs conform to; the brain refuses a mismatch

    def skill(self, name: str) -> SkillSpec | None:
        return next((s for s in self.skills if s.name == name), None)


# ------------------------------------------------------------------ resources (MCP: resources/read)


@dataclass
class ObjectObs:
    id: str
    name: str                           # how people refer to it: "block 5", "the red cube"
    x: float
    y: float
    z: float
    where: str                          # place id | "table" | "on:<object id>" | "gripper:<arm id>" | "person"
    label: str | None = None            # a number or letter read off it
    size: float | None = None           # characteristic size, m
    claimed_by: str | None = None       # arm id that has claimed it (two-arm coordination)


@dataclass
class PlaceObs:
    id: str
    name: str                           # "tray slot 3 (the leftmost slot)"
    kind: str                           # slot | bin | table | person | group | zone
    x: float
    y: float
    capacity: int                       # 1 = one object; 0 = any number
    members: tuple[str, ...] = ()       # a group lists its slots
    radius: float = 0.03                # an object whose centre is this close is "in" the place


@dataclass
class HandObs:
    x: float
    y: float
    z: float
    vx: float = 0.0                     # m/s
    vy: float = 0.0
    held_out: bool = False              # palm up and still: offered to take something


@dataclass
class ArmObs:
    id: str
    x: float
    y: float
    z: float
    holding: str | None                 # object id between closed fingers
    gripper_width: float
    mode: str                           # idle | running | holding | paused | stopped | tripped | failed
    skill: str | None = None            # id of the skill running on this arm, so a reconnecting brain can status() it


@dataclass
class WorldState:
    """resource://world: what perception reports now. Never ground truth once off the sim."""
    t: float                            # seconds, monotonic
    objects: dict[str, ObjectObs]
    places: dict[str, PlaceObs]
    hands: list[HandObs]
    arms: dict[str, ArmObs]

    def occupant(self, place_id: str) -> str | None:
        return next((o.id for o in self.objects.values() if o.where == place_id), None)

    def top_of(self, oid: str) -> str | None:
        return next((o.id for o in self.objects.values() if o.where == f"on:{oid}"), None)


@dataclass
class SkillStatus:
    """resource://skills/{id}: where a running skill is, in words the decision loop can use."""
    id: str
    arm: str
    name: str
    state: str                          # running | holding | paused | done | failed | cancelled
    phase_text: str                     # literal: "lowered at tray slot 3 holding block 5, not yet released"
    reason: str | None = None           # why it failed
    heading: tuple[float, float] | None = None   # (x, y) the arm is moving toward, for "in the arm's path"
    args: dict = field(default_factory=dict)
    phase: str = ""                     # machine name of the phase ("descend"), for recorders and conformance tests


# ------------------------------------------------------------------ notifications (MCP: notifications/*)

EVENTS = ("skill_done", "skill_failed", "safety_trip", "scene_change", "heartbeat_lost")


@dataclass
class RobotEvent:
    t: float
    kind: str                           # one of EVENTS
    text: str                           # literal description
    skill_id: str | None = None
    arm: str | None = None
    object: str | None = None
    data: dict = field(default_factory=dict)


# ------------------------------------------------------------------ the server


@runtime_checkable
class RobotServer(Protocol):
    """One robot (any number of arms) behind one interface. Thread-safe; every call returns fast.

    MCP mapping: manifest() = initialize/tools/list; world() and status() = resources; start/hold/
    pause/resume/stop/retarget/heartbeat = tools; subscribe() = notifications.

    Rules, learned from where MCP servers hurt:
    - Nothing blocks. start() returns an id; progress is status(); completion is an event.
    - The server owns the state. A brain that reconnects reads world() and status() and carries
      on; the arm never depends on the client remembering anything.
    - Control calls are idempotent: hold() on a held arm, resume() on a running one, stop() twice
      are all no-ops, never errors.
    - Every event about a skill carries the skill_id start() returned.
    - Bad arguments are refused by precondition() or start() with a literal reason; a skill that
      started and then failed reports "<code>: <text>" in status().reason and a skill_failed event.
      The two are never confused.
    - Manifest text is data. The brain writes its own description for the planner from it; code
      rules never read it."""

    def manifest(self) -> Manifest: ...

    def world(self) -> WorldState:
        """resource://world"""

    def status(self, skill_id: str) -> SkillStatus:
        """resource://skills/{id}. Unknown id: KeyError."""

    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        """tool: None if the skill could start now on this arm, else why not (literal)."""

    def start(self, arm: str, skill: str, args: dict) -> str:
        """tool: begin a skill; returns its id at once. Done/failed arrive as events.

        Refusal (unknown skill or arm, bad or missing args, a precondition that fails, STOP pressed)
        raises ValueError(reason) with the literal reason and emits no event: the MCP tool-error
        path. Starting on a busy arm cancels its running skill (state "cancelled", reason
        "cancelled: replaced by <id>", no event) and begins the new one."""

    def hold(self, arm: str) -> None:
        """tool: stop moving at a safe point, keep the grip and the skill's state; resume() continues.
        On an idle arm it is a no-op (nothing to hold; the mode stays idle)."""

    def pause(self, arm: str) -> None:
        """tool: like hold, but the skill will not continue until resume(); used for a person nearby.
        No-op on an idle arm: not starting a skill while someone is close is the brain's decision."""

    def resume(self, arm: str) -> None:
        """tool: continue after hold or pause, and the only way to clear a tripped arm."""

    def retarget(self, skill_id: str) -> None:
        """tool: aim the running skill at where its object is now."""

    def stop(self) -> None:
        """tool: STOP. Every arm parks; every running skill ends failed with "stopped: ..." and a
        skill_failed event; later start() calls are refused. Terminal for this server: there is no
        reset, a new server is made. Only code calls this."""

    def heartbeat(self) -> None:
        """tool: the brain is alive. A server that misses heartbeats for its timeout holds every arm."""

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        """notifications: the callback runs on the server's thread; return fast."""

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        """Remove a callback; unknown callbacks are ignored."""


# ------------------------------------------------------------------ the recorder (part of the server)


@runtime_checkable
class Recorder(Protocol):
    """Every tick of every skill, for training data and replay. Written by the server, not the brain,
    so it captures what the policy saw and did at the policy's own rate."""

    def frame(self, t: float, arm: str, skill_id: str | None, phase: str, observation: dict, action: dict) -> None: ...

    def event(self, ev: RobotEvent) -> None: ...

    def note(self, **fields) -> None:
        """Anything that isn't a frame or an event: skill_start (skill, args), trial setup, a verdict."""

    def close(self) -> Any: ...
