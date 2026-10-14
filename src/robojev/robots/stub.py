"""A RobotServer with no physics: H1's test double and M1's second robot (docs/harness-spikes.md).

A kinematic toy behind the real protocol (protocol.py), so the brain can be exercised on a real
clock without MuJoCo or a model. A clock thread advances skills on wall time:

- Grasping skills (pick_and_place, stack_on, hand_over, insert) reach for the object for the first
  half of their duration. At the midpoint they grasp if the object is still within `GRASP_TOLERANCE`
  of where the arm aimed (where it was at start, or at the last retarget()), else they fail with
  "grasp_failed". For the second half the object rides in the gripper; on completion it is set down
  at the destination. A one-object place that filled up meanwhile fails the skill with "blocked".
  insert's destination is a holder; the seated object's yaw becomes the holder's opening yaw. An
  occupied holder is refused at precondition() (the brain can see that before it starts).
- push reaches the object, then on completion the object teleports `distance` m in the direction
  asked and lands in whichever slot or bin its centre is now inside (else on the table).
- reorient reaches the object (first half), then turns it where it lies (second half): on completion
  its yaw is the direction asked; it never leaves its place or enters the gripper.
- survey and hold just take their time.
- ArmObs.skill names the skill on each arm (None when idle), as the protocol allows.

A scripted person can move an object, put a hand near either arm's gripper, or trip either arm's
safety filter, at a time or when a condition on the world holds, so scene_change and safety_trip
events occur.

Two arms (`widowx_pair`, `pair_scene`): bases facing each other across the table, positions in one
"table" frame shared by both (+x from arm_0's base toward arm_1's), each arm with its own workspace
box and home. The strip both reach is the shared zone, a place of kind "zone" whose members are two
hand-over slots. The **shared-zone lock** is the server's, as on the real pair: only one arm may be
inside the zone at a time. The stub models time, not space, so "inside" is a flag a skill holds for
the segment that goes into the zone: the reach when its object starts in the zone, the carry (or
push) when its destination is in it. A skill whose next segment needs the zone while the other arm
holds it waits where it is, making no progress, its mode still "running", with the phase text
"waiting outside the shared zone for <arm> to leave it ..." (phase "wait_zone"). The flag is kept
while the skill is held, paused or tripped (the arm is still in there) and given up when the segment
ends or the skill ends, fails, is cancelled or STOP is pressed. `zone_log` records every enter,
leave and wait.

Protocol rules, and how this server keeps them:
  - Nothing blocks: every call takes the lock briefly; motion happens on the clock thread.
  - Control calls are idempotent: hold() on a held arm, resume() on a running one, pause() on an
    idle one, stop() twice are no-ops. hold() leaves a paused arm paused (pause is the stronger
    state: it waits for resume() because a person is near).
  - Every event about a skill carries its skill_id. Events are built under the lock and delivered
    after it is released, on the thread that caused them, so a subscriber may call back in.
  - precondition() and start() refuse bad arguments (checked against the manifest's JSON schema)
    and impossible starts (STOP included) with the same literal reason; start() refuses by raising
    ValueError(reason) and creates no skill, so no event is ever sent for a refusal. A skill that
    started and then failed says "<code>: <text>" (codes from catalog.REASONS) in status().reason
    and in a skill_failed event, and leaves the arm's mode "failed" until the next start. The two
    are never confused.
  - start() on a busy arm cancels its skill: state "cancelled", reason "cancelled: replaced by
    <id>", no event (the caller asked for it). STOP ends every skill "failed" with "stopped: ..."
    and a skill_failed event: no skill survives, and a brain other than the one that pressed STOP
    must hear of it.
  - status() carries `phase`, a machine name (reach, carry, push, survey, hold, done, failed,
    cancelled), beside the literal `phase_text`.
  - Missed heartbeats for `heartbeat_timeout_s` (once the first has arrived) hold every running arm
    and send heartbeat_lost; resume() is the brain's to send.
  - STOP parks every arm at its own home.

Choices the protocol leaves open, made here (reported, not settled):
  - precondition() checks the arguments and the world, not whether the arm is busy: the brain asks
    it about steps other than the running one.
  - status() of an unknown id raises KeyError.
  - A grasping skill on an object in another arm's gripper is refused at precondition() and start().
  - A free spot on the table is near the table place's centre when the arm reaches it, else in front
    of the arm's own home (each arm puts things down in its own half).

Units: metres and seconds. Base frame: +x forward, +y left, +z up (as on the real arm).
"""
from __future__ import annotations

import copy
import itertools
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from robojev import catalog
from robojev.protocol import (ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, ObjectObs, PlaceObs, RobotEvent, SkillStatus,
                       WorldState)

# ------------------------------------------------------------------ manifests and a scene

WORKSPACE = ((0.10, 0.50), (-0.30, 0.30), (-0.02, 0.35))
HOME = (0.20, 0.0)
TRAVEL_Z = 0.12
GRASPING = ("pick_and_place", "stack_on", "hand_over", "insert")     # the object rides in the gripper to a destination
IN_PLACE = ("push", "reorient")                                        # the object is handled where it lies
DIRECTION_XY = {"left": (0.0, 1.0), "right": (0.0, -1.0), "toward_robot": (-1.0, 0.0), "away_from_robot": (1.0, 0.0)}
DIRECTION_YAW = {"away_from_robot": 0.0, "left": math.pi / 2, "toward_robot": math.pi, "right": -math.pi / 2}   # rad about +z
GRASP_TOLERANCE = 0.02      # m: an object moved further than this since the arm aimed is missed
DURATIONS = {"pick_and_place": 1.0, "stack_on": 1.0, "hand_over": 1.5, "insert": 1.2, "reorient": 0.8, "push": 0.6, "survey": 0.5}


def widowx_like(robot: str = "widowx-stub") -> Manifest:
    """One WidowX-AI-like follower offering every standard skill."""
    arm = ArmSpec("arm_0", "base", WORKSPACE, GripperSpec(max_width=0.044, can_grasp=True), TRAVEL_Z,
                  "one WidowX AI-like follower arm")
    return Manifest(robot, (arm,), catalog.STANDARD, catalog=catalog.CATALOG_VERSION)


def push_only(robot: str = "pusher-stub") -> Manifest:
    """One arm whose end effector cannot grasp: it touches and pushes (M1's second manifest)."""
    arm = ArmSpec("arm_0", "base", WORKSPACE, GripperSpec(max_width=0.0, can_grasp=False), TRAVEL_Z,
                  "one arm with a fixed fingertip: it can touch and push, not grasp")
    return Manifest(robot, (arm,), (catalog.PUSH, catalog.SURVEY, catalog.HOLD), catalog=catalog.CATALOG_VERSION)


PAIR_WORKSPACES = {"arm_0": ((0.10, 0.40), (-0.30, 0.30), (-0.02, 0.35)),
                   "arm_1": ((0.30, 0.60), (-0.30, 0.30), (-0.02, 0.35))}
PAIR_HOMES = {"arm_0": (0.20, 0.0), "arm_1": (0.50, 0.0)}
ZONE_X = 0.35               # the shared zone's centre line: both workspaces cover x 0.30 to 0.40


def widowx_pair(robot: str = "widowx-pair-stub") -> Manifest:
    """Two WidowX-AI-like followers facing each other, both offering every standard skill. Workspaces
    are in the shared "table" frame: arm_0 reaches x 0.10 to 0.40 m, arm_1 x 0.30 to 0.60 m, so the
    strip x 0.30 to 0.40 m (the shared zone) is the only part both reach."""
    arms = tuple(ArmSpec(a, "table", ws, GripperSpec(max_width=0.044, can_grasp=True), TRAVEL_Z,
                         f"a WidowX AI-like follower, base at x {0.0 if a == 'arm_0' else 0.70:.2f} m")
                 for a, ws in PAIR_WORKSPACES.items())
    return Manifest(robot, arms, catalog.STANDARD, catalog=catalog.CATALOG_VERSION)


def pair_scene(numbers=(5, 1, 3)) -> tuple[list[ObjectObs], list[PlaceObs]]:
    """The two-arm table: numbered 4 cm blocks in a row in arm_0's half (x 0.18), the shared zone and
    its two hand-over slots on the centre line (x 0.35), and a tray of one slot per block in arm_1's
    half (x 0.54). Slot 1 is the leftmost (+y) as arm_0 sees the table; the tray and the zone are
    groups of their slots; the table and the person are places too."""
    n = len(numbers)
    slots = [PlaceObs(f"tray_slot_{i}", f"tray slot {i}" + (" (the leftmost slot)" if i == 1 else
                                                          " (the rightmost slot)" if i == n else ""),
                      "slot", 0.54, 0.12 - 0.24 * (i - 1) / (n - 1) if n > 1 else 0.0, 1)
             for i in range(1, n + 1)]
    zone = [PlaceObs(f"zone_slot_{i}", f"shared zone slot {i}", "slot", ZONE_X, y, 1)
            for i, y in ((1, 0.05), (2, -0.05))]
    places = slots + zone + [
        PlaceObs("tray", "the tray (any slot)", "group", 0.54, 0.0, 0, tuple(p.id for p in slots)),
        PlaceObs("shared_zone", "the shared zone (both arms reach it; one arm inside at a time)", "zone", ZONE_X, 0.0, 0,
                 tuple(p.id for p in zone), radius=0.08),
        PlaceObs("table", "the table", "table", 0.20, -0.22, 0, radius=0.30),
        PlaceObs("person", "the person", "person", 0.35, -0.45, 0)]
    objs = [ObjectObs(f"block_{k}", f"block {k}", 0.18, 0.12 - 0.12 * i, 0.0, "table", label=str(k), size=0.04)
            for i, k in enumerate(numbers)]
    return objs, places


def sort_scene(numbers=(1, 3, 5), n_slots: int | None = None) -> tuple[list[ObjectObs], list[PlaceObs]]:
    """Numbered 4 cm blocks in a row on the table in front of a row of tray slots. Slot 1 is the
    leftmost (+y); a "tray" group, the table and the person are places too."""
    n = n_slots or len(numbers)
    slots = [PlaceObs(f"tray_slot_{i}", f"tray slot {i}" + (" (the leftmost slot)" if i == 1 else
                                                          " (the rightmost slot)" if i == n else ""),
                      "slot", 0.36, 0.12 - 0.06 * (i - 1), 1) for i in range(1, n + 1)]
    places = slots + [PlaceObs("tray", "the tray (any slot)", "group", 0.36, 0.0, 0, tuple(p.id for p in slots)),
                      PlaceObs("table", "the table", "table", 0.22, -0.22, 0, radius=0.30),
                      PlaceObs("person", "the person", "person", 0.55, 0.0, 0)]
    objs = [ObjectObs(f"block_{k}", f"block {k}", 0.20, 0.15 - 0.10 * i, 0.0, "table", label=str(k), size=0.04)
            for i, k in enumerate(numbers)]
    return objs, places


def tools_scene() -> tuple[list[ObjectObs], list[PlaceObs]]:
    """Three surgical tools (rods: a scalpel, forceps, scissors) lying in a row on the table, their
    handles pointing different ways, in front of a rack of three holders whose openings face the
    robot. A "rack" group, the table and the person are places too. Every object carries `yaw` and
    `shape`, every holder `yaw`, so insert and reorient have something to work on."""
    holders = [PlaceObs(f"holder_{i}", f"rack holder {i}" + (" (the leftmost)" if i == 1 else " (the rightmost)" if i == 3 else ""),
                        "holder", 0.36, 0.10 - 0.10 * (i - 1), 1, yaw=math.pi) for i in (1, 2, 3)]
    places = holders + [PlaceObs("rack", "the rack (any holder)", "group", 0.36, 0.0, 0, tuple(h.id for h in holders)),
                        PlaceObs("table", "the table", "table", 0.22, -0.22, 0, radius=0.30),
                        PlaceObs("person", "the person", "person", 0.55, 0.0, 0)]
    tools = (("scalpel", 0.0), ("forceps", math.pi / 2), ("scissors", -math.pi / 2))
    objs = [ObjectObs(name, f"the {name}", 0.20, 0.12 - 0.12 * i, 0.0, "table", label=name, size=0.12, yaw=yaw, shape="rod")
            for i, (name, yaw) in enumerate(tools)]
    return objs, places


# ------------------------------------------------------------------ internal records


@dataclass
class _Run:
    id: str
    arm: str
    name: str
    args: dict
    duration: float
    start_xy: tuple[float, float]
    aim: tuple[float, float] | None     # where the object was at start or the last retarget()
    elapsed: float = 0.0
    state: str = "running"              # running | holding | paused | done | failed | cancelled
    grasped: bool = False               # object in the gripper (or, for push, under the fingertip)
    reason: str | None = None
    zone_reach: bool = False            # the reach goes into the shared zone (the object starts there)
    zone_carry: bool = False            # the carry or push goes into the shared zone (its destination is there)
    zone_wait: str | None = None        # the arm inside the zone this skill is waiting for, while it waits

    @property
    def active(self) -> bool:
        return self.state in ("running", "holding", "paused")


@dataclass
class _Beat:
    """One scripted action of the person: at `when` s after open(), or once when(world) first holds."""
    when: float | Callable[[WorldState], bool]
    what: str
    act: Callable[[], list[RobotEvent]]
    fired: bool = False


# ------------------------------------------------------------------ the server


class StubRobotServer:
    """Implements protocol.RobotServer. Not protocol: open()/close(), the scripted person
    (person_moves, person_hand, trip, at), and the `calls`, `history`, `script_log` records tests read."""

    def __init__(self, manifest: Manifest, objects: list[ObjectObs], places: list[PlaceObs], *,
                 durations: dict[str, float] | None = None, tick_s: float = 0.02, heartbeat_timeout_s: float = 1.0,
                 homes: dict[str, tuple[float, float]] | None = None):
        """`homes`: arm id -> (x, y) where it starts and parks; an arm not named starts at HOME."""
        self._manifest = manifest
        self._lock = threading.RLock()
        self._objs = {o.id: copy.deepcopy(o) for o in objects}
        self._places = {p.id: p for p in places}
        self._homes = {a.id: (homes or {}).get(a.id, HOME) for a in manifest.arms}
        self._arms = {a.id: ArmObs(a.id, *self._homes[a.id], a.travel_z, None, a.gripper.max_width if a.gripper else 0.0, "idle")
                      for a in manifest.arms}
        self._specs = {a.id: a for a in manifest.arms}
        self._hands: list[HandObs] = []
        self._runs: dict[str, _Run] = {}
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._beats: list[_Beat] = []
        self._ids = itertools.count(1)
        self._dur = {**DURATIONS, **(durations or {})}
        self.tick_s, self.heartbeat_timeout_s = tick_s, heartbeat_timeout_s
        self._t0 = time.monotonic()
        self._last_hb: float | None = None
        self._hb_lost = False
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self.calls: list[tuple[float, str, tuple]] = []      # (t, tool, args): every tool call, refused or not
        self.history: list[tuple[float, str, str]] = []      # (t, arm, mode) on every mode change
        self.script_log: list[tuple[float, str]] = []        # (t, what) when a scripted beat fires
        self.heartbeats = 0
        self._zone_owner: str | None = None                  # id of the skill inside the shared zone
        self.zone_log: list[tuple[float, str, str]] = []     # (t, arm, "enter" | "leave" | "wait")

    # ---------------------------------------------------------------- lifecycle (not protocol)
    def open(self) -> StubRobotServer:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._clock, name="stub-robot", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> StubRobotServer:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def now(self) -> float:
        return time.monotonic() - self._t0

    # ---------------------------------------------------------------- the scripted person (not protocol)
    def at(self, when: float | Callable[[WorldState], bool], what: str, act: Callable[[], list[RobotEvent]]) -> None:
        """Run `act` under the lock at `when`; it returns the events to send. A callable `when` gets a
        live, read-only view of the world each tick."""
        with self._lock:
            self._beats.append(_Beat(when, what, act))

    def person_moves(self, when, oid: str, to: str | None = None, xy: tuple[float, float] | None = None) -> None:
        """The person moves an object into a place, or to (x, y) on the table."""
        def act() -> list[RobotEvent]:
            o = self._objs[oid]
            if to is not None:
                p = self._places[to]
                o.x, o.y, o.where = p.x, p.y, to
            else:
                o.x, o.y, o.where = xy[0], xy[1], "table"
            return [self._ev("scene_change", f"a person moved {o.name}; it is now {self._where_text(o.where)}", object=oid)]
        self.at(when, f"person moves {oid}", act)

    def person_hand(self, when, for_s: float = 1.0, dist: float = 0.15, speed: float = 0.1, arm: str = "arm_0",
                    xy: tuple[float, float] | None = None) -> None:
        """A hand appears `dist` m from `arm`'s gripper on the person's side (or at `xy`, at the
        gripper's height), reaching toward that gripper at `speed` m/s (reported, not moved: the toy
        hand holds its pose), and leaves `for_s` s later."""
        def leave() -> list[RobotEvent]:
            self._hands = []
            return [self._ev("scene_change", "the person's hand has gone out of view")]

        def act() -> list[RobotEvent]:
            a = self._arms[arm]
            if xy is None:
                ang = math.atan2(-a.y, 0.6 - a.x)
                hx, hy = a.x + dist * math.cos(ang), a.y + dist * math.sin(ang)
            else:
                hx, hy = xy
                ang = math.atan2(hy - a.y, hx - a.x)
            ux, uy = math.cos(ang), math.sin(ang)
            self._hands = [HandObs(hx, hy, a.z, -speed * ux, -speed * uy)]
            self._beats.append(_Beat(self.now() + for_s, "person's hand leaves", leave))
            d = math.hypot(hx - a.x, hy - a.y)
            return [self._ev("scene_change", f"a person's hand appeared {d:.2f} m from {arm}'s gripper"
                             if len(self._arms) > 1 else f"a person's hand appeared {dist:.2f} m from the gripper")]
        self.at(when, "person's hand appears", act)

    def move(self, oid: str, x: float, y: float, yaw: float | None = None) -> None:
        """Not protocol: a scripted person moves an object to (x, y) now, into whichever slot or bin
        holds that point, else onto the table. Nothing is announced (unlike person_moves): whoever
        plays the person reports it, as perception would. Same shape as the sim server's move()."""
        with self._lock:
            o = self._objs[oid]
            o.x, o.y = x, y
            o.where = self._landed(o)

    def trip(self, when, text: str = "effort limit exceeded", arm: str = "arm_0") -> None:
        """The safety filter trips on `arm`: its skill holds where it is and the arm reports "tripped"."""
        def act() -> list[RobotEvent]:
            run = self._active(arm)
            if run is not None and run.state == "running":
                run.state = "holding"
            self._mode(arm, "tripped")
            return [self._ev("safety_trip", f"the safety filter tripped: {text}", arm=arm,
                             skill_id=run.id if run else None)]
        self.at(when, f"safety trip on {arm}", act)

    # ---------------------------------------------------------------- protocol: resources
    def manifest(self) -> Manifest:
        return self._manifest

    def world(self) -> WorldState:
        with self._lock:
            return WorldState(self.now(), copy.deepcopy(self._objs), copy.deepcopy(self._places),
                              copy.deepcopy(self._hands), copy.deepcopy(self._arms))

    def status(self, skill_id: str) -> SkillStatus:
        with self._lock:
            r = self._runs.get(skill_id)
            if r is None:
                raise KeyError(f"no skill {skill_id}")
            return SkillStatus(id=r.id, arm=r.arm, name=r.name, state=r.state, phase_text=self._phase(r),
                               phase=self._phase_name(r), reason=r.reason, heading=self._heading(r), args=dict(r.args))

    # ---------------------------------------------------------------- protocol: tools
    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        with self._lock:
            return self._why_not(arm, skill, args)

    def start(self, arm: str, skill: str, args: dict) -> str:
        with self._lock:
            self.calls.append((self.now(), "start", (arm, skill, dict(args))))
            why = self._why_not(arm, skill, args)
            if why is not None:
                raise ValueError(why)
            sid = f"sk{next(self._ids)}"
            old = self._active(arm)
            if old is not None:
                old.state, old.reason = "cancelled", f"cancelled: replaced by {sid}"
                self._leave_zone(old)
            a, o = self._arms[arm], self._objs.get(args.get("object", ""))
            dur = float(args.get("seconds", 1.0)) if skill == "hold" else self._dur.get(skill, 1.0)
            r = _Run(sid, arm, skill, dict(args), dur, (a.x, a.y), (o.x, o.y) if o else None)
            if skill in GRASPING and a.holding == args.get("object"):
                r.elapsed, r.grasped = r.duration / 2, True      # already in the gripper: carry from here
            if r.aim is not None:
                r.zone_reach = not r.grasped and self._in_zone(*r.aim)
                r.zone_carry = self._in_zone(*self._dest(r))
            self._runs[sid] = r
            a.skill = sid
            self._mode(arm, "running")
        return sid

    def hold(self, arm: str) -> None:
        self._control("hold", arm, frm=("running",), to="holding")

    def pause(self, arm: str) -> None:
        self._control("pause", arm, frm=("running", "holding"), to="paused")

    def resume(self, arm: str) -> None:
        self._control("resume", arm, frm=("holding", "paused"), to="running")

    def retarget(self, skill_id: str) -> None:
        with self._lock:
            self.calls.append((self.now(), "retarget", (skill_id,)))
            r = self._runs.get(skill_id)
            o = self._objs.get(r.args.get("object", "")) if r is not None else None
            if r is not None and r.active and o is not None and not r.grasped:
                r.aim = (o.x, o.y)

    def stop(self) -> None:
        evs = []
        with self._lock:
            self.calls.append((self.now(), "stop", ()))
            for r in [r for r in self._runs.values() if r.active]:
                evs.append(self._fail(r, "stopped: STOP was pressed; every arm parked", idle=False))
            for a in self._arms.values():
                if a.holding:                     # parking sets the held object down under the gripper
                    o = self._objs[a.holding]
                    o.where, o.x, o.y, a.holding = "table", a.x, a.y, None
                a.x, a.y, a.z = *self._homes[a.id], self._specs[a.id].travel_z
                a.skill = None
                self._mode(a.id, "stopped")
        self._emit(evs)

    def heartbeat(self) -> None:
        with self._lock:
            self.heartbeats += 1
            self._last_hb, self._hb_lost = self.now(), False

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            if callback in self._subs:
                self._subs.remove(callback)

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    # ---------------------------------------------------------------- control internals
    def _control(self, tool: str, arm: str, frm: tuple[str, ...], to: str) -> None:
        """hold / pause / resume: move the arm's skill from a state in `frm` to `to`; anything else is a no-op."""
        with self._lock:
            self.calls.append((self.now(), tool, (arm,)))
            a = self._arms.get(arm)
            if a is None or a.mode == "stopped":
                return
            r = self._active(arm)
            if r is None or r.state not in frm:
                return                            # idle, already there, or a state this tool leaves alone
            r.state = to
            self._mode(arm, to)

    def _active(self, arm: str) -> _Run | None:
        return next((r for r in self._runs.values() if r.arm == arm and r.active), None)

    def _mode(self, arm: str, mode: str) -> None:
        a = self._arms[arm]
        if a.mode != mode:
            a.mode = mode
            self.history.append((self.now(), arm, mode))

    def _ev(self, kind: str, text: str, **kw) -> RobotEvent:
        return RobotEvent(self.now(), kind, text, **kw)

    def _emit(self, evs: list[RobotEvent]) -> None:
        """Deliver events. Always called with the lock released."""
        for ev in evs:
            for cb in list(self._subs):
                cb(ev)

    # ---------------------------------------------------------------- the clock
    def _clock(self) -> None:
        last = time.monotonic()
        while not self._halt.wait(self.tick_s):
            now = time.monotonic()
            dt, last = now - last, now
            with self._lock:
                evs = self._tick(dt)
            self._emit(evs)

    def _tick(self, dt: float) -> list[RobotEvent]:
        evs: list[RobotEvent] = []
        t = self.now()
        if self._last_hb is not None and not self._hb_lost and t - self._last_hb > self.heartbeat_timeout_s:
            self._hb_lost = True
            for r in self._runs.values():
                if r.state == "running":
                    r.state = "holding"
                    self._mode(r.arm, "holding")
            evs.append(self._ev("heartbeat_lost", f"no heartbeat for {self.heartbeat_timeout_s:.1f} s: every arm is holding"))
        live = WorldState(t, self._objs, self._places, self._hands, self._arms)
        for b in list(self._beats):
            if not b.fired and (b.when(live) if callable(b.when) else t >= b.when):
                b.fired = True
                self.script_log.append((t, b.what))
                evs += b.act()
        for r in list(self._runs.values()):
            if r.state == "running":
                evs += self._advance(r, dt)
        return evs

    def _advance(self, r: _Run, dt: float) -> list[RobotEvent]:
        if not self._zone_free(r):
            return []                             # waiting outside the shared zone: no progress
        r.elapsed += dt
        f = min(1.0, r.elapsed / r.duration)
        if r.name in ("survey", "hold"):
            return [self._finish(r)] if f >= 1.0 else []
        a = self._arms[r.arm]
        o = self._objs.get(r.args.get("object", ""))
        if not r.grasped:
            k = min(1.0, f / 0.5)
            a.x = r.start_xy[0] + (r.aim[0] - r.start_xy[0]) * k
            a.y = r.start_xy[1] + (r.aim[1] - r.start_xy[1]) * k
            if f < 0.5:
                return []
            if o is None or o.where.startswith("gripper:") or o.where == "person" or \
                    math.hypot(o.x - r.aim[0], o.y - r.aim[1]) > GRASP_TOLERANCE:
                name = o.name if o is not None else r.args.get("object")
                return [self._fail(r, f"grasp_failed: {name} is no longer where the arm reached for it")]
            r.grasped = True
            if r.name not in IN_PLACE:
                o.where, a.holding = f"gripper:{r.arm}", o.id
            if not self._zone_free(r):            # the carry goes into the zone and the other arm is in it
                r.elapsed = r.duration / 2
                a.x, a.y = r.aim
                if r.name not in IN_PLACE:
                    o.x, o.y = a.x, a.y
                return []
        dest = self._dest(r)
        k = max(0.0, (f - 0.5) / 0.5)
        a.x, a.y = r.aim[0] + (dest[0] - r.aim[0]) * k, r.aim[1] + (dest[1] - r.aim[1]) * k
        if r.name not in IN_PLACE:
            o.x, o.y = a.x, a.y
        return [self._finish(r)] if f >= 1.0 else []

    def _dest(self, r: _Run) -> tuple[float, float]:
        if r.name == "push":
            dx, dy = DIRECTION_XY[r.args["direction"]]
            return (r.aim[0] + r.args["distance"] * dx, r.aim[1] + r.args["distance"] * dy)
        if r.name == "reorient":
            return r.aim                          # turned where it lies
        if r.name == "hand_over":
            p = self._places.get("person")
            return (p.x, p.y) if p is not None else (0.5, 0.0)
        if r.name == "stack_on":
            b = self._objs[r.args["onto"]]
            return (b.x, b.y)
        p = self._places[r.args["place"]]
        if p.kind == "table":                     # a free spot: near the table place's centre, in this arm's reach
            cx = p.x if _inside(self._specs[r.arm].workspace, p.x, p.y) else self._homes[r.arm][0]
            return self._free_spot(r.arm, cx, p.y, but=r.args.get("object"))
        return (p.x, p.y)

    def _free_spot(self, arm: str, cx: float, cy: float, but: str | None, pitch: float = 0.06) -> tuple[float, float]:
        """The grid point around (cx, cy), nearest first, inside the arm's workspace with no other
        object within `pitch`; failing all, the centre."""
        others = [(o.x, o.y) for o in self._objs.values() if o.id != but and not o.where.startswith("gripper:")]
        cands = sorted(((cx + i * pitch, cy + j * pitch) for i in range(-2, 3) for j in range(-4, 5)),
                       key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        for x, y in cands:
            if _inside(self._specs[arm].workspace, x, y) and all(math.hypot(x - ox, y - oy) >= pitch for ox, oy in others):
                return (x, y)
        return (cx, cy)

    def _finish(self, r: _Run) -> RobotEvent:
        oid = r.args.get("object")
        if r.name in GRASPING:
            a, o = self._arms[r.arm], self._objs[oid]
            if r.name in ("pick_and_place", "insert"):
                p = self._places[r.args["place"]]
                occ = next((x.name for x in self._objs.values() if x.where == p.id and x.id != oid), None)
                if p.capacity == 1 and occ is not None:
                    return self._fail(r, f"blocked: {p.name} is occupied by {occ}")
                where = p.id
                if r.name == "insert" and p.yaw is not None:
                    o.yaw = p.yaw                 # seated: the front follows the holder's opening
            elif r.name == "stack_on":
                base = self._objs[r.args["onto"]]
                where, o.z = f"on:{base.id}", base.z + (base.size or 0.04)
            else:
                where = "person"
            o.x, o.y = self._dest(r)
            o.where, a.holding = where, None
        elif r.name == "push":
            o = self._objs[oid]
            o.x, o.y = self._dest(r)
            o.where = self._landed(o)
        elif r.name == "reorient":
            self._objs[oid].yaw = DIRECTION_YAW[r.args["direction"]]
        r.state = "done"
        self._leave_zone(r)
        self._arms[r.arm].skill = None
        self._mode(r.arm, "idle")
        return self._ev("skill_done", f"{r.name} finished: {self._phase(r)}", skill_id=r.id, arm=r.arm, object=oid)

    def _fail(self, r: _Run, why: str, idle: bool = True) -> RobotEvent:
        """End a skill that started: "<code>: <text>" in status().reason and in the event; the arm
        reads "failed" until its next start (`idle=False`: the caller sets the arm's mode)."""
        r.state, r.reason = "failed", why
        self._leave_zone(r)
        self._arms[r.arm].skill = None
        if idle:
            self._mode(r.arm, "failed")
        return self._ev("skill_failed", why, skill_id=r.id, arm=r.arm, object=r.args.get("object"), data={"reason": why})

    def _landed(self, o: ObjectObs) -> str:
        """Where a pushed object ends up: the slot or bin whose radius holds its centre, else the table."""
        for p in self._places.values():
            if p.kind in ("slot", "bin") and math.hypot(o.x - p.x, o.y - p.y) <= p.radius:
                return p.id
        return "table"

    # ---------------------------------------------------------------- the shared-zone lock
    def _in_zone(self, x: float, y: float) -> bool:
        """(x, y) is inside a shared zone (a place of kind "zone", by its radius)."""
        return any(p.kind == "zone" and math.hypot(x - p.x, y - p.y) <= p.radius for p in self._places.values())

    def _zone_free(self, r: _Run) -> bool:
        """Whether `r` may make progress: its current segment stays out of the zone (any flag it held is
        given up), or goes in and the zone is free or already its own (taken now). Else it waits."""
        needs = r.zone_carry if r.grasped else r.zone_reach
        if not needs:
            self._leave_zone(r)
            return True
        if self._zone_owner in (None, r.id):
            if self._zone_owner is None:
                self._zone_owner = r.id
                self.zone_log.append((self.now(), r.arm, "enter"))
            r.zone_wait = None
            return True
        inside = self._runs[self._zone_owner].arm
        if r.zone_wait != inside:
            self.zone_log.append((self.now(), r.arm, "wait"))
        r.zone_wait = inside
        return False

    def _leave_zone(self, r: _Run) -> None:
        r.zone_wait = None
        if self._zone_owner == r.id:
            self._zone_owner = None
            self.zone_log.append((self.now(), r.arm, "leave"))

    # ---------------------------------------------------------------- preconditions
    def _why_not(self, arm: str, skill: str, args: dict) -> str | None:
        """None if `skill` could start on `arm` now, else "<code>: <literal text>"."""
        spec = self._manifest.skill(skill)
        if arm not in self._arms or spec is None or (spec.arms and arm not in spec.arms):
            return f"precondition: {arm} has no skill {skill}"
        if self._arms[arm].mode == "stopped":
            return "precondition: the arm is stopped (STOP was pressed)"
        bad = _arg_problem(spec.args_schema, args)
        if bad is not None:
            return f"precondition: {skill} {bad}"
        if "object" not in args:
            return None
        a, o = self._arms[arm], self._objs.get(args["object"])
        if o is None:
            return f"not_in_view: {args['object']} is not in view"
        if o.where == "person":
            return f"precondition: {o.name} is held by the person"
        if skill in GRASPING and o.where.startswith("gripper:") and o.where != f"gripper:{arm}":
            return f"precondition: {o.name} is in {o.where[8:]}'s gripper"
        if not _inside(self._specs[arm].workspace, o.x, o.y):
            return f"unreachable: {o.name} is outside {arm}'s reach"
        top = next((x.name for x in self._objs.values() if x.where == f"on:{o.id}"), None)
        if skill in GRASPING or skill == "reorient":           # reorient regrasps the object where it lies
            g = self._specs[arm].gripper
            if g is None or not g.can_grasp:
                return f"precondition: {arm} cannot grasp"
            if a.holding and a.holding != o.id:
                return f"precondition: the gripper is holding {self._objs[a.holding].name}"
            if top is not None:
                return f"blocked: {o.name} has {top} on top of it"
        if skill == "pick_and_place":
            p = self._places.get(args["place"])
            if p is None or p.kind not in ("slot", "bin", "table"):
                return f"precondition: {args['place']} is not a place to put things"
            if p.kind != "table" and not _inside(self._specs[arm].workspace, p.x, p.y):
                return f"unreachable: {p.name} is outside {arm}'s reach"
        if skill == "insert":
            p = self._places.get(args["place"])
            if p is None or p.kind != "holder":
                return f"precondition: {args['place']} is not a holder"
            if o.shape is not None and o.shape != "rod":
                return f"precondition: {o.name} is a {o.shape}; a holder takes a rod"
            if not _inside(self._specs[arm].workspace, p.x, p.y):
                return f"unreachable: {p.name} is outside {arm}'s reach"
            occ = next((x.name for x in self._objs.values() if x.where == p.id and x.id != o.id), None)
            if occ is not None:
                return f"precondition: {p.name} is occupied by {occ}"
        if skill == "stack_on" and (args["onto"] not in self._objs or args["onto"] == o.id):
            return f"precondition: {args['onto']} is not another object in view"
        if skill in IN_PLACE and o.where.startswith("gripper:"):
            return f"precondition: {o.name} is in the gripper"
        return None

    # ---------------------------------------------------------------- literal text
    def _where_text(self, where: str) -> str:
        if where in self._places:
            return f"in {self._places[where].name}"
        if where.startswith("on:"):
            return f"on top of {self._objs[where[3:]].name}"
        if where.startswith("gripper:"):
            return f"in {where[8:]}'s gripper"
        return {"table": "on the table", "person": "held by the person"}.get(where, where)

    def _phase(self, r: _Run) -> str:
        """Where the skill is, literally, as the decision loop reads it."""
        if r.name == "survey":
            return "lifted, looking over the table"
        if r.name == "hold":
            return "holding still"
        o = self._objs.get(r.args.get("object", ""))
        on = o.name if o is not None else r.args.get("object", "")
        dest = (self._places[r.args["place"]].name if r.name in ("pick_and_place", "insert") else
                self._objs[r.args["onto"]].name if r.name == "stack_on" else
                "the person" if r.name == "hand_over" else r.args.get("direction", "").replace("_", " "))
        if r.state == "done":
            return (f"{on} pushed {dest}" if r.name == "push" else f"{on} turned to face {dest}" if r.name == "reorient" else
                    f"{on} seated in {dest}" if r.name == "insert" else f"{on} put down at {dest}")
        if r.zone_wait is not None and r.active:
            what = (f"holding {on}" if r.grasped and r.name not in IN_PLACE else
                    f"next to {on}" if r.grasped else f"not yet at {on}")
            return f"waiting outside the shared zone for {r.zone_wait} to leave it, {what}"
        if r.name == "push":
            return f"pushing {on} {dest}" if r.grasped else f"moving to {on} to push it {dest}, not yet touching it"
        if r.name == "reorient":
            return f"turning {on} to face {dest}" if r.grasped else f"moving to {on} to turn it {dest}, not yet grasped"
        if r.grasped:
            return f"carrying {on} to {dest}, not yet {'seated' if r.name == 'insert' else 'released'}"
        return f"moving to {on} to pick it up, not yet grasped"

    @staticmethod
    def _phase_name(r: _Run) -> str:
        if r.state in ("done", "failed", "cancelled"):
            return r.state
        if r.name in ("survey", "hold"):
            return r.name
        if r.zone_wait is not None:
            return "wait_zone"
        if not r.grasped:
            return "reach"
        return {"push": "push", "reorient": "turn"}.get(r.name, "carry")

    def _heading(self, r: _Run) -> tuple[float, float] | None:
        if not r.active or r.aim is None:
            return None
        return self._dest(r) if r.grasped else r.aim


# ------------------------------------------------------------------ helpers


def _inside(ws: tuple, x: float, y: float) -> bool:
    (x0, x1), (y0, y1), _ = ws
    return x0 <= x <= x1 and y0 <= y <= y1


_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


def _arg_problem(schema: dict, args: dict) -> str | None:
    """The first way `args` breaks a flat catalog schema (properties with type, enum, minimum,
    maximum; required; additionalProperties false), as literal text, or None."""
    props = schema.get("properties", {})
    if not isinstance(args, dict):
        return "takes an object of arguments"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(args) - set(props))
        if extra:
            return f"takes no argument {', '.join(extra)}"
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        return f"needs {', '.join(missing)}"
    for k, v in args.items():
        p = props.get(k, {})
        want = _TYPES.get(p.get("type", ""))
        if want is not None and (not isinstance(v, want) or (p.get("type") != "boolean" and isinstance(v, bool))):
            return f"{k} must be a {p['type']}, not {v!r}"
        if "enum" in p and v not in p["enum"]:
            return f"{k} must be one of {', '.join(map(str, p['enum']))}, not {v!r}"
        if "minimum" in p and v < p["minimum"] or "maximum" in p and v > p["maximum"]:
            return f"{k} must be between {p.get('minimum')} and {p.get('maximum')}, not {v!r}"
    return None
