"""A RobotServer with no physics: the M1 fixture and H1's test double (docs/harness-spikes.md).

A kinematic toy behind the real protocol (protocol.py). A clock thread advances skills on wall time:
grasping skills reach for the object for the first half of their duration (grasp check at the
midpoint against where the object was when the skill started or was last re-targeted), carry it for
the second half, and set it down on completion; push teleports the object 6 cm on completion;
survey and hold just take their time. A scripted person can move an object, put a hand near the
arm, or trip the safety filter, at a time or when a condition on the world holds, so scene_change
and safety_trip events occur. Events are emitted outside the lock, on the clock thread.

Choices the protocol leaves open, made here so the brain can be tested (docs: report, not settle):
  - precondition() checks the world only (objects, places, grip ability, STOP), not whether the arm
    is busy: the brain asks it about steps other than the running one.
  - start() replaces whatever skill the arm has (running, holding or paused) without an event: the
    protocol has no cancel, so a new plan that drops the current step starts the next one instead.
  - A failed precondition at start() arrives as skill_failed on the next tick, as the protocol says.

Units: metres and seconds. Base frame: +x forward, +y left, +z up (as on the real arm).
"""
from __future__ import annotations

import copy
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .protocol import (ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, ObjectObs, PlaceObs, RobotEvent, SkillSpec,
                       SkillStatus, WorldState)

# ------------------------------------------------------------------ manifests

DIRECTIONS = ("left", "right", "toward_robot", "away_from_robot")
_DIR_XY = {"left": (0.0, 1.0), "right": (0.0, -1.0), "toward_robot": (-1.0, 0.0), "away_from_robot": (1.0, 0.0)}
PUSH_M = 0.06
WORKSPACE = ((0.10, 0.50), (-0.30, 0.30), (-0.02, 0.35))
HOME = (0.20, 0.0)

_OBJ = {"type": "string", "description": "the object id"}


def _args(**props) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


MOVE_OBJECT = SkillSpec("move_object", "pick the object up, carry it and put it down in the place (a tray slot, a bin, or "
                        "\"table\" for a free spot on the table)",
                        _args(object=_OBJ, target={"type": "string", "description": "a place id: a slot, a bin or table"}))
STACK_ON = SkillSpec("stack_on", "pick the object up and set it on top of the target object",
                     _args(object=_OBJ, target={"type": "string", "description": "the id of the object to stack on"}))
PUSH = SkillSpec("push", f"slide the object about {PUSH_M * 100:.0f} cm in a direction (the robot's point of view) with "
                 "one fingertip, without picking it up",
                 _args(object=_OBJ, direction={"type": "string", "enum": list(DIRECTIONS), "description": "which way"}))
HAND_OVER = SkillSpec("hand_over", "pick the object up and hold it out to the person; done when the person has taken it",
                      _args(object=_OBJ))
SURVEY = SkillSpec("survey", "lift the gripper and look over the table again", _args())
HOLD = SkillSpec("hold", "stay still for about a second", _args())
GRASPING = ("move_object", "stack_on", "hand_over")


def widowx_like(robot: str = "widowx-stub") -> Manifest:
    arm = ArmSpec("arm_0", "base", WORKSPACE, GripperSpec(max_width=0.044, can_grasp=True), travel_z=0.12,
                  description="one WidowX AI-like follower arm")
    return Manifest(robot, (arm,), (MOVE_OBJECT, STACK_ON, PUSH, HAND_OVER, SURVEY, HOLD))


def push_only(robot: str = "pusher-stub") -> Manifest:
    arm = ArmSpec("arm_0", "base", WORKSPACE, GripperSpec(max_width=0.0, can_grasp=False), travel_z=0.12,
                  description="one arm with a fixed fingertip: it can touch and push, not grasp")
    return Manifest(robot, (arm,), (PUSH, SURVEY, HOLD))


def sort_scene(numbers=(1, 3, 5), n_slots: int | None = None) -> tuple[list[ObjectObs], list[PlaceObs]]:
    """Numbered 4 cm blocks on the table in front of a row of tray slots; slot 1 is the leftmost (+y)."""
    n = n_slots or len(numbers)
    slots = [PlaceObs(f"tray_slot_{i}", f"tray slot {i}" + (" (the leftmost slot)" if i == 1 else " (the rightmost slot)" if i == n else ""),
                      "slot", 0.36, 0.12 - 0.06 * (i - 1), 1) for i in range(1, n + 1)]
    places = slots + [PlaceObs("tray", "the tray (any slot)", "group", 0.36, 0.0, 0, tuple(p.id for p in slots)),
                      PlaceObs("table", "the table", "table", 0.22, -0.22, 0, radius=0.30),
                      PlaceObs("person", "the person", "person", 0.55, 0.0, 0)]
    objs = [ObjectObs(f"block_{k}", f"block {k}", 0.20, 0.15 - 0.10 * i, 0.0, "table", label=str(k), size=0.04)
            for i, k in enumerate(numbers)]
    return objs, places


# ------------------------------------------------------------------ the server

DURATIONS = {"move_object": 1.0, "stack_on": 1.0, "hand_over": 1.5, "push": 0.6, "survey": 0.5, "hold": 1.0}


@dataclass
class _Run:
    id: str
    arm: str
    name: str
    args: dict
    duration: float
    start_xy: tuple
    aim: tuple | None                   # where the object was at start or the last retarget()
    elapsed: float = 0.0
    state: str = "running"              # running | holding | paused | done | failed
    grasped: bool = False
    reason: str | None = None
    reported: bool = False              # a could-not-start failure has been sent as an event


@dataclass
class _Beat:
    when: float | Callable[[WorldState], bool]
    act: Callable[[], list]
    fired: bool = False


@dataclass
class StubRobotServer:
    manifest_: Manifest
    objects: list[ObjectObs] = field(default_factory=list)
    places: list[PlaceObs] = field(default_factory=list)
    durations: dict = field(default_factory=dict)
    tick_s: float = 0.02
    heartbeat_timeout_s: float = 1.0

    def __post_init__(self):
        self._lock = threading.RLock()
        self._objs = {o.id: o for o in self.objects}
        self._places = {p.id: p for p in self.places}
        z = {a.id: a.travel_z for a in self.manifest_.arms}
        self._arms = {a.id: ArmObs(a.id, *HOME, z[a.id], None, a.gripper.max_width if a.gripper else 0.0, "idle")
                      for a in self.manifest_.arms}
        self._hands: list[HandObs] = []
        self._runs: dict[str, _Run] = {}
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._beats: list[_Beat] = []
        self._ids = itertools.count(1)
        self._dur = {**DURATIONS, **self.durations}
        self._t0 = time.monotonic()
        self._last_hb: float | None = None
        self._hb_lost = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.history: list[tuple[float, str, str]] = []     # (t, arm, mode) on every change, for tests
        self.calls: list[tuple[float, str, tuple]] = []     # (t, tool, args): what the brain asked for

    # ---------------------------------------------------------------- lifecycle (not protocol)
    def open(self) -> "StubRobotServer":
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._clock, name="stub-robot", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def now(self) -> float:
        return time.monotonic() - self._t0

    # ---------------------------------------------------------------- the scripted person (not protocol)
    def at(self, when, act: Callable[[], list]):
        """Run `act` (under the lock; returns events to emit) at `when` s, or once when(world) is true."""
        self._beats.append(_Beat(when, act))

    def person_moves(self, when, oid: str, to: str | None = None, xy: tuple | None = None):
        """The person moves an object to a place, or to an (x, y) on the table."""
        def act():
            o = self._objs[oid]
            if to is not None:
                p = self._places[to]
                o.x, o.y, o.where = p.x, p.y, to
            else:
                o.x, o.y, o.where = xy[0], xy[1], "table"
            return [self._ev("scene_change", f"a person moved {o.name}; it is now {self._where_text(o.where)}", object=oid)]
        self.at(when, act)

    def person_hand(self, when, for_s: float = 1.0, dist: float = 0.15, speed: float = 0.1, arm: str = "arm_0"):
        """A hand appears `dist` m from the gripper, reaching toward it at `speed` m/s (reported, not
        moved: the toy hand holds its pose), and leaves `for_s` s later."""
        def leave():
            self._hands = []
            return [self._ev("scene_change", "the person's hand has gone out of view")]

        def act():
            a = self._arms[arm]
            ang = math.atan2(-a.y, 0.6 - a.x)     # from the person's side of the table
            hx, hy = a.x + dist * math.cos(ang), a.y + dist * math.sin(ang)
            self._hands = [HandObs(hx, hy, a.z, -speed * math.cos(ang), -speed * math.sin(ang))]
            self.at(self.now() + for_s, leave)
            return [self._ev("scene_change", f"a person's hand appeared {dist:.2f} m from the gripper")]
        self.at(when, act)

    def trip(self, when, text: str = "effort limit exceeded", arm: str = "arm_0"):
        def act():
            for r in self._runs.values():
                if r.arm == arm and r.state == "running":
                    r.state = "holding"
            self._mode(arm, "tripped")
            return [self._ev("safety_trip", f"the safety filter tripped: {text}", arm=arm)]
        self.at(when, act)

    # ---------------------------------------------------------------- protocol
    def manifest(self) -> Manifest:
        return self.manifest_

    def world(self) -> WorldState:
        with self._lock:
            return WorldState(self.now(), copy.deepcopy(self._objs), copy.deepcopy(self._places),
                              copy.deepcopy(self._hands), copy.deepcopy(self._arms))

    def status(self, skill_id: str) -> SkillStatus:
        with self._lock:
            r = self._runs[skill_id]
            return SkillStatus(r.id, r.arm, r.name, r.state, self._phase(r), r.reason, self._heading(r), dict(r.args))

    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        with self._lock:
            return self._why_not(arm, skill, args)

    def start(self, arm: str, skill: str, args: dict) -> str:
        with self._lock:
            self.calls.append((self.now(), "start", (arm, skill, dict(args))))
            for rid in [rid for rid, r in self._runs.items() if r.arm == arm and r.state in ("running", "holding", "paused")]:
                del self._runs[rid]                      # replaced: the protocol has no cancel
            sid = f"sk{next(self._ids)}"
            why = self._why_not(arm, skill, args)
            a = self._arms[arm]
            o = self._objs.get(args.get("object", ""))
            r = _Run(sid, arm, skill, dict(args), self._dur.get(skill, 1.0), (a.x, a.y), (o.x, o.y) if o else None)
            self._runs[sid] = r
            if why:
                r.state, r.reason = "failed", f"could not start: {why}"
            elif skill in GRASPING and a.holding == args["object"]:
                r.elapsed, r.grasped = r.duration / 2, True     # already in the gripper: carry from here
            if not why:
                self._mode(arm, "running")
            return sid

    def hold(self, arm: str) -> None:
        self._set(arm, "hold", ("running", "paused"), "holding")

    def pause(self, arm: str) -> None:
        self._set(arm, "pause", ("running", "holding"), "paused")

    def resume(self, arm: str) -> None:
        self._set(arm, "resume", ("holding", "paused"), "running")

    def retarget(self, skill_id: str) -> None:
        with self._lock:
            self.calls.append((self.now(), "retarget", (skill_id,)))
            r = self._runs.get(skill_id)
            o = self._objs.get(r.args.get("object", "")) if r else None
            if r and o and not r.grasped:
                r.aim = (o.x, o.y)

    def stop(self) -> None:
        evs = []
        with self._lock:
            self.calls.append((self.now(), "stop", ()))
            for r in self._runs.values():
                if r.state in ("running", "holding", "paused"):
                    r.state, r.reason, r.reported = "failed", "STOP", True
                    evs.append(self._ev("skill_failed", f"{r.name} stopped by STOP", skill_id=r.id, arm=r.arm))
            for a in self._arms.values():
                if a.holding:                            # parking sets the held object down where the arm is
                    o = self._objs[a.holding]
                    o.where, a.holding = "table", None
                a.x, a.y = HOME
                self._mode(a.id, "stopped")
        self._emit(evs)

    def heartbeat(self) -> None:
        with self._lock:
            self._last_hb, self._hb_lost = self.now(), False

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    # ---------------------------------------------------------------- internals
    def _set(self, arm: str, tool: str, frm: tuple, to: str):
        with self._lock:
            self.calls.append((self.now(), tool, (arm,)))
            if self._arms[arm].mode == "stopped":
                return
            runs = [r for r in self._runs.values() if r.arm == arm and r.state in frm]
            for r in runs:
                r.state = to
            if runs:                                      # an idle arm has nothing to hold or resume
                self._mode(arm, to)

    def _mode(self, arm: str, mode: str):
        a = self._arms[arm]
        if a.mode != mode:
            a.mode = mode
            self.history.append((self.now(), arm, mode))

    def _ev(self, kind: str, text: str, **kw) -> RobotEvent:
        return RobotEvent(self.now(), kind, text, **kw)

    def _emit(self, evs: list):
        for ev in evs:
            for cb in list(self._subs):
                cb(ev)

    def _clock(self):
        last = time.monotonic()
        while not self._stop.wait(self.tick_s):
            now = time.monotonic()
            dt, last = now - last, now
            with self._lock:
                evs = self._tick(dt)
            self._emit(evs)

    def _tick(self, dt: float) -> list:
        evs = []
        t = self.now()
        if self._last_hb is not None and not self._hb_lost and t - self._last_hb > self.heartbeat_timeout_s:
            self._hb_lost = True
            for r in self._runs.values():
                if r.state == "running":
                    r.state = "holding"
                    self._mode(r.arm, "holding")
            evs.append(self._ev("heartbeat_lost", f"no heartbeat for {self.heartbeat_timeout_s:.1f} s: every arm is holding"))
        ws = None
        for b in list(self._beats):
            if b.fired:
                continue
            if callable(b.when):
                ws = ws or WorldState(t, self._objs, self._places, self._hands, self._arms)
                due = b.when(ws)
            else:
                due = t >= b.when
            if due:
                b.fired = True
                evs += b.act()
        for r in list(self._runs.values()):
            if r.state == "failed" and not r.reported:
                r.reported = True
                evs.append(self._ev("skill_failed", r.reason, skill_id=r.id, arm=r.arm, object=r.args.get("object"),
                                    data={"reason": r.reason}))
            elif r.state == "running":
                evs += self._advance(r, dt)
        return evs

    def _advance(self, r: _Run, dt: float) -> list:
        a = self._arms[r.arm]
        r.elapsed += dt
        f = min(1.0, r.elapsed / r.duration)
        oid = r.args.get("object")
        o = self._objs.get(oid) if oid else None
        if r.name in ("survey", "hold"):
            return self._finish(r) if f >= 1.0 else []
        if not r.grasped:
            if r.aim is not None:
                k = min(1.0, f / 0.5)
                a.x, a.y = r.start_xy[0] + (r.aim[0] - r.start_xy[0]) * k, r.start_xy[1] + (r.aim[1] - r.start_xy[1]) * k
            if f < 0.5:
                return []
            if o is None or o.where == "person" or math.hypot(o.x - r.aim[0], o.y - r.aim[1]) > 0.02:
                return self._fail(r, f"nothing there: {o.name if o else oid} is no longer where the arm reached for it")
            if r.name == "push":
                r.grasped = True                          # in contact with the fingertip
            else:
                o.where, a.holding, r.grasped = f"gripper:{r.arm}", oid, True
        dest = self._dest(r)
        k = (f - 0.5) / 0.5
        a.x, a.y = r.aim[0] + (dest[0] - r.aim[0]) * k, r.aim[1] + (dest[1] - r.aim[1]) * k
        if r.name != "push" and o is not None:
            o.x, o.y = a.x, a.y
        return self._finish(r) if f >= 1.0 else []

    def _dest(self, r: _Run) -> tuple:
        if r.name == "push":
            d = _DIR_XY[r.args["direction"]]
            return (r.aim[0] + PUSH_M * d[0], r.aim[1] + PUSH_M * d[1])
        if r.name == "hand_over":
            p = self._places.get("person")
            return (p.x, p.y) if p else (0.5, 0.0)
        t = r.args["target"]
        if r.name == "stack_on":
            b = self._objs[t]
            return (b.x, b.y)
        if t == "table":                                  # a free spot: next to the table place's centre
            n = sum(1 for o in self._objs.values() if o.where == "table")
            p = self._places[t]
            return (p.x, p.y + 0.06 * n)
        p = self._places[t]
        return (p.x, p.y)

    def _finish(self, r: _Run) -> list:
        a = self._arms[r.arm]
        oid = r.args.get("object")
        if r.name in GRASPING:
            o = self._objs[oid]
            if r.name == "move_object":
                t = r.args["target"]
                p = self._places[t]
                occ = next((x.name for x in self._objs.values() if x.where == t and x.id != oid), None)
                if p.capacity == 1 and occ:
                    return self._fail(r, f"{p.name} is occupied by {occ}")
                where = t
            elif r.name == "stack_on":
                where = f"on:{r.args['target']}"
                o.z = self._objs[r.args["target"]].z + (o.size or 0.04)
            else:
                where = "person"
            x, y = self._dest(r)
            o.x, o.y, o.where, a.holding = x, y, where, None
        elif r.name == "push":
            o = self._objs[oid]
            o.x, o.y = self._dest(r)
        r.state = "done"
        self._mode(r.arm, "idle")
        return [self._ev("skill_done", f"{r.name} finished: {self._phase(r)}", skill_id=r.id, arm=r.arm, object=oid)]

    def _fail(self, r: _Run, why: str) -> list:
        r.state, r.reason, r.reported = "failed", why, True
        self._mode(r.arm, "idle")
        return [self._ev("skill_failed", why, skill_id=r.id, arm=r.arm, object=r.args.get("object"), data={"reason": why})]

    def _why_not(self, arm: str, skill: str, args: dict) -> str | None:
        spec = self.manifest_.skill(skill)
        if arm not in self._arms or spec is None:
            return f"{arm} has no skill {skill}"
        if self._arms[arm].mode == "stopped":
            return "the arm is stopped (STOP)"
        missing = [k for k in spec.args_schema.get("required", []) if k not in args]
        if missing:
            return f"missing {', '.join(missing)}"
        if not args.get("object"):
            return None
        oid, a = args["object"], self._arms[arm]
        o = self._objs.get(oid)
        if o is None:
            return f"{oid} is not in view"
        if o.where == "person":
            return f"{o.name} is held by the person"
        top = next((x.name for x in self._objs.values() if x.where == f"on:{oid}"), None)
        if skill in GRASPING:
            g = next(s.gripper for s in self.manifest_.arms if s.id == arm)
            if g is None or not g.can_grasp:
                return "this arm cannot grasp"
            if a.holding and a.holding != oid:
                return f"the gripper is holding {self._objs[a.holding].name}"
            if top:
                return f"{o.name} has {top} on top of it"
        if skill == "move_object":
            p = self._places.get(args["target"])
            if p is None or p.kind not in ("slot", "bin", "table"):
                return f"{args['target']} is not a place to put things"
        if skill == "stack_on" and (args["target"] not in self._objs or args["target"] == oid):
            return f"{args['target']} is not another object"
        if skill == "push":
            if args.get("direction") not in DIRECTIONS:
                return "push needs a direction"
            if o.where.startswith("gripper"):
                return f"{o.name} is in the gripper"
        return None

    def _where_text(self, where: str) -> str:
        if where in self._places:
            return f"in {self._places[where].name}"
        if where.startswith("on:"):
            return f"on top of {self._objs[where[3:]].name}"
        return {"table": "on the table", "person": "held by the person"}.get(where, where)

    def _phase(self, r: _Run) -> str:
        o = self._objs.get(r.args.get("object", ""))
        on = o.name if o else ""
        if r.name in ("survey", "hold"):
            return {"survey": "lifted, looking over the table", "hold": "holding still"}[r.name]
        dest = (self._places[r.args["target"]].name if r.name == "move_object" else
                self._objs[r.args["target"]].name if r.name == "stack_on" else
                "the person" if r.name == "hand_over" else r.args.get("direction", ""))
        if r.state == "done":
            return f"{on} put down at {dest}" if r.name != "push" else f"{on} pushed {dest}"
        if r.name == "push":
            return f"pushing {on} {dest}" if r.grasped else f"moving to {on} to push it {dest}, not yet touching it"
        if r.grasped:
            return f"carrying {on} to {dest}, not yet released"
        return f"moving to {on} to pick it up, not yet grasped"

    def _heading(self, r: _Run) -> tuple | None:
        if r.state in ("done", "failed") or r.aim is None:
            return None
        return self._dest(r) if r.grasped else r.aim
