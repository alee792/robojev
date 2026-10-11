"""What every WidowX robot server shares above its backend: the protocol's skill and control surface
over `skills.Skill` runs, the heartbeat watchdog, the safety-trip bookkeeping, the event fan-out,
the catalog argument checks and the shared-zone lock's rules. `SimRobotServer` (MuJoCo) and
`RealRobotServer` (trossen_arm, robots/widowx_real) subclass `SkillServer` and add only their clock,
their arms, `world()`, `manifest()` and `stop()`.

Extracted from the sim server at D1 so the real server runs the same code paths, not a copy. The
choices the sim server's docstring lists (refusal by ValueError, replacement on a busy arm, the
watchdog arming on the first heartbeat, no resume on a fresh heartbeat) live here.

Frames, since K5: an arm's `skills.ArmIO` surface speaks its own base frame (the policies never
know where the base stands), the server's world speaks the common frame. The argument checks read
positions through the arm (`a.obj`, `a.place`), so "reachable" is judged in the frame the arm's
workspace box is in; ids, places and occupancy come from the frame-free `WorldView`.
"""
from __future__ import annotations

import itertools
import math
import threading
from typing import Callable, Iterable, Protocol

from robojev.protocol import Manifest, PlaceObs, Recorder, RobotEvent, SkillStatus
from robojev.robots.widowx_sim import skills


class ServedArm(skills.ArmIO, Protocol):
    """What the server needs from an arm besides its `skills.ArmIO` surface. Fields are read and
    written under the server's lock."""
    id: str
    skill: skills.Skill | None
    stopped: bool
    frozen_why: str | None           # why the setpoint is frozen with no skill to say so: "heartbeat" | "trip"

    @property
    def active(self) -> bool:
        """a skill is running (or holding, or paused) on this arm"""
    @property
    def frozen(self) -> bool:
        """the setpoint is frozen, whoever froze it"""
    def mode(self) -> str: ...
    def observation(self) -> dict: ...
    def action(self) -> dict: ...


class WorldView(Protocol):
    """The frame-free reads the argument checks need: the sim's ground truth or the real server's
    bookkeeping. Positions are read through the arm instead."""
    places: dict[str, PlaceObs]

    def ids(self) -> Iterable[str]:
        """every object id"""
    def pose(self, oid: str) -> skills.ObjPose | None: ...
    def where(self, oid: str, held_by: dict[str, str]) -> str: ...
    def occupant(self, pid: str, held_by: dict[str, str]) -> str | None: ...


class LagWatch:
    """Fires once when the EE has lagged the setpoint by more than `trip_m` for `persist_s`, and
    re-arms when the lag drops back under: one trip per obstacle, not one per physics step."""

    def __init__(self, trip_m: float, persist_s: float):
        self.trip_m, self.persist_s = trip_m, persist_s
        self.over_s = 0.0
        self.fired = False

    def update(self, lag: float, dt: float) -> bool:
        if lag <= self.trip_m:
            self.over_s, self.fired = 0.0, False
            return False
        self.over_s += dt
        if self.over_s >= self.persist_s and not self.fired:
            self.fired = True
            return True
        return False


class ZoneLock:
    """The shared-zone lock's rules (docs/widowx-pair.md): one arm in the zone at a time, no model in
    the loop. The geometry is the backend's; each tick it says per arm whether the arm is `inside`
    the zone (its fingers overlap the strip), `entering` it (a running skill is about to), and
    `beyond` its side line (closer to the zone's centre line than the other arm can be while an arm
    works there: the WidowX wrist housing stands 10 cm ahead of the fingertips). The rules:

    - The holder is the arm inside. It keeps the lock while it is inside or still entering, and
      loses it the tick it is neither: it left, or its skill stopped before it went in (held, failed,
      replaced), so a granted lock is never kept by an arm that is not moving.
    - An arm that wants in while another holds the lock waits; waiters are granted in the order
      they first asked, the moment the lock is free AND every other arm is behind its side line
      (else the other arm's wrist would meet the holder's over the zone). A waiter that cannot be
      granted for that reason is told which arm is in the way; an arm that stops wanting in leaves
      the queue.
    - A waiter that is itself beyond its side line, queued behind another, must back off to its
      side line (`yielding`): otherwise two arms carrying toward the zone at once would both stand
      at the strip's edge, blocking each other for ever. The one that asked first is granted the
      moment the other is back.
    - No deadlock by construction: at most one arm is inside, a waiting arm stands still or backs
      off, the server moves an idle arm back behind its side line (`SimRobotServer._park`), so the
      holder always leaves unless its skill failed in there, which the brain resolves by starting
      its next skill on that arm; the waiter's status names the holder meanwhile.
    - Nothing here can be overridden from the protocol: resume() on a waiting skill leaves it waiting.

    `step()` returns, for every arm that must wait this tick, why ("held by arm_0", or "arm_1 is
    still within reach of it"); `yielding` is then the waiters that must back off."""

    def __init__(self):
        self.holder: str | None = None
        self.queue: list[str] = []          # arms waiting, in request order
        self.yielding: set[str] = set()

    def step(self, inside: dict[str, bool], entering: dict[str, bool], beyond: dict[str, bool]) -> dict[str, str]:
        if self.holder is not None and not inside[self.holder] and not entering[self.holder]:
            self.holder = None
        if self.holder is None:
            self.holder = next((a for a, ins in inside.items() if ins), None)
        for arm, ent in entering.items():
            if ent and arm != self.holder and arm not in self.queue:
                self.queue.append(arm)
        self.queue = [a for a in self.queue if entering.get(a) and a != self.holder]
        if self.holder is None:
            for cand in self.queue:
                if not any(beyond[o] for o in inside if o != cand):
                    self.holder = cand
                    self.queue.remove(cand)
                    break
        waits: dict[str, str] = {}
        for a in self.queue:
            if self.holder is not None:
                waits[a] = f"held by {self.holder}"
            else:
                blocker = next(o for o in inside if o != a and beyond[o])
                waits[a] = f"{blocker} is still within reach of it"
        first = self.queue[:1] if self.holder is None else []
        self.yielding = {a for a in self.queue if beyond[a] and a not in first}
        return waits


def check_args(spec_schema: dict, skill: str, args: dict) -> str | None:
    """The catalog schema check: required, unknown, type, enum and range. None or a literal reason."""
    missing = [k for k in spec_schema.get("required", []) if k not in args]
    if missing:
        return f"precondition: {skill} needs {', '.join(missing)}"
    extra = [k for k in args if k not in spec_schema.get("properties", {})]
    if extra:
        return f"precondition: {skill} takes no argument {', '.join(extra)}"
    for k, v in args.items():
        prop = spec_schema["properties"][k]
        kind = prop.get("type")
        if kind == "string" and not isinstance(v, str) or kind == "number" and not isinstance(v, (int, float)):
            return f"precondition: {k} must be a {kind}"
        if "enum" in prop and v not in prop["enum"]:
            return f"precondition: {k} must be one of {', '.join(prop['enum'])}, not {v!r}"
        if kind == "number" and not prop.get("minimum", -math.inf) <= v <= prop.get("maximum", math.inf):
            return f"precondition: {k} must be between {prop.get('minimum')} and {prop.get('maximum')}"
    return None


class SkillServer:
    """protocol.RobotServer minus the backend. A subclass sets, before any call: `arms` (id ->
    ServedArm, each also a skills.ArmIO), `world_view`, `travel_z`, `table_z`, `max_width`, and
    implements `t`, `manifest()`, `world()` and `stop()`. Its tick calls `_watchdog(evs)` before the
    policies and `_tick_skills(evs)` for them, `_trip(...)` when its safety filter fires, and
    `_emit(evs)` after releasing the lock."""

    arms: dict[str, ServedArm]
    world_view: WorldView
    travel_z: float
    table_z: float
    max_width: float

    def __init__(self, recorder: Recorder | None, params: skills.Params | None, heartbeat_timeout_s: float):
        self.recorder = recorder
        self.params = params or skills.Params()
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._lock = threading.RLock()
        self._runs: dict[str, skills.Skill] = {}
        self._ids = itertools.count(1)
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._last_hb: float | None = None
        self._hb_lost = False
        self._stopped = False

    @property
    def t(self) -> float:
        """the server's clock, seconds; events and the heartbeat timeout are measured on it"""
        raise NotImplementedError

    def manifest(self) -> Manifest:
        raise NotImplementedError

    # ---------------------------------------------------------------- protocol: resources and tools
    def status(self, skill_id: str) -> SkillStatus:
        with self._lock:
            run = self._runs.get(skill_id)
            if run is None:
                raise KeyError(f"no skill {skill_id}")
            return run.status()

    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        with self._lock:
            return self._why_not(arm, skill, args)

    def start(self, arm: str, skill: str, args: dict) -> str:
        with self._lock:
            why = self._why_not(arm, skill, args)
            if why:
                raise ValueError(why)
            a = self.arms[arm]
            sid = f"sk{next(self._ids)}"
            if a.active:
                old = a.skill
                old.state, old.reason, old.notified = "cancelled", f"cancelled: replaced by {sid}", True
            a.unfreeze()
            a.skill = skills.SKILLS[skill](sid, arm, args, self.params)
            self._runs[sid] = a.skill
            note = getattr(self.recorder, "note", None)       # outside protocol.Recorder; the JSONL recorder has it
            if note is not None:
                note(kind="skill_start", skill_id=sid, arm=arm, skill=skill, args=dict(args))
            return sid

    def hold(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.hold(a)

    def pause(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.pause(a)

    def resume(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.resume(a)
                if a.skill.state == "holding":        # still waiting for the shared zone: the lock outranks resume
                    return
            a.unfreeze()

    def retarget(self, skill_id: str) -> None:
        with self._lock:
            run = self._runs.get(skill_id)
            if run is None or run.finished:
                return
            run.retarget(self.arms[run.arm])

    def heartbeat(self) -> None:
        with self._lock:
            self._last_hb, self._hb_lost = self.t, False

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            if callback in self._subs:
                self._subs.remove(callback)

    # ---------------------------------------------------------------- for the subclass's tick
    def _watchdog(self, evs: list[RobotEvent]) -> None:
        """Hold every arm once the heartbeat is `heartbeat_timeout_s` late. Arms on the first
        heartbeat(); fires once per loss."""
        t = self.t
        if self._last_hb is not None and not self._hb_lost and t - self._last_hb > self.heartbeat_timeout_s:
            self._hb_lost = True
            for a in self.arms.values():
                if a.stopped:
                    continue
                if a.active:
                    a.skill.hold(a)
                elif not a.frozen:
                    a.freeze()
                    a.frozen_why = "heartbeat"
            evs.append(self._ev("heartbeat_lost", f"no heartbeat for {self.heartbeat_timeout_s:.1f} s: every arm is holding"))

    def _tick_skills(self, evs: list[RobotEvent]) -> None:
        """One policy tick per arm, a recorder frame per tick, and the done/failed event of a skill
        that finished (in its tick or outside it: retarget, STOP)."""
        t = self.t
        for a in self.arms.values():
            a.t = t
            s = a.skill
            if s is None:
                continue
            if not s.finished:
                s.tick(a)
                if self.recorder is not None:
                    self.recorder.frame(t, a.id, s.id, s.phase, a.observation(), a.action())
            if s.finished and not s.notified:
                s.notified = True
                if s.state == "done":
                    evs.append(self._ev("skill_done", f"{s.name} finished: {s.text}", skill_id=s.id, arm=a.id,
                                        object=s.args.get("object")))
                else:
                    evs.append(self._ev("skill_failed", s.reason, skill_id=s.id, arm=a.id, object=s.args.get("object"),
                                        data={"phase": s.phase}))

    def _apply_zone_waits(self, waits: dict[str, str]) -> None:
        """The lock's verdict for this tick: skills on arms in `waits` wait out of the zone (the text
        says why: "held by arm_0"); a skill that was waiting and is not in `waits` proceeds. Only
        running skills are ever waiting, so an idle arm needs nothing here."""
        for a in self.arms.values():
            s = a.skill
            if s is None or s.finished:
                continue
            if a.id in waits:
                s.wait(a, waits[a.id])
            elif s.waiting is not None:
                s.proceed(a)

    def _trip(self, a: ServedArm, why: str, data: dict, evs: list[RobotEvent]) -> None:
        """The safety filter fired on `a`: hold its skill (resume() is the only way out) or freeze
        the idle arm, and say so."""
        if a.stopped:
            return
        sid = None
        if a.active:
            a.skill.trip(a, why)
            sid = a.skill.id
        else:
            a.freeze()
            a.frozen_why = "trip"
        evs.append(self._ev("safety_trip", f"the safety filter tripped on {a.id}: {why}; holding", skill_id=sid, arm=a.id,
                            data=data))

    def _fail_all(self, reason: str, evs: list[RobotEvent]) -> None:
        """STOP's first half: every running skill ends failed with `reason` and an event; every arm
        reads stopped. The subclass then parks."""
        self._stopped = True
        for a in self.arms.values():
            if a.active:
                s = a.skill
                s.fail(a, reason)
                s.notified = True
                evs.append(self._ev("skill_failed", s.reason, skill_id=s.id, arm=a.id, object=s.args.get("object")))
            a.stopped = True

    # ---------------------------------------------------------------- internals
    def _arm(self, arm: str) -> ServedArm:
        a = self.arms.get(arm)
        if a is None:
            raise ValueError(f"no arm {arm}")
        return a

    def _held_by(self) -> dict[str, str]:
        return {oid: a.id for a in self.arms.values() for oid in [a.held()] if oid}

    def _claimed_by(self) -> dict[str, str]:
        """object id -> the arm whose running skill names it as its object (ObjectObs.claimed_by)."""
        return {a.skill.args["object"]: a.id for a in self.arms.values()
                if a.active and "object" in a.skill.args}

    def _ev(self, kind: str, text: str, **kw) -> RobotEvent:
        return RobotEvent(self.t, kind, text, **kw)

    def _emit(self, evs: list[RobotEvent]) -> None:
        for ev in evs:
            if self.recorder is not None:
                self.recorder.event(ev)
            for cb in list(self._subs):
                cb(ev)

    def _why_not(self, arm: str, skill: str, args: dict) -> str | None:
        """None, or a literal reason prefixed with a catalog.REASONS code. Positions come through the
        arm (its own frame, where its workspace box is); ids, places and occupancy from the world."""
        a = self.arms.get(arm)
        spec = self.manifest().skill(skill)
        if a is None:
            return f"precondition: no arm {arm}"
        if spec is None:
            return f"precondition: {arm} has no skill {skill}"
        if a.stopped:
            return "precondition: the arm is stopped (STOP)"
        why = check_args(spec.args_schema, skill, args)
        if why:
            return why
        w, travel_z = self.world_view, self.travel_z
        held = self._held_by()
        if skill in ("pick_and_place", "stack_on", "hand_over"):
            why = self._grasp_why_not(a, args["object"], held)
            if why:
                return why
        if skill == "pick_and_place":
            oid, pid = args["object"], args["place"]
            pl = w.places.get(pid)
            if pl is None:
                return f"precondition: there is no place {pid} in the world"
            occ = w.occupant(pid, held)
            if pl.kind == "slot" and occ and occ != oid:
                return f"blocked: {pl.name} is occupied by {occ}"
            xy = a.place(pid)
            if xy is None:
                return f"blocked: {pl.name} has no free slot"
            if not a.reachable((xy[0], xy[1], travel_z)):
                return f"unreachable: {pid} at ({xy[0]:.2f}, {xy[1]:.2f}) is outside the arm's workspace"
        if skill == "stack_on":
            oid, onto = args["object"], args["onto"]
            b = a.obj(onto)
            if b is None:
                return f"not_in_view: there is no object {onto} in the world"
            if onto == oid:
                return f"precondition: {oid} cannot be stacked on itself"
            if onto in held:
                return f"precondition: {onto} is held by {held[onto]}"
            if not a.reachable((b.x, b.y, travel_z)):
                return f"unreachable: {onto} at ({b.x:.2f}, {b.y:.2f}) is outside the arm's workspace"
            top = next((o for o in w.ids() if o != oid and w.where(o, held) == f"on:{onto}"), None)
            if top:
                return f"blocked: {onto} already has {top} on it"
            o = a.obj(oid)
            if b.z + b.size / 2 + o.size + self.params.tip_height > travel_z:      # the carried block would hit it
                return f"unreachable: the top of {onto} is {b.z + b.size / 2 - self.table_z:.3f} m above the table, too high to stack on"
        if skill == "push":
            oid = args["object"]
            o = a.obj(oid)
            if o is None:
                return f"not_in_view: there is no object {oid} in the world"
            if oid in held:
                return f"precondition: {oid} is held by {held[oid]}"
            if a.held():
                return f"precondition: the gripper is holding {a.held()}"
            claimed = self._claimed_by().get(oid)
            if claimed and claimed != a.id:
                return f"precondition: {oid} is claimed by {claimed}: its {self.arms[claimed].skill.name} is running"
            start, limit = skills.push_path(o, args["direction"], args["distance"], self.params)
            push_z = self.table_z + self.params.tip_height
            for xy, what in ((start, "starts"), (limit, "ends")):
                if not a.reachable((xy[0], xy[1], push_z)):
                    return (f"unreachable: pushing {oid} {args['distance'] * 100:.0f} cm {args['direction']} {what} at "
                            f"({xy[0]:.2f}, {xy[1]:.2f}), outside the arm's workspace")
        if skill == "hand_over":
            ox, oy = self.params.offer_xy
            if not a.reachable((ox, oy, self.params.offer_z)):
                return f"unreachable: the hand-over point ({ox:.2f}, {oy:.2f}) is outside the arm's workspace"
        return None

    def _grasp_why_not(self, a: ServedArm, oid: str, held: dict[str, str]) -> str | None:
        """What stops this arm picking `oid` up: unknown, too wide, in another gripper, this gripper
        full of something else, claimed by another arm's running skill, out of reach. Already in this
        gripper is fine (the skill starts at lift)."""
        o = a.obj(oid)
        if o is None:
            return f"not_in_view: there is no object {oid} in the world"
        if o.size + 0.004 > self.max_width:
            return f"precondition: {oid} is {o.size * 100:.1f} cm wide; the fingers open to {self.max_width * 100:.1f} cm"
        mine = a.held()
        if mine and mine != oid:
            return f"precondition: the gripper is holding {mine}"
        if held.get(oid, a.id) != a.id:
            return f"precondition: {oid} is held by {held[oid]}"
        claimed = self._claimed_by().get(oid)
        if claimed and claimed != a.id:
            return f"precondition: {oid} is claimed by {claimed}: its {self.arms[claimed].skill.name} is running"
        if not a.reachable((o.x, o.y, self.travel_z)):
            return f"unreachable: {oid} at ({o.x:.2f}, {o.y:.2f}) is outside the arm's workspace"
        return None
