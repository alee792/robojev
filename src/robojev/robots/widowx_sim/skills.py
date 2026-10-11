"""Skill policies: run by the robot server at the policy rate (50 Hz). One tick = read the arm and
the world, set the next goal. They never touch MuJoCo or a driver: everything goes through the
`ArmIO` surface the server hands them, so the same policies sit on the real backend at D1.

The grasp (measured on the model, 2026-10-10, correcting the K1 note in docs/harness-spikes.md):
the fingers are tapered. Only the fingertip pieces, 0-1.5 cm behind the EE point, reach the
centreline when closed; the pad boxes fall away at ~7 deg (3.4 mm each at 4 cm, 7.4 mm at 7 cm).
So a 4 cm cube is held by a fingertip pinch on its lower 1.5 cm: the gripper points straight down
(90 deg pitch, reachable out to x = 0.42 at travel height), the tips go to 0.5 cm above the table,
and the jaws close to the block's width minus a squeeze. Pitch is never chosen; only yaw is, aligned
with the block's faces (mod 90 deg) and nearest the direction perpendicular to the arm's azimuth,
which is where the wrist roll is zero, so the roll stays within +-45 deg.

The pre-grasp opening shrinks for neighbours (the L1 follow-up): the pads' outer faces sit
`finger_t` outside the fingertips, so next to another block the jaws open only as wide as keeps a
pad 5 mm clear of it, never less than the block plus 8 mm. A 6 cm slot pitch picks (tested).

The push (K4): the closed gripper's pad boxes stick out past the fingertips on every side (13.6 mm
along the closing axis, 13.7 / 19.7 mm across it), so a block is pushed by the pads, which start
1.35 cm above the fingertips. The fingertips go to grasp height, which puts the push resultant
~2.9 cm up a 4 cm cube; it slides without tipping only while the block-table friction is under
s / 2h = 0.7, which is why scene.py gives that pair 0.5 (at 1.0 the cube rolled over). The slide is
closed-loop on the object: the fingers advance until the object has travelled the distance.

Every phase has literal text the decision loop can read. hold()/pause() work from any phase: the
setpoint freezes where it is and the gripper command is left alone, so a held grip stays a grip.
resume() continues the same phase, because each phase re-issues its goal every tick and keeps no
plan beyond it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from robojev import catalog
from robojev.protocol import HandObs, SkillStatus


@dataclass
class ObjPose:
    x: float
    y: float
    z: float
    yaw: float            # rad about +z
    size: float           # edge, m


class ArmIO(Protocol):
    """What a skill may read and command. Implemented by the server, one per arm. Positions are in
    the base frame, metres and seconds; `t` is the server's clock (sim time in sim)."""
    t: float
    table_z: float
    travel_z: float
    max_width: float      # m between the fingertips fully open

    def ee(self) -> tuple[float, float, float]: ...
    def ee_yaw(self) -> float:
        """direction of the closing axis in the xy plane, rad"""
    def arrived(self) -> bool:
        """the setpoint has reached the goal (the EE may still lag a little)"""
    def width(self) -> float:
        """m between the fingertips now"""
    def reachable(self, xyz) -> bool:
        """inside the workspace box the safety filter clamps to"""
    def obj(self, oid: str) -> ObjPose | None: ...
    def objects(self) -> dict[str, ObjPose]:
        """every object's pose: neighbours of a grasp, what sits on a stack"""
    def where(self, oid: str) -> str:
        """the object's `where` as world() reports it (place id | table | on:<id> | gripper:<arm>)"""
    def place(self, pid: str) -> tuple[float, float] | None:
        """where to put something down for this place id (a group resolves to a free member)"""
    def hands(self) -> list[HandObs]:
        """the hands perception reports now, as world().hands"""
    def held(self) -> str | None:
        """object id between the closed fingers, from physics (both sides touching, width within
        tolerance of the object's size); None when nothing is"""
    def goto(self, xyz, speed: float, yaw: float | None = None) -> None: ...
    def grip(self, width: float) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...


@dataclass(frozen=True)
class Params:
    """Grasp and motion numbers in one place. Tuned on the K2 and K4 trials (robojev/conformance.py)."""
    tip_height: float = 0.005     # fingertips above the table at grasp: the pinch takes the cube's lower 1.5 cm
    place_drop: float = 0.004     # release this much higher than the grasp height, so the block falls 4 mm
    squeeze: float = 0.008        # command this much narrower than the block: 4 N per side at kp 1000 (sim)
    open_margin: float = 0.03     # open to block width + this before descending (capped at the jaw max)
    open_min: float = 0.008       # ... and never less than block width + this, whatever the neighbours
    finger_t: float = 0.014       # a pad's outer face sits this far outside its fingertip's inner face (model, 2026-10-10)
    neighbour_gap: float = 0.005  # clearance kept between a pad and a neighbouring block
    v_travel: float = 0.15        # m/s at travel height (= the safety filter's cap)
    v_vertical: float = 0.10      # m/s descending and lifting
    v_final: float = 0.05         # m/s for the last 2 cm of a descent
    v_push: float = 0.05          # m/s sliding a block along the table
    push_gap: float = 0.015       # the pads start this far short of the block's face before the slide
    push_slack: float = 0.02      # the slide may run this far past where the object should have arrived
    push_coast: float = 0.004     # stop the slide this short of the distance: the EE's lag behind its setpoint
                                  # at v_push, which it closes after the stop (measured: +5 mm overrun without)
    push_lag_m: float = 0.02      # fingers this far behind where the slide should have them = blocked
                                  # (under the safety filter's 3 cm lag trip; a free slide lags ~6 mm)
    offer_xy: tuple[float, float] = (0.40, 0.0)   # hand-over point: the far edge of the workspace, where a person stands
    offer_z: float = 0.08         # fingertips there, 10 cm above the table
    hand_reach: float = 0.06      # a hand this close laterally and this far below the fingertips can take the object
    xy_tol: float = 0.006         # EE within this of the goal counts as arrived
    z_tol: float = 0.004
    yaw_tol: float = 0.03         # rad, before descending
    drop_grace: float = 0.15      # s without finger contact before a carried block counts as dropped
    settle_s: float = 0.1         # s of still jaws before judging the grasp
    close_s: float = 2.0          # s after which a close or release that has not settled is judged anyway
    timeout_s: float = 15.0       # active (not held) seconds for the whole skill; hand_over's wait gets its own 15
    stall_s: float = 3.0          # s without 3 mm of EE progress in a motion phase
    stall_m: float = 0.003


# Advertised as-is from the standard catalog: shared names are what make recordings transfer.
SPECS = (catalog.PICK_AND_PLACE, catalog.STACK_ON, catalog.PUSH, catalog.HAND_OVER, catalog.HOLD, catalog.SURVEY)

# Push directions in the base frame (+x forward, +y left), as the catalog names them.
DIRECTIONS: dict[str, tuple[float, float]] = {"left": (0.0, 1.0), "right": (0.0, -1.0),
                                              "toward_robot": (-1.0, 0.0), "away_from_robot": (1.0, 0.0)}


def wrap(a: float) -> float:
    """to (-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def grasp_yaw(x: float, y: float, block_yaw: float) -> float:
    """Closing-axis yaw that meets a face of the block squarely and stays nearest the direction
    perpendicular to the arm's azimuth (zero wrist roll). The four candidates differ by 90 deg, so
    the roll is at most +-45 deg."""
    neutral = math.atan2(y, x) + math.pi / 2
    rel = (block_yaw - neutral + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return neutral + rel


def open_width(size: float, others: list[ObjPose], at: tuple[float, float], max_width: float, p: Params) -> float:
    """Pre-grasp opening for a block of `size` at `at`: width + open_margin, shrunk so a pad's outer
    face stays `neighbour_gap` clear of the nearest other block (taken as lying along the closing
    axis, the worst case), never under width + open_min, never over the jaw max."""
    w = size + p.open_margin
    for o in others:
        face = math.hypot(o.x - at[0], o.y - at[1]) - o.size / 2        # that block's near face from our centre
        w = min(w, 2 * (face - p.finger_t - p.neighbour_gap))
    return min(max_width, max(size + p.open_min, w))


def push_path(o: ObjPose, direction: str, distance: float, p: Params) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the fingertips start a push and the furthest they may go: (start, limit), xy. The start
    is `push_gap` short of the block's face on the side opposite the direction, allowing for the pads
    sticking out `finger_t` and for a yawed block's corner; the limit is where the pads would be
    with the block moved `distance`, plus `push_slack`."""
    dx, dy = DIRECTIONS[direction]
    rel = o.yaw - math.atan2(dy, dx)
    reach = o.size / 2 * (abs(math.cos(rel)) + abs(math.sin(rel)))     # the block's extent along the direction
    back = reach + p.finger_t
    start = (o.x - (back + p.push_gap) * dx, o.y - (back + p.push_gap) * dy)
    limit = (o.x + (distance - back + p.push_slack) * dx, o.y + (distance - back + p.push_slack) * dy)
    return start, limit


def hand_beneath(hands: list[HandObs], ee: tuple[float, float, float], p: Params) -> HandObs | None:
    """The nearest hand within `hand_reach` laterally and from 1 cm above to `hand_reach` below the
    fingertips; held out or not."""
    near = [h for h in hands if math.hypot(h.x - ee[0], h.y - ee[1]) <= p.hand_reach and -0.01 <= ee[2] - h.z <= p.hand_reach]
    return min(near, key=lambda h: math.dist((h.x, h.y, h.z), ee), default=None)


def fmt_xyz(p) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"


class Skill:
    """Base: timing, hold/pause/resume, stall and timeout, the status record. Subclasses set `name`
    and `first_phase` and implement `run(io)`; `monitor(io)` runs every tick even while held."""
    name = ""
    first_phase = "start"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        self.id, self.arm, self.args, self.p = sid, arm, dict(args), p
        self.state = "running"          # running | holding | paused | done | failed
        self.reason: str | None = None  # "<code>: <literal text>" once failed
        self.phase = self.first_phase   # named from the start: status() before the first tick is honest
        self.text = "starting"
        self.heading: tuple[float, float] | None = None
        self.tripped = False            # held by the safety filter, not by the brain
        self.trip_text = ""
        self.notified = False           # the server has sent the done/failed event
        self.active_s = 0.0             # time spent running (held time excluded): the timeout
        self.phase_s = 0.0              # same, since the phase began
        self._last_t: float | None = None
        self._dt = 0.0
        self._prog_t = 0.0
        self._prog_p: tuple[float, float, float] | None = None

    # -- lifecycle (called by the server, under its lock) --------------------------------------------
    def tick(self, io: ArmIO) -> None:
        if self.finished:
            return
        if self._last_t is None:
            self._last_t = io.t
            self._enter(io, self.first_phase)
        self._dt = io.t - self._last_t
        self._last_t = io.t
        if self.state in ("holding", "paused"):
            self.monitor(io)
            return
        self.active_s += self._dt
        self.phase_s += self._dt
        if self.active_s > self.p.timeout_s:
            self.fail(io, self.timeout_reason())
            return
        self.monitor(io)
        for _ in range(3):          # a phase that finishes hands over within the tick: the next one
            ph = self.phase         # sets its text and goal at once, so status() never lags a phase
            if self.state != "running":
                break
            self.run(io)
            if self.phase == ph:
                break

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    def hold(self, io: ArmIO) -> None:
        if self.state == "running":
            io.freeze()
            self.state = "holding"

    def pause(self, io: ArmIO) -> None:
        if self.state in ("running", "holding"):
            io.freeze()
            self.state = "paused"

    def trip(self, io: ArmIO, text: str) -> None:
        """The safety filter tripped: hold and remember why; resume() is the only way out. A paused
        skill stays paused (the brain's request outranks the filter's)."""
        if not self.finished:
            io.freeze()
            if self.state == "running":
                self.state = "holding"
            self.tripped, self.trip_text = True, text

    def resume(self, io: ArmIO) -> None:
        if self.state in ("holding", "paused"):
            io.unfreeze()
            self.state = "running"
            self.tripped = False
            self._prog_t, self._prog_p = io.t, io.ee()    # a hold is not a stall

    def retarget(self, io: ArmIO) -> None:
        pass

    # -- helpers ---------------------------------------------------------------------------------
    def _enter(self, io: ArmIO, phase: str) -> None:
        self.phase, self.phase_s = phase, 0.0
        self._prog_t, self._prog_p = io.t, io.ee()

    def _stalled(self, io: ArmIO) -> bool:
        """No `stall_m` of EE progress for `stall_s` while the phase has not finished."""
        p = io.ee()
        if self._prog_p is None or math.dist(p, self._prog_p) > self.p.stall_m:
            self._prog_t, self._prog_p = io.t, p
            return False
        return io.t - self._prog_t > self.p.stall_s

    def _at(self, io: ArmIO, goal, z_tol: float | None = None) -> bool:
        x, y, z = io.ee()
        return (io.arrived() and math.hypot(x - goal[0], y - goal[1]) <= self.p.xy_tol
                and abs(z - goal[2]) <= (z_tol if z_tol is not None else self.p.z_tol))

    def timeout_reason(self) -> str:
        return f"timeout: {self.p.timeout_s:.0f} s of motion, still in phase '{self.phase}' ({self.text})"

    def monitor(self, io: ArmIO) -> None:
        pass

    def run(self, io: ArmIO) -> None:
        raise NotImplementedError

    def done(self, io: ArmIO, text: str) -> None:
        self.state, self.phase, self.text, self.heading = "done", "done", text, None

    def fail(self, io: ArmIO, reason: str) -> None:
        """`reason` is "<code>: <literal text>" with a catalog.REASONS code. The arm freezes where it
        is: the brain decides what happens next."""
        self.state, self.reason, self.heading = "failed", reason, None
        io.freeze()

    def status(self) -> SkillStatus:
        text = self.text
        if self.state == "holding":
            text = (f"safety trip ({self.trip_text}), held: " if self.tripped else "held: ") + text
        elif self.state == "paused":
            text = "paused: " + text
        elif self.state == "failed":
            text = f"failed in phase '{self.phase}': {self.reason}"
        return SkillStatus(self.id, self.arm, self.name, self.state, text, self.reason, self.heading, dict(self.args),
                           phase=self.phase)


class Grasping(Skill):
    """The body every grasping skill shares: approach at travel height -> descend straight down ->
    close -> lift -> carry -> lower -> release -> retreat. Starts at `lift` when the object is already
    between the fingers (a correction replaces a running skill mid-carry; the catalog requires it).
    Subclasses say where to carry to (`_aim_dest`), how low to go (`_lower_z`), whether the fingers
    may open yet (`_may_release`), and the words (`dest_text`, `_done_text`)."""
    first_phase = "approach"
    PHASES = ("approach", "descend", "close", "lift", "carry", "lower", "release", "retreat")

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.obj = str(args["object"])
        self.target: tuple[float, float] | None = None   # where the object was when aimed at
        self.size = 0.04
        self.yaw = 0.0
        self.open_w = 0.0                                # pre-grasp opening, chosen when aimed
        self.release_w = 0.0                             # opening at the destination, chosen on arrival
        self.dest: tuple[float, float] | None = None
        self.dest_text = ""                              # "slot_2", "block_2", "the hand-over point"
        self.releasing = False                           # the fingers have been told to open
        self._w_prev: float | None = None
        self._w_still = 0.0
        self._lost_s = 0.0

    # -- what a subclass decides ---------------------------------------------------------------------
    def _aim_dest(self, io: ArmIO) -> bool:
        """Set `dest` (xy), `dest_text` and `heading`, or fail and return False."""
        raise NotImplementedError

    def _lower_z(self, io: ArmIO) -> float | None:
        """EE height at which to let go, read live; None once `fail` has been called."""
        raise NotImplementedError

    def _may_release(self, io: ArmIO) -> bool:
        """True when the fingers may open; a subclass that waits sets `text` meanwhile."""
        return True

    def _done_text(self) -> str:
        raise NotImplementedError

    # -- aiming --------------------------------------------------------------------------------------
    def _aim(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is no longer in the world")
            return False
        if not io.reachable((o.x, o.y, io.travel_z)):
            self.fail(io, f"unreachable: {self.obj} at ({o.x:.2f}, {o.y:.2f}) is outside the arm's workspace")
            return False
        self.target, self.size = (o.x, o.y), o.size
        self.yaw = grasp_yaw(o.x, o.y, o.yaw)
        others = [q for oid, q in io.objects().items() if oid != self.obj]
        self.open_w = open_width(o.size, others, self.target, io.max_width, self.p)
        self.heading = self.target
        return True

    def retarget(self, io: ArmIO) -> None:
        """Re-read the target. Before the grasp the approach follows the object; once descending onto
        a spot it has left, go back up and approach again. Carrying or lowering, re-aim the
        destination the same way (a stack's base block can be moved too)."""
        if self.phase in ("approach", "descend"):
            old = self.target
            if self._aim(io) and self.phase == "descend" and old is not None and math.dist(old, self.target) > 0.01:
                self._enter(io, "approach")
        elif self.phase in ("carry", "lower"):
            old = self.dest
            if self._aim_dest(io) and self.phase == "lower" and old is not None and math.dist(old, self.dest) > 0.01:
                self._enter(io, "carry")

    def monitor(self, io: ArmIO) -> None:
        """Runs held or not: a block that leaves the fingers between the grasp and the release is a
        failure wherever we are (a hand-over's wait included)."""
        if self.phase in ("lift", "carry", "lower") or (self.phase == "release" and not self.releasing):
            if io.held() == self.obj:
                self._lost_s = 0.0
            else:
                self._lost_s += self._dt
                if self._lost_s > self.p.drop_grace:
                    self.fail(io, f"dropped: {self.obj} left the fingers {self._when()} at {fmt_xyz(io.ee())}")

    def _when(self) -> str:
        return f"during {self.phase}"

    # -- the phases ----------------------------------------------------------------------------------
    def run(self, io: ArmIO) -> None:
        p, ph = self.p, self.phase
        grasp_z = io.table_z + p.tip_height
        if ph == "approach":
            if self.target is None:
                if io.held() == self.obj:          # already in the gripper: carry on from here
                    x, y, _ = io.ee()
                    self.target = (x, y)
                    self.size = (io.obj(self.obj) or ObjPose(x, y, 0, 0, self.size)).size
                    self.open_w = min(io.max_width, self.size + p.open_margin)
                    self._enter(io, "lift")
                    return
                if not self._aim(io):
                    return
            tx, ty = self.target
            goal = (tx, ty, io.travel_z)
            self.text = f"moving above {self.obj} at travel height"
            io.grip(self.open_w)
            io.goto(goal, p.v_travel, yaw=self.yaw)
            if self._at(io, goal, z_tol=0.01) and abs(wrap(io.ee_yaw() - self.yaw)) < p.yaw_tol:
                self._enter(io, "descend")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress approaching {self.obj}: EE stuck at {fmt_xyz(io.ee())}")
        elif ph == "descend":
            tx, ty = self.target
            goal = (tx, ty, grasp_z)
            self.text = f"descending onto {self.obj}, fingers open {self.open_w * 100:.1f} cm"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - grasp_z < 0.02 else p.v_vertical, yaw=self.yaw)
            if self._at(io, goal):
                self._enter(io, "close")
                self._w_prev, self._w_still = None, 0.0
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress descending onto {self.obj}: fingertips stuck {z - io.table_z:.3f} m above the table")
        elif ph == "close":
            self.text = f"closing the fingers on {self.obj}"
            io.grip(max(0.0, self.size - p.squeeze))
            w = io.width()
            still = self._w_prev is not None and abs(w - self._w_prev) < 2e-4
            self._w_still = self._w_still + self._dt if still else 0.0
            self._w_prev = w
            if self._w_still >= p.settle_s or self.phase_s > p.close_s:
                if io.held() == self.obj:
                    self._lost_s = 0.0
                    self._enter(io, "lift")
                else:
                    self.fail(io, f"grasp_failed: the fingers closed to {w * 100:.1f} cm; {self.obj} is {self.size * 100:.1f} cm wide and not between them")
        elif ph == "lift":
            tx, ty = self.target
            goal = (tx, ty, io.travel_z)
            self.text = f"lifting {self.obj}"
            io.goto(goal, p.v_vertical)
            if self._at(io, goal, z_tol=0.01):
                if self._aim_dest(io):
                    self._enter(io, "carry")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress lifting {self.obj} at {fmt_xyz(io.ee())}")
        elif ph == "carry":
            dx, dy = self.dest
            goal = (dx, dy, io.travel_z)
            self.text = f"carrying {self.obj} to {self.dest_text}"
            io.goto(goal, p.v_travel)
            if self._at(io, goal, z_tol=0.01):
                self._enter(io, "lower")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress carrying {self.obj} at {fmt_xyz(io.ee())}")
        elif ph == "lower":
            lz = self._lower_z(io)
            if lz is None:
                return
            dx, dy = self.dest
            goal = (dx, dy, lz)
            self.text = f"lowering {self.obj} at {self.dest_text}, not yet released"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - lz < 0.02 else p.v_vertical)
            if self._at(io, goal, z_tol=0.006):
                self._enter(io, "release")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress lowering {self.obj} at {self.dest_text}: EE {z - io.table_z:.3f} m above the table")
        elif ph == "release":
            if not self.releasing and not self._may_release(io):
                return
            if not self.releasing:
                self.releasing = True
                self._enter(io, "release")               # the opening gets its own clock after any wait
                x, y, _ = io.ee()
                others = [q for oid, q in io.objects().items() if oid != self.obj]
                self.release_w = open_width(self.size, others, (x, y), io.max_width, p)
            self.text = f"releasing {self.obj} at {self.dest_text}"
            io.grip(self.release_w)
            if io.width() >= self.release_w - 0.002:
                self._enter(io, "retreat")
            elif self.phase_s > p.close_s:
                self.fail(io, f"blocked: the fingers did not open at {self.dest_text}: width {io.width() * 100:.1f} cm")
        elif ph == "retreat":
            dx, dy = self.dest
            goal = (dx, dy, io.travel_z)
            self.text = f"released {self.obj} at {self.dest_text}, rising clear"
            io.goto(goal, p.v_vertical)
            if self._at(io, goal, z_tol=0.01):
                self.done(io, self._done_text())
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress rising clear of {self.dest_text} at {fmt_xyz(io.ee())}")


class PickAndPlace(Grasping):
    """Put the object down at a place: a slot's centre, or a group's first free slot."""
    name = "pick_and_place"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.place = self.dest_text = str(args["place"])

    def _aim_dest(self, io: ArmIO) -> bool:
        d = io.place(self.place)
        if d is None:
            self.fail(io, f"precondition: {self.place} is not a place with room in the world")
            return False
        if not io.reachable((d[0], d[1], io.travel_z)):
            self.fail(io, f"unreachable: {self.place} at ({d[0]:.2f}, {d[1]:.2f}) is outside the arm's workspace")
            return False
        self.dest, self.heading = d, d
        return True

    def _lower_z(self, io: ArmIO) -> float:
        return io.table_z + self.p.tip_height + self.p.place_drop

    def _done_text(self) -> str:
        return f"{self.obj} placed at {self.place}"


class StackOn(Grasping):
    """Put the object down on top of another: carry to above `onto`, lower until the held block
    hangs `place_drop` above its top (read live, so a base that is itself stacked or nudged is
    followed), release, retreat. `onto` with something on it already is `blocked`."""
    name = "stack_on"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.onto = self.dest_text = str(args["onto"])

    def _base(self, io: ArmIO) -> ObjPose | None:
        o = io.obj(self.onto)
        if o is None:
            self.fail(io, f"precondition: {self.onto} is no longer in the world")
            return None
        top = next((oid for oid in io.objects() if oid != self.obj and io.where(oid) == f"on:{self.onto}"), None)
        if top is not None:
            self.fail(io, f"blocked: {self.onto} already has {top} on it")
            return None
        return o

    def _aim_dest(self, io: ArmIO) -> bool:
        o = self._base(io)
        if o is None:
            return False
        if not io.reachable((o.x, o.y, io.travel_z)):
            self.fail(io, f"unreachable: {self.onto} at ({o.x:.2f}, {o.y:.2f}) is outside the arm's workspace")
            return False
        self.dest = self.heading = (o.x, o.y)
        return True

    def _lower_z(self, io: ArmIO) -> float | None:
        o = self._base(io)
        if o is None:
            return None
        return o.z + o.size / 2 + self.p.tip_height + self.p.place_drop     # the held block's bottom is tip_height under the EE

    def _done_text(self) -> str:
        return f"{self.obj} stacked on {self.onto}"


class HandOver(Grasping):
    """Carry the object to the hand-over point (`offer_xy`, `offer_z`: the far edge of the workspace,
    where a person stands), hold it out, and open only once a hand reported `held_out` is beneath
    the fingertips (`hand_beneath`). The wait has its own `timeout_s`; a block taken from the fingers
    before the release reads `dropped`."""
    name = "hand_over"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.dest_text = "the hand-over point"

    def _aim_dest(self, io: ArmIO) -> bool:
        d = self.p.offer_xy
        if not io.reachable((d[0], d[1], self.p.offer_z)):
            self.fail(io, f"unreachable: the hand-over point ({d[0]:.2f}, {d[1]:.2f}) is outside the arm's workspace")
            return False
        self.dest, self.heading = d, d
        return True

    def _lower_z(self, io: ArmIO) -> float:
        return self.p.offer_z

    def _enter(self, io: ArmIO, phase: str) -> None:
        super()._enter(io, phase)
        if phase == "release":
            self.active_s = 0.0          # the motion is over; the wait for a hand gets the full timeout

    def timeout_reason(self) -> str:
        if self.phase == "release":
            return f"timeout: held {self.obj} out for {self.p.timeout_s:.0f} s and no hand came to take it"
        return super().timeout_reason()

    def _when(self) -> str:
        return "while held out for a hand" if self.phase == "release" else super()._when()

    def _may_release(self, io: ArmIO) -> bool:
        h = hand_beneath(io.hands(), io.ee(), self.p)
        if h is not None and h.held_out:
            return True
        x, y, _ = io.ee()
        self.text = (f"holding {self.obj} out at ({x:.2f}, {y:.2f}), waiting for a hand beneath it"
                     if h is None else f"holding {self.obj} out; a hand is beneath it but not held out to take it")
        return False

    def _done_text(self) -> str:
        return f"{self.obj} released into the hand"


class Push(Skill):
    """Slide the object along the table with the closed gripper: approach behind it at travel
    height -> descend to grasp height -> slide until the object has travelled `distance` ->
    retreat. The slide watches the object, not the EE: it stops when the object has gone far enough,
    reads `blocked` when the fingers fall `push_lag_m` behind where the slide should have them
    (something in the way; before the safety filter's lag trip), and `stalled` when the fingers
    reach their limit with the object left behind."""
    name = "push"
    first_phase = "approach"
    PHASES = ("approach", "descend", "slide", "retreat")

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.obj = str(args["object"])
        self.direction = str(args["direction"])
        self.distance = float(args["distance"])
        self.d = DIRECTIONS[self.direction]
        self.start: tuple[float, float] | None = None
        self.limit: tuple[float, float] | None = None
        self.origin: tuple[float, float] | None = None   # the object when the slide began
        self.moved = 0.0                                  # its progress along the direction, m
        self.yaw = 0.0

    def _progress(self, o: ObjPose) -> float:
        return (o.x - self.origin[0]) * self.d[0] + (o.y - self.origin[1]) * self.d[1]

    def _aim(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is no longer in the world")
            return False
        start, limit = push_path(o, self.direction, self.distance, self.p)
        push_z = io.table_z + self.p.tip_height
        for xyz, what in ((start + (io.travel_z,), "starts"), (start + (push_z,), "starts"), (limit + (push_z,), "ends")):
            if not io.reachable(xyz):
                self.fail(io, f"unreachable: pushing {self.obj} {self.distance * 100:.0f} cm {self.direction} {what} "
                              f"at ({xyz[0]:.2f}, {xyz[1]:.2f}), outside the arm's workspace")
                return False
        self.start, self.limit = start, limit
        self.yaw = grasp_yaw(o.x, o.y, math.atan2(self.d[1], self.d[0]))
        self.heading = (o.x + self.distance * self.d[0], o.y + self.distance * self.d[1])
        return True

    def retarget(self, io: ArmIO) -> None:
        """Before the slide, re-aim at where the object is now (a moved object means a new start)."""
        if self.phase in ("approach", "descend"):
            old = self.start
            if self._aim(io) and self.phase == "descend" and old is not None and math.dist(old, self.start) > 0.01:
                self._enter(io, "approach")

    def run(self, io: ArmIO) -> None:
        p, ph = self.p, self.phase
        push_z = io.table_z + p.tip_height
        if ph == "approach":
            if self.start is None and not self._aim(io):
                return
            goal = (self.start[0], self.start[1], io.travel_z)
            self.text = f"moving behind {self.obj} at travel height, fingers closed, to push it {self.direction}"
            io.grip(0.0)
            io.goto(goal, p.v_travel, yaw=self.yaw)
            if self._at(io, goal, z_tol=0.01) and abs(wrap(io.ee_yaw() - self.yaw)) < p.yaw_tol:
                self._enter(io, "descend")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress approaching {self.obj}: EE stuck at {fmt_xyz(io.ee())}")
        elif ph == "descend":
            goal = (self.start[0], self.start[1], push_z)
            self.text = f"descending behind {self.obj}, fingers closed"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - push_z < 0.02 else p.v_vertical, yaw=self.yaw)
            if self._at(io, goal):
                o = io.obj(self.obj)
                if o is None:
                    self.fail(io, f"precondition: {self.obj} is no longer in the world")
                    return
                self.origin, self.moved = (o.x, o.y), 0.0
                self._enter(io, "slide")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress descending behind {self.obj}: fingertips stuck {z - io.table_z:.3f} m above the table")
        elif ph == "slide":
            o = io.obj(self.obj)
            if o is None:
                self.fail(io, f"precondition: {self.obj} is no longer in the world")
                return
            self.moved = self._progress(o)
            cm, of = self.moved * 100, self.distance * 100
            self.text = f"pushing {self.obj} {self.direction}: {cm:.1f} of {of:.1f} cm"
            if self.moved >= self.distance - p.push_coast:
                x, y, z = io.ee()
                io.goto((x, y, z), p.v_push)          # stop where the fingers are
                self._enter(io, "retreat")
                return
            goal = (self.limit[0], self.limit[1], push_z)
            io.goto(goal, p.v_push)
            x, y, _ = io.ee()
            run = (self.limit[0] - self.start[0]) * self.d[0] + (self.limit[1] - self.start[1]) * self.d[1]
            due = min(p.v_push * self.phase_s, run)            # where the slide's setpoint is by now
            got = (x - self.start[0]) * self.d[0] + (y - self.start[1]) * self.d[1]
            if self._at(io, goal):
                self.fail(io, f"stalled: {self.obj} moved only {cm:.1f} of {of:.1f} cm {self.direction} and the fingers "
                              f"are at the end of their run at {fmt_xyz(io.ee())}: it slipped off them")
            elif due - got > p.push_lag_m:
                self.fail(io, f"blocked: {self.obj} stopped after {cm:.1f} of {of:.1f} cm {self.direction}; "
                              f"something is in the way at {fmt_xyz(io.ee())}")
        elif ph == "retreat":
            goal = (self.heading[0], self.heading[1], io.travel_z)
            self.text = f"pushed {self.obj} {self.moved * 100:.1f} cm {self.direction}, rising clear"
            io.goto(goal, p.v_vertical)
            if self._at(io, goal, z_tol=0.01):
                self.done(io, f"{self.obj} pushed {self.moved * 100:.1f} cm {self.direction}")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress rising clear of {self.obj} at {fmt_xyz(io.ee())}")

    def _enter(self, io: ArmIO, phase: str) -> None:
        super()._enter(io, phase)
        if phase == "retreat":
            x, y, _ = io.ee()
            self.heading = (x, y)          # rise where the fingers stopped


class HoldStill(Skill):
    """The planned skill `hold`: stay where the arm is for a while (default 1 s), grip unchanged."""
    name = "hold"
    first_phase = "still"

    def run(self, io: ArmIO) -> None:
        secs = float(self.args.get("seconds", 1.0))
        self.text = f"holding still ({self.phase_s:.1f} of {secs:.1f} s)"
        if self.phase_s >= secs:
            self.done(io, f"held still for {secs:.1f} s")


class Survey(Skill):
    """Rise straight up to travel height, keeping xy and the grip, so the cameras see the table."""
    name = "survey"
    first_phase = "rise"

    def run(self, io: ArmIO) -> None:
        x, y, _ = io.ee()
        if self.heading is None:
            self.heading = (x, y)
        goal = (self.heading[0], self.heading[1], io.travel_z)
        self.text = "rising to travel height to look over the table"
        io.goto(goal, self.p.v_vertical)
        if self._at(io, goal, z_tol=0.01):
            self.done(io, "at travel height, clear of the table")
        elif self._stalled(io):
            self.fail(io, f"stalled: no progress rising at {fmt_xyz(io.ee())}")


SKILLS: dict[str, type[Skill]] = {"pick_and_place": PickAndPlace, "stack_on": StackOn, "push": Push,
                                  "hand_over": HandOver, "hold": HoldStill, "survey": Survey}
