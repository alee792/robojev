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
which is where the wrist roll is zero, so the roll stays within +-45 deg; the carry re-aims it the
same way for the destination (K5 found the gripper housing meeting the upper arm when a block was
set down 11 cm from the base with the pick's yaw kept, 60 deg of roll).

The pre-grasp opening shrinks for neighbours (the L1 follow-up): the pads' outer faces sit
`finger_t` outside the fingertips, so next to another block the jaws open only as wide as keeps a
pad 5 mm clear of it, never less than the block plus 8 mm. A 6 cm slot pitch picks (tested).

The push (K4): the closed gripper's pad boxes stick out past the fingertips on every side (13.6 mm
along the closing axis, 13.7 / 19.7 mm across it), so a block is pushed by the pads, which start
1.35 cm above the fingertips. The fingertips go to grasp height, which puts the push resultant
~2.9 cm up a 4 cm cube; it slides without tipping only while the block-table friction is under
s / 2h = 0.7, which is why scene.py gives that pair 0.5 (at 1.0 the cube rolled over). The slide is
closed-loop on the object: the fingers advance until the object has travelled the distance.

Grasp by shape (the tool scene): the grasp is chosen from `ObjPose.shape`, not by a skill. A cube
keeps the fingertip pinch above: four candidate closing-axis yaws, one per face. A rod (a handle-
like tool) is pinched across its handle at the handle's midpoint (`ObjPose.handle` from the centre
along the front): two candidates, the closing axis perpendicular to the rod either way round, the
jaws opened to the handle's width plus the margin. Measured on the model (2026-10-14): the fingertip
pieces are 14.6 mm tall and 15.4 mm along the rod and start at the EE point, so with the tips 5 mm
over the table a 12 mm rod's 14 mm handle is pinched over its upper 7 mm, 1.6 N a finger at the 8 mm
squeeze; over a 0.15 m/s carry with up to 170 deg of wrist roll it shifts under 4 mm and 2.2 deg in
the fingers. The same pinch lets the rod slide up when it is pressed onto a wall (see Insert).

The wrist roll (insert, reorient): the roll is the closing axis's angle from the direction
perpendicular to the arm's azimuth, where joint_5 is at zero. The model's joint_5 turns +-180 deg
and the IK follows it anywhere in the one-arm workspace with no self-contact (swept 2026-10-14), so
`Params.roll_max` (120 deg) is a margin kept from the joint stop, not a measured collision; the
real arms must confirm their own. Yaw is commanded as a roll held constant while the arm moves
(`Grasping._steer`), stepping at most a radian from where the wrist is, because the backend's yaw
rate limiter takes the short way round and a turn of more than half a turn would otherwise end
past the stop. A turn the roll range cannot make in one grasp is split (`turn_plan`): put the object
down turned as far as the range allows, regrasp with the other candidate, finish. With a range of
at least 90 deg a rod needs at most one regrasp and a cube at most one too (its candidates are 90
deg apart); the loop is general.

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
    yaw: float            # rad about +z: the direction the front points (a cube's marked face, a rod's handle)
    size: float           # the width the fingers close to: a cube's edge, a rod's handle, m
    shape: str = "cube"   # "cube" | "rod": picks the grasp (grasp_candidates, grasp_point)
    length: float = 0.0   # along the front: a rod's length; 0 = size
    handle: float = 0.0   # the pinch point, from the centre along the front: a rod's handle midpoint, 0 for a cube

    @property
    def extent(self) -> float:
        """how far from its centre the object reaches, at most"""
        return max(self.size, self.length) / 2


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
    def place(self, pid: str) -> tuple[float, ...] | None:
        """where to put something down for this place id (a group resolves to a free member): (x, y),
        or (x, y, yaw) for a holder, the yaw being the direction a seated object's front points"""
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
    roll_max: float = math.radians(120)   # the wrist roll either way from zero a grasp may use (see the module docstring)
    v_insert: float = 0.03        # m/s pressing a rod into a holder
    insert_above: float = 0.03    # the slow press starts this far above the seat: the rod's bottom 1 cm over 15 mm walls
    insert_lag_m: float = 0.010   # fingers this far behind the press's setpoint = the rod is on a wall (blocked); a free
                                  # descent at v_insert lags ~3 mm, the safety filter trips at 30
    seat_tol: float = 0.004       # EE within this of the seat height, setpoint there = seated
    settle_insert_s: float = 0.3  # s for the EE to close on an arrived setpoint before the height is judged
    front_tol: float = 0.15       # rad: a reoriented object's front within this of the asked direction is done (the
                                  # conformance judge allows 15 deg)
    max_rounds: int = 3           # grasps a turn may take before it reads stalled (two suffice at roll_max >= 90 deg)


# Advertised as-is from the standard catalog: shared names are what make recordings transfer.
SPECS = (catalog.PICK_AND_PLACE, catalog.STACK_ON, catalog.PUSH, catalog.HAND_OVER, catalog.INSERT, catalog.REORIENT,
         catalog.HOLD, catalog.SURVEY)

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


def neutral_yaw(x: float, y: float) -> float:
    """The closing-axis yaw with zero wrist roll at (x, y): perpendicular to the arm's azimuth."""
    return math.atan2(y, x) + math.pi / 2


def roll_of(yaw: float, x: float, y: float) -> float:
    """The wrist roll a closing-axis yaw means at (x, y), in (-pi, pi]."""
    return wrap(yaw - neutral_yaw(x, y))


def front_of(direction: str) -> float:
    """The yaw an object's front points when it points `direction` (catalog.DIRECTIONS, base frame)."""
    dx, dy = DIRECTIONS[direction]
    return math.atan2(dy, dx)


def grasp_relations(shape: str) -> tuple[float, ...]:
    """The angles, from one grasp candidate to the next, a shape's front can sit at relative to the
    closing axis: a cube's faces are 90 deg apart, a rod is pinched across its axis either way round."""
    return (0.0, math.pi) if shape == "rod" else (0.0, math.pi / 2, math.pi, -math.pi / 2)


def grasp_candidates(o: ObjPose) -> list[float]:
    """Closing-axis yaws that meet the object squarely: a cube's four (one per face), a rod's two
    (across its handle, either way round)."""
    base = o.yaw + (math.pi / 2 if o.shape == "rod" else 0.0)
    return [base + k for k in grasp_relations(o.shape)]


def grasp_point(o: ObjPose) -> tuple[float, float]:
    """Where the fingertips close: the centre, or `handle` along the front for a rod."""
    return (o.x + o.handle * math.cos(o.yaw), o.y + o.handle * math.sin(o.yaw))


@dataclass(frozen=True)
class TurnPlan:
    """How to get a held object's front to `F` at the destination: `direct` with the wrist at `roll`
    there, or (not direct) an intermediate put-down at the pick spot with the wrist at `roll`, turned
    as far toward the goal as the range allows, after which a regrasp reaches the destination."""
    direct: bool
    roll: float


def turn_plan(shape: str, delta: float, F: float, dest: tuple[float, float], here: tuple[float, float],
              r_now: float, roll_max: float) -> TurnPlan:
    """`delta` is the front's angle from the closing axis as held (fixed by the grasp), `F` the front
    wanted at `dest` (EE xy), `here` the EE xy for a put-down at the pick spot, `r_now` the wrist's
    roll at the pick. Direct when the destination roll is within the range. Otherwise choose the
    grasp relation to regrasp with (the one that makes the destination roll smallest), and turn now
    so that the regrasp's own roll is as near zero as the range allows: with a range of 90 deg or
    more both rolls of the next round are then within it."""
    r1 = roll_of(F - delta, *dest)
    if abs(r1) <= roll_max:
        return TurnPlan(True, r1)
    e = min((k for k in grasp_relations(shape) if k != 0.0), key=lambda k: abs(roll_of(F - wrap(delta - k), *dest)))
    want = wrap(-e)
    if abs(abs(want) - math.pi) < 1e-9:              # half a turn either way: the way that turns less
        want = math.copysign(math.pi, r_now if r_now != 0.0 else 1.0)
    return TurnPlan(False, max(-roll_max, min(roll_max, want)))


def open_width(size: float, others: list[ObjPose], at: tuple[float, float], max_width: float, p: Params) -> float:
    """Pre-grasp opening for a block of `size` at `at`: width + open_margin, shrunk so a pad's outer
    face stays `neighbour_gap` clear of the nearest other block (taken as lying along the closing
    axis, the worst case; a rod's reach is its half-length), never under width + open_min, never over
    the jaw max."""
    w = size + p.open_margin
    for o in others:
        face = math.hypot(o.x - at[0], o.y - at[1]) - o.extent           # that object's near face from our centre
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
        # What keeps a holding skill still, by holder: "brain" (hold()), "trip" (the safety filter;
        # the value is its text), "zone" (the shared-zone lock; the value says why: "held by arm_1").
        # Running again needs every holder gone: resume() clears the brain's and the filter's, only
        # the lock clears its own.
        self.holds: dict[str, str] = {}
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

    @property
    def tripped(self) -> bool:
        """held by the safety filter, not by the brain"""
        return "trip" in self.holds

    @property
    def trip_text(self) -> str:
        return self.holds.get("trip", "")

    @property
    def waiting(self) -> str | None:
        """why this skill is waiting for the shared zone ("held by arm_1"), else None"""
        return self.holds.get("zone")

    def _still(self, io: ArmIO, holder: str, why: str) -> None:
        """Record a holder; freeze the arm the first time that holder appears (the lock may move a
        waiting arm back to its side line, so a repeated wait() must not re-freeze it)."""
        if not self.finished:
            if holder not in self.holds:
                io.freeze()
            self.holds[holder] = why
            if self.state == "running":
                self.state = "holding"

    def hold(self, io: ArmIO) -> None:
        if self.state in ("running", "holding"):
            self._still(io, "brain", "held")

    def pause(self, io: ArmIO) -> None:
        if self.state in ("running", "holding"):
            io.freeze()
            self.state = "paused"

    def trip(self, io: ArmIO, text: str) -> None:
        """The safety filter tripped: hold and remember why; resume() is the only way out. A paused
        skill stays paused (the brain's request outranks the filter's)."""
        self._still(io, "trip", text)

    def wait(self, io: ArmIO, why: str) -> None:
        """The shared-zone lock: stay out of the zone until `proceed()`; `why` is literal ("held by
        arm_1"). A paused or held skill remembers the wait, so the brain's resume() cannot take it in."""
        self._still(io, "zone", why)

    def proceed(self, io: ArmIO) -> None:
        """The lock is this arm's: go on, unless the brain or the filter also holds the skill."""
        if self.holds.pop("zone", None) is not None and self.state == "holding" and not self.holds:
            self._go(io)

    def resume(self, io: ArmIO) -> None:
        """Clears the brain's hold, a pause and a trip; a skill still waiting for the shared zone
        stays holding, and runs the moment the lock is granted."""
        if self.state in ("holding", "paused"):
            self.holds.pop("brain", None)
            self.holds.pop("trip", None)
            if self.holds:
                self.state = "holding"
            else:
                self._go(io)

    def _go(self, io: ArmIO) -> None:
        io.unfreeze()
        self.state = "running"
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
            if self.tripped:
                text = f"safety trip ({self.trip_text}), held: " + text
            elif self.waiting is not None:
                text = f"waiting for the shared zone, {self.waiting}: " + text
            else:
                text = "held: " + text
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
    Subclasses say where to carry to (`_aim_dest`, which may also set `dest_roll`, the wrist roll to
    arrive with), how low to go (`_lower_z`), what follows the lowering (`_after_lower`, an extra
    phase run by `_run_extra`), whether the fingers may open yet (`_may_release`), whether the skill
    is finished once clear (`_after_retreat`: a turning skill starts another round) and the words
    (`dest_text`, `_done_text`). The grasp itself follows the object's shape (grasp_candidates).

    Wrist yaw is commanded through `_steer` as a roll held constant while the arm moves; the roll a
    phase wants is `self.roll` (the pick's) or `dest_roll` (the destination's); without a `dest_roll`
    the carry re-aims within the pick's 90-deg family, as K5 tuned it."""
    first_phase = "approach"
    PHASES = ("approach", "descend", "close", "lift", "carry", "lower", "release", "retreat")
    HOLDING_PHASES = ("lift", "carry", "lower")     # ... plus `release` before the fingers open
    YAW_STEP = 1.0                                  # rad: the most a yaw command leads the wrist by

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.obj = str(args["object"])
        self.target: tuple[float, float] | None = None   # where the fingertips close (the grasp point) when aimed at
        self.size = 0.04
        self.yaw = 0.0                                   # the pick's closing-axis yaw
        self.roll = 0.0                                  # ... as a wrist roll at the target
        self.open_w = 0.0                                # pre-grasp opening, chosen when aimed
        self.release_w = 0.0                             # opening at the destination, chosen on arrival
        self.dest: tuple[float, float] | None = None
        self.dest_roll: float | None = None              # the wrist roll to arrive at `dest` with; None = the pick's family
        self.dest_text = ""                              # "slot_2", "block_2", "the hand-over point"
        self.delta: float | None = None                  # the front's angle from the closing axis as held (set at the grasp)
        self.hold_dz = 0.0                               # the object's centre above the EE point as held
        self.pick_centre: tuple[float, float] | None = None   # the object's centre when grasped: where a turn puts it down
        self.releasing = False                           # the fingers have been told to open
        self._w_prev: float | None = None
        self._w_still = 0.0
        self._lost_s = 0.0

    # -- what a subclass decides ---------------------------------------------------------------------
    def _aim_dest(self, io: ArmIO) -> bool:
        """Set `dest` (xy), `dest_text` and `heading` (and `dest_roll` if a roll is required), or fail
        and return False."""
        raise NotImplementedError

    def _lower_z(self, io: ArmIO) -> float | None:
        """EE height at which to let go, read live; None once `fail` has been called."""
        raise NotImplementedError

    def _after_lower(self) -> str:
        """The phase after the lowering has arrived: release, or a subclass's own (insert)."""
        return "release"

    def _run_extra(self, io: ArmIO, phase: str) -> None:
        """A phase of the subclass's own."""
        raise NotImplementedError(phase)

    def _may_release(self, io: ArmIO) -> bool:
        """True when the fingers may open; a subclass that waits sets `text` meanwhile."""
        return True

    def _after_retreat(self, io: ArmIO) -> bool:
        """True when the skill is finished once clear of the destination; a turning skill that has
        more to do starts its next round here and returns False."""
        return True

    def _pick_yaw(self, io: ArmIO, o: ObjPose, at: tuple[float, float]) -> float:
        """The closing-axis yaw to grasp with: the candidate nearest zero roll (a subclass with a
        destination in mind chooses with that in view)."""
        return min(grasp_candidates(o), key=lambda c: abs(roll_of(c, *at)))

    def _done_text(self) -> str:
        raise NotImplementedError

    # -- aiming --------------------------------------------------------------------------------------
    def _aim(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is no longer in the world")
            return False
        at = grasp_point(o)
        if not io.reachable((at[0], at[1], io.travel_z)):
            self.fail(io, f"unreachable: {self.obj} at ({o.x:.2f}, {o.y:.2f}) is outside the arm's workspace")
            return False
        self.target, self.size = at, o.size
        self.yaw = self._pick_yaw(io, o, at)
        self.roll = roll_of(self.yaw, *at)
        others = [q for oid, q in io.objects().items() if oid != self.obj]
        self.open_w = open_width(o.size, others, at, io.max_width, self.p)
        self.heading = at
        return True

    def _grasped(self, io: ArmIO) -> None:
        """Remember how the object sits in the fingers: its front's angle from the closing axis, and
        its centre (the turn spot). Read from the world, so a slipped grasp is still described."""
        o = io.obj(self.obj)
        if o is not None:
            self.delta = wrap(o.yaw - io.ee_yaw())
            self.hold_dz = o.z - io.ee()[2]
            self.pick_centre = (o.x, o.y)
            self.size = o.size

    def _steer(self, io: ArmIO, roll: float, at: tuple[float, float]) -> tuple[float, bool]:
        """The yaw to command now so the wrist arrives at `at` with `roll`, leading the wrist by at
        most YAW_STEP along the turn still to make (the rate limiter takes the short way round; a
        turn of more than half a turn must be led the right way), and whether it is there. The
        command is anchored to the goal's azimuth, not the EE's, so it holds still as the EE settles
        and the backend's `arrived()` can read true."""
        x, y, _ = io.ee()
        end = neutral_yaw(*at) + roll
        d = end - (neutral_yaw(x, y) + roll_of(io.ee_yaw(), x, y))   # the physical turn left, as a real number
        return io.ee_yaw() + max(-self.YAW_STEP, min(self.YAW_STEP, d)), abs(wrap(io.ee_yaw() - end)) < self.p.yaw_tol

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
        if self.phase in self.HOLDING_PHASES or (self.phase == "release" and not self.releasing):
            if io.held() == self.obj:
                self._lost_s = 0.0
            else:
                self._lost_s += self._dt
                if self._lost_s > self.p.drop_grace:
                    self.fail(io, f"dropped: {self.obj} left the fingers {self._when()} at {fmt_xyz(io.ee())}")

    def _when(self) -> str:
        return f"during {self.phase}"

    def _new_round(self, io: ArmIO) -> None:
        """Start over from the approach (a turning skill's regrasp): a fresh aim, a fresh timeout."""
        self.target = self.dest = self.dest_roll = None
        self.releasing = False
        self.active_s = 0.0
        self._enter(io, "approach")

    # -- the phases ----------------------------------------------------------------------------------
    def run(self, io: ArmIO) -> None:
        p, ph = self.p, self.phase
        grasp_z = io.table_z + p.tip_height
        if ph == "approach":
            if self.target is None:
                if io.held() == self.obj:          # already in the gripper: carry on from here
                    x, y, _ = io.ee()
                    self.target = (x, y)
                    self._grasped(io)
                    self.open_w = min(io.max_width, self.size + p.open_margin)
                    self._enter(io, "lift")
                    return
                if not self._aim(io):
                    return
            tx, ty = self.target
            goal = (tx, ty, io.travel_z)
            self.text = f"moving above {self.obj} at travel height"
            io.grip(self.open_w)
            yaw, there = self._steer(io, self.roll, self.target)
            io.goto(goal, p.v_travel, yaw=yaw)
            if self._at(io, goal, z_tol=0.01) and there:
                self._enter(io, "descend")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress approaching {self.obj}: EE stuck at {fmt_xyz(io.ee())}")
        elif ph == "descend":
            tx, ty = self.target
            goal = (tx, ty, grasp_z)
            self.text = f"descending onto {self.obj}, fingers open {self.open_w * 100:.1f} cm"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - grasp_z < 0.02 else p.v_vertical, yaw=self._steer(io, self.roll, self.target)[0])
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
                    self._grasped(io)
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
            # Without a required roll: the pick's 90-deg family, re-aimed so the roll is near zero at the
            # destination too (near the base the 19 cm gripper housing meets the upper arm at 60 deg of roll).
            roll = self.dest_roll if self.dest_roll is not None else roll_of(grasp_yaw(dx, dy, self.yaw), dx, dy)
            self.text = f"carrying {self.obj} to {self.dest_text}"
            yaw, there = self._steer(io, roll, self.dest)
            io.goto(goal, p.v_travel, yaw=yaw)
            if self._at(io, goal, z_tol=0.01) and there:
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
                self._enter(io, self._after_lower())
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
                if self._after_retreat(io):
                    self.done(io, self._done_text())
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress rising clear of {self.dest_text} at {fmt_xyz(io.ee())}")
        else:
            self._run_extra(io, ph)


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
        if len(d) > 2:
            self.fail(io, f"precondition: {self.place} is a holder: insert seats things there")
            return False
        o = io.obj(self.obj)
        if o is not None and o.handle and self.delta is not None:
            # The object's centre goes on the place; the fingers hold it `handle` along its front, which at
            # the destination is the carry's yaw plus the grasp's offset. The yaw depends on the azimuth of
            # where the fingers end up, so a few fixed-point steps (a few degrees of correction at most).
            ee = d
            for _ in range(4):
                front = grasp_yaw(ee[0], ee[1], self.yaw) + self.delta
                ee = (d[0] + o.handle * math.cos(front), d[1] + o.handle * math.sin(front))
            d = ee
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


class Turning(Grasping):
    """What insert and reorient share: the object must end with its front pointing a given way
    (`_goal`: the front and the EE xy at the destination), the pick is chosen with that in view, and
    when the wrist's roll range cannot get there in one grasp the object is put down at its pick spot
    turned as far as the range allows and regrasped (turn_plan), round after round, each round a
    full approach-to-retreat with its own timeout."""
    turning = False                  # this round is an intermediate put-down, not the destination

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.rounds = 0

    def _goal(self, io: ArmIO, o: ObjPose) -> tuple[float, tuple[float, float]] | None:
        """(the front's yaw wanted, the EE xy at the destination), or None after a fail."""
        raise NotImplementedError

    def _pick_yaw(self, io: ArmIO, o: ObjPose, at: tuple[float, float]) -> float:
        """Among the candidates within the roll range, one that reaches the destination in this
        grasp if any can (the least roll overall), else the one whose turn-and-regrasp rolls least."""
        g = self._goal(io, o)
        cands = [c for c in grasp_candidates(o) if abs(roll_of(c, *at)) <= self.p.roll_max]
        if g is None or not cands:
            return super()._pick_yaw(io, o, at)
        F, dest = g

        def cost(c: float) -> tuple[int, float]:
            r0 = roll_of(c, *at)
            plan = turn_plan(o.shape, wrap(o.yaw - c), F, dest, (o.x, o.y), r0, self.p.roll_max)
            return (0 if plan.direct else 1), abs(r0) + abs(plan.roll - r0)
        return min(cands, key=cost)

    def _put_down_xy(self, o: ObjPose, roll: float) -> tuple[float, float]:
        """The EE xy that puts the object's centre back on its pick spot with the wrist at `roll`
        there: the pinch is `handle` along the front, and the front depends on the azimuth, so a
        few fixed-point steps (the correction is a few degrees at most)."""
        cx, cy = self.pick_centre
        ee = (cx, cy)
        for _ in range(4):
            front = neutral_yaw(*ee) + roll + self.delta
            ee = (cx + o.handle * math.cos(front), cy + o.handle * math.sin(front))
        return ee

    def _aim_dest(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is no longer in the world")
            return False
        g = self._goal(io, o)
        if g is None:
            return False
        F, dest = g
        x, y, _ = io.ee()
        plan = turn_plan(o.shape, self.delta, F, dest, self.pick_centre, roll_of(io.ee_yaw(), x, y), self.p.roll_max)
        if plan.direct:
            self.turning = False
            self.dest_text = self._dest_name()
        else:
            self.turning = True
            dest = self._put_down_xy(o, plan.roll)
            self.dest_text = "its spot, turned as far as the wrist goes, to regrasp"
        if not io.reachable((dest[0], dest[1], io.travel_z)):
            self.fail(io, f"unreachable: {self.dest_text} at ({dest[0]:.2f}, {dest[1]:.2f}) is outside the arm's workspace")
            return False
        self.dest, self.dest_roll, self.heading = dest, plan.roll, dest
        return True

    def _dest_name(self) -> str:
        raise NotImplementedError

    def _lower_z(self, io: ArmIO) -> float:
        return io.table_z + self.p.tip_height + self.p.place_drop

    def _after_retreat(self, io: ArmIO) -> bool:
        if not self.turning and self._finished(io):
            return True
        self.rounds += 1
        if self.rounds >= self.p.max_rounds:
            self.fail(io, f"stalled: {self.obj} is not where it should point after {self.rounds} grasps ({self.text})")
            return False
        self._new_round(io)
        return False

    def _finished(self, io: ArmIO) -> bool:
        """The destination round is over: is the object as asked? (Insert trusts its seating.)"""
        return True


class Insert(Turning):
    """Seat a rod in a holder: pick it by its handle, carry it over the holder with the wrist turned
    so the rod lies along the opening, lower to just above the walls, then press down slowly
    (`insert` phase) watching the fingers against the setpoint and the rod against the fingers:
    seated when the EE reaches the seat height with the setpoint there and the rod still riding where
    it was grasped; `blocked` when the fingers fall `insert_lag_m` behind the setpoint (the rod is on
    a wall and the arm with it), when they stop short of the seat, or when the rod rises in the
    fingers by more than `seat_tol` (it is on a wall and the press is pushing it up through the pinch:
    measured, a 4 N pinch lets a 12 mm rod slide before the arm's lag shows). Then release and rise.
    A holder taken meanwhile, or no free holder in a group, is `blocked`; the preconditions refuse a
    cube or an occupied holder."""
    name = "insert"
    PHASES = ("approach", "descend", "close", "lift", "carry", "lower", "insert", "release", "retreat")
    HOLDING_PHASES = ("lift", "carry", "lower", "insert")

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.place = str(args["place"])
        self.dest_text = self.place
        self._z0: float | None = None

    def _goal(self, io: ArmIO, o: ObjPose):
        d = io.place(self.place)
        if d is None:
            self.fail(io, f"blocked: {self.place} has no free holder")
            return None
        if len(d) < 3:
            self.fail(io, f"precondition: {self.place} is not a holder")
            return None
        taken = next((oid for oid in io.objects() if oid != self.obj and io.where(oid) == self.place), None)
        if taken is not None:
            self.fail(io, f"blocked: {self.place} is occupied by {taken}")
            return None
        x, y, F = d
        return F, (x + o.handle * math.cos(F), y + o.handle * math.sin(F))

    def _dest_name(self) -> str:
        return self.place

    def _lower_z(self, io: ArmIO) -> float:
        if self.turning:
            return super()._lower_z(io)
        return io.table_z + self.p.tip_height + self.p.insert_above

    def _after_lower(self) -> str:
        return "release" if self.turning else "insert"

    def _enter(self, io: ArmIO, phase: str) -> None:
        super()._enter(io, phase)
        if phase == "insert":
            self._z0 = io.ee()[2]
            self._arrived_s = 0.0

    def _run_extra(self, io: ArmIO, phase: str) -> None:
        p = self.p
        dx, dy = self.dest
        seat = io.table_z + p.tip_height
        goal = (dx, dy, seat)
        z = io.ee()[2]
        self.text = f"pressing {self.obj} into {self.place}, {max(0.0, z - seat) * 1000:.0f} mm above the seat"
        io.goto(goal, p.v_insert)
        due = min(p.v_insert * self.phase_s, self._z0 - seat)       # where the press's setpoint is by now
        got = self._z0 - z
        o = io.obj(self.obj)
        risen = (o.z - z - self.hold_dz) if o is not None else 0.0  # the rod climbing in the fingers: it is on a wall
        above = (z - seat) * 1000
        if risen > p.seat_tol:
            self.fail(io, f"blocked: {self.obj} is riding up in the fingers ({risen * 1000:.0f} mm) {above:.0f} mm above the "
                          f"seat of {self.place}: it is resting on a wall, not in the opening")
        elif io.arrived():
            self._arrived_s += self._dt
            if abs(z - seat) <= p.seat_tol:
                self._enter(io, "release")
                return
            if self._arrived_s > p.settle_insert_s:
                self.fail(io, f"blocked: {self.obj} stopped {above:.0f} mm above the seat of {self.place}: it is not in the opening")
        elif due - got > p.insert_lag_m:
            self.fail(io, f"blocked: {self.obj} stopped {above:.0f} mm above the seat of {self.place}: "
                          f"it is resting on a wall, not in the opening")

    def _done_text(self) -> str:
        return f"{self.obj} seated in {self.place}"


class Reorient(Turning):
    """Turn the object in place so its front points `direction`: pick it, turn the wrist while
    carrying it the few centimetres that keep its centre over its spot (a rod is held off-centre),
    put it down, release, rise; regrasp and go on if the roll range ran out. Done when the front
    reads within `front_tol` of the direction, else another round, up to `max_rounds`."""
    name = "reorient"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        super().__init__(sid, arm, args, p)
        self.direction = str(args["direction"])
        self.F = front_of(self.direction)
        self.origin: tuple[float, float] | None = None   # the object's centre at the start: where it ends
        self.dest_text = f"its spot, pointing {self.direction}"

    def _goal(self, io: ArmIO, o: ObjPose):
        if self.origin is None:
            self.origin = (o.x, o.y)
        return self.F, (self.origin[0] + o.handle * math.cos(self.F), self.origin[1] + o.handle * math.sin(self.F))

    def _dest_name(self) -> str:
        return f"its spot, pointing {self.direction}"

    def _finished(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is no longer in the world")
            return False
        off = wrap(o.yaw - self.F)
        self.text = f"{self.obj} points {math.degrees(off):+.0f} deg off {self.direction}"
        return abs(off) <= self.p.front_tol

    def _done_text(self) -> str:
        return f"{self.obj} turned to point {self.direction}"


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
                                  "hand_over": HandOver, "insert": Insert, "reorient": Reorient,
                                  "hold": HoldStill, "survey": Survey}
