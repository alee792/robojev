"""Skill policies: run by the robot server at the policy rate (50 Hz), one tick = read the arm and the
world, set the next goal. They never touch MuJoCo; everything goes through the `ArmIO` surface the
server hands them, so the same policies can sit on the real backend at D1.

The grasp (docs/harness-spikes.md, "Grasp geometry"): the pads' gripping faces span 1.2-7 cm above
the fingertips with the gripper pointing straight down, so the tips go to ~0.5 cm above the table
and the pads straddle a 4 cm cube from 1.7 cm up. Pitch is always 90 deg; only yaw is chosen,
aligned with the block's faces (mod 90 deg) and as close to the arm's azimuth as that allows, which
keeps the wrist roll within +-45 deg.

Every phase has literal text the decision loop can read. hold()/pause() work from any phase: the
setpoint freezes where it is and the gripper command is left alone, so a held grip stays a grip;
resume() continues the same phase because each phase re-issues its goal every tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from . import catalog
from .protocol import SkillStatus

ARM_IO_DOC = """What a skill may read and command. Implemented by the server; positions in the base
frame, metres, seconds; `t` is the server's clock (sim time here)."""


@dataclass
class ObjPose:
    x: float
    y: float
    z: float
    yaw: float            # rad about +z
    size: float           # edge, m


class ArmIO(Protocol):
    t: float
    table_z: float
    travel_z: float
    max_width: float
    frozen: bool

    def ee(self) -> tuple[float, float, float]: ...
    def ee_yaw(self) -> float: ...
    def arrived(self) -> bool:
        """the setpoint has reached the goal (the EE may still lag a little)"""
    def width(self) -> float: ...
    def obj(self, oid: str) -> ObjPose | None: ...
    def place(self, pid: str) -> tuple[float, float] | None: ...
    def held(self) -> str | None:
        """object id between the closed pads, from physics (both pads touching, width within tolerance)"""
    def goto(self, xyz, speed: float, yaw: float | None = None) -> None: ...
    def grip(self, width: float) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...


@dataclass
class Params:
    """Grasp and motion numbers in one place. Tuned on the K2 trials; see docs/harness-spikes.md."""
    tip_height: float = 0.005     # fingertips above the table at grasp: pads on a 4 cm cube from 1.7 cm up
    place_drop: float = 0.004     # release this much higher than the grasp height, so the block falls 4 mm
    squeeze: float = 0.008        # command this much narrower than the block: ~8 N per pad at kp 1000
    open_margin: float = 0.03     # open to block width + this before descending (capped at the jaw max)
    v_travel: float = 0.15        # m/s at travel height (= the safety filter's cap)
    v_vertical: float = 0.10      # m/s descending and lifting
    v_final: float = 0.05         # m/s for the last 2 cm of a descent
    xy_tol: float = 0.006         # EE within this of the goal counts as arrived
    z_tol: float = 0.004
    yaw_tol: float = 0.03         # rad, before descending
    drop_grace: float = 0.15      # s without pad contact before a carried block counts as dropped
    settle_s: float = 0.1         # s of still jaws before judging the grasp
    timeout_s: float = 15.0       # active (not held) seconds for the whole skill
    stall_s: float = 3.0          # s without 3 mm of EE progress in a motion phase


# Advertised as-is from the standard catalog: shared names are what make recordings transfer.
SPECS = (catalog.MOVE_OBJECT, catalog.HOLD, catalog.SURVEY)
REASONS = catalog.REASONS      # a reason is "<code>: <literal text>"


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def grasp_yaw(x: float, y: float, block_yaw: float) -> float:
    """Gripper yaw whose closing axis meets a face of the block squarely and stays nearest the arm's
    azimuth: the four candidates differ by 90 deg, so the wrist roll is at most +-45 deg."""
    az = math.atan2(y, x)
    rel = (block_yaw - az + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return az + rel


class Skill:
    """Base: timing, hold/pause/resume, stall and timeout, the status record. Subclasses implement
    `first_phase`, `run(io)` and optionally `monitor(io)` (checks that run even while held)."""
    name = ""
    first_phase = "start"

    def __init__(self, sid: str, arm: str, args: dict, p: Params):
        self.id, self.arm, self.args, self.p = sid, arm, dict(args), p
        self.state = "running"          # running | holding | paused | done | failed
        self.reason: str | None = None
        self.phase = "start"
        self.text = "starting"
        self.heading: tuple[float, float] | None = None
        self.tripped = False
        self.trip_text = ""
        self.active_s = 0.0             # time spent running (held time excluded) -> the timeout
        self.phase_s = 0.0              # same, since the phase began
        self._last_t: float | None = None
        self._dt = 0.0
        self._prog_t = 0.0
        self._prog_p: tuple[float, float, float] | None = None

    # -- lifecycle (called by the server, under its lock) --------------------------------------------
    def tick(self, io: ArmIO) -> None:
        if self.state in ("done", "failed"):
            return
        if self._last_t is None:
            self._last_t = io.t
            self._enter(io, self.first_phase)
        dt = io.t - self._last_t
        self._last_t = io.t
        self._dt = dt
        if self.state in ("holding", "paused"):
            self.monitor(io)
            return
        self.active_s += dt
        self.phase_s += dt
        if self.active_s > self.p.timeout_s:
            return self.fail(io, f"timeout: {self.p.timeout_s:.0f} s of motion, still in phase '{self.phase}' ({self.text})")
        self.monitor(io)
        if self.state == "running":
            self.run(io)

    def hold(self, io: ArmIO) -> None:
        if self.state == "running":
            io.freeze()
            self.state = "holding"

    def pause(self, io: ArmIO) -> None:
        if self.state in ("running", "holding"):
            io.freeze()
            self.state = "paused"

    def trip(self, io: ArmIO, text: str) -> None:
        """Safety filter tripped: hold, remember why; resume() is the only way out."""
        if self.state in ("running", "holding", "paused"):
            io.freeze()
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
        """No 3 mm of EE progress for stall_s while the phase has not finished."""
        p = io.ee()
        if self._prog_p is None or math.dist(p, self._prog_p) > 0.003:
            self._prog_t, self._prog_p = io.t, p
            return False
        return io.t - self._prog_t > self.p.stall_s

    def _at(self, io: ArmIO, goal, z_tol: float | None = None) -> bool:
        x, y, z = io.ee()
        return (io.arrived() and math.hypot(x - goal[0], y - goal[1]) <= self.p.xy_tol
                and abs(z - goal[2]) <= (z_tol if z_tol is not None else self.p.z_tol))

    def monitor(self, io: ArmIO) -> None:
        pass

    def run(self, io: ArmIO) -> None:
        raise NotImplementedError

    def done(self, io: ArmIO, text: str) -> None:
        self.state, self.phase, self.text, self.heading = "done", "done", text, None

    def fail(self, io: ArmIO, reason: str) -> None:
        self.state, self.reason, self.heading = "failed", reason, None
        io.freeze()    # a failed skill leaves the arm where it is; the brain decides what next

    def status(self) -> SkillStatus:
        text = self.text
        if self.state == "holding":
            text = (f"safety trip ({self.trip_text}), held: " if self.tripped else "held: ") + text
        elif self.state == "paused":
            text = "paused: " + text
        elif self.state == "failed":
            text = f"failed in phase '{self.phase}': {self.reason}"
        return SkillStatus(self.id, self.arm, self.name, self.state, text, self.reason, self.heading, dict(self.args))


class MoveObject(Skill):
    name = "move_object"
    first_phase = "approach"
    PHASES = ("approach", "descend", "close", "lift", "carry", "lower", "release", "retreat")

    def __init__(self, sid, arm, args, p):
        super().__init__(sid, arm, args, p)
        self.obj = str(args["object"])
        self.place = str(args["place"])
        self.target: tuple[float, float] | None = None   # where the object was when aimed at
        self.size = 0.04
        self.yaw = 0.0
        self.dest: tuple[float, float] | None = None
        self._w_prev = None
        self._w_still = 0.0
        self._lost_s = 0.0

    def _aim(self, io: ArmIO) -> bool:
        o = io.obj(self.obj)
        if o is None:
            self.fail(io, f"precondition: {self.obj} is not in the world")
            return False
        self.target, self.size = (o.x, o.y), o.size
        self.yaw = grasp_yaw(o.x, o.y, o.yaw)
        self.heading = self.target
        return True

    def retarget(self, io: ArmIO) -> None:
        """Re-read the object. Before the grasp the approach follows it; once descending onto a spot
        the object has left, go back up and approach again."""
        if self.phase not in ("approach", "descend"):
            return
        old = self.target
        if not self._aim(io):
            return
        if self.phase == "descend" and old is not None and math.dist(old, self.target) > 0.01:
            self._enter(io, "approach")

    def monitor(self, io: ArmIO) -> None:
        """Runs held or not: a block that leaves the pads after the grasp is a failure wherever we are."""
        if self.phase in ("lift", "carry", "lower"):
            if io.held() == self.obj:
                self._lost_s = 0.0
            else:
                self._lost_s += self._dt
                if self._lost_s > self.p.drop_grace:
                    x, y, z = io.ee()
                    self.fail(io, f"dropped: {self.obj} left the pads during {self.phase} at ({x:.2f}, {y:.2f}, {z:.2f})")

    def run(self, io: ArmIO) -> None:
        p, ph = self.p, self.phase
        grasp_z = io.table_z + p.tip_height
        open_w = min(io.max_width, self.size + p.open_margin)
        if ph == "approach":
            if self.target is None and not self._aim(io):
                return
            tx, ty = self.target
            goal = (tx, ty, io.travel_z)
            self.text = f"moving above {self.obj} at travel height"
            io.grip(open_w)
            io.goto(goal, p.v_travel, yaw=self.yaw)
            if self._at(io, goal, z_tol=0.01) and abs(wrap(io.ee_yaw() - self.yaw)) < p.yaw_tol:
                self._enter(io, "descend")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress approaching {self.obj}: EE stuck at {_fmt(io.ee())}")
        elif ph == "descend":
            tx, ty = self.target
            goal = (tx, ty, grasp_z)
            self.text = f"descending onto {self.obj}"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - grasp_z < 0.02 else p.v_vertical, yaw=self.yaw)
            if self._at(io, goal):
                self._enter(io, "close")
                self._w_prev, self._w_still = None, 0.0
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress descending onto {self.obj}: fingertips stuck {z - io.table_z:.3f} m above the table")
        elif ph == "close":
            self.text = f"closing on {self.obj}"
            io.grip(max(0.0, self.size - p.squeeze))
            w = io.width()
            self._w_still = self._w_still + 0.02 if (self._w_prev is not None and abs(w - self._w_prev) < 2e-4) else 0.0
            self._w_prev = w
            if self._w_still >= p.settle_s or self.phase_s > 2.0:
                if io.held() == self.obj:
                    self._lost_s = 0.0
                    self._enter(io, "lift")
                else:
                    self.fail(io, f"grasp_failed: the fingers closed to {w * 100:.1f} cm, {self.obj} is {self.size * 100:.1f} cm wide and not between the pads")
        elif ph == "lift":
            tx, ty = self.target
            goal = (tx, ty, io.travel_z)
            self.text = f"lifting {self.obj}"
            io.goto(goal, p.v_vertical)
            if self._at(io, goal, z_tol=0.01):
                d = io.place(self.place)
                if d is None:
                    return self.fail(io, f"precondition: {self.place} is not a place in the world")
                self.dest, self.heading = d, d
                self._enter(io, "carry")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress lifting {self.obj} at {_fmt(io.ee())}")
        elif ph == "carry":
            dx, dy = self.dest
            goal = (dx, dy, io.travel_z)
            self.text = f"carrying {self.obj} to {self.place}"
            io.goto(goal, p.v_travel)
            if self._at(io, goal, z_tol=0.01):
                self._enter(io, "lower")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress carrying {self.obj} at {_fmt(io.ee())}")
        elif ph == "lower":
            dx, dy = self.dest
            goal = (dx, dy, grasp_z + p.place_drop)
            self.text = f"lowering {self.obj} at {self.place}, not yet released"
            z = io.ee()[2]
            io.goto(goal, p.v_final if z - goal[2] < 0.02 else p.v_vertical)
            if self._at(io, goal, z_tol=0.006):
                self._enter(io, "release")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress lowering {self.obj} at {self.place}: EE {z - io.table_z:.3f} m above the table")
        elif ph == "release":
            self.text = f"releasing {self.obj} at {self.place}"
            io.grip(open_w)
            if io.width() >= self.size + 0.012:
                self._enter(io, "retreat")
            elif self.phase_s > 2.0:
                self.fail(io, f"blocked: the gripper did not open at {self.place}: width {io.width() * 100:.1f} cm")
        elif ph == "retreat":
            dx, dy = self.dest
            goal = (dx, dy, io.travel_z)
            self.text = f"released {self.obj} at {self.place}, rising clear"
            io.goto(goal, p.v_vertical)
            if self._at(io, goal, z_tol=0.01):
                self.done(io, f"{self.obj} placed at {self.place}")
            elif self._stalled(io):
                self.fail(io, f"stalled: no progress rising clear of {self.place} at {_fmt(io.ee())}")


class HoldStill(Skill):
    """The planned skill 'hold': stay where the arm is for a while (default 1 s), grip unchanged."""
    name = "hold"
    first_phase = "still"

    def run(self, io: ArmIO) -> None:
        secs = float(self.args.get("seconds", 1.0))
        self.text = f"holding still ({self.phase_s:.1f} of {secs:.1f} s)"
        if self.phase_s >= secs:
            self.done(io, f"held still for {secs:.1f} s")


class Survey(Skill):
    """Rise straight up to travel height, keeping xy and the grip."""
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
            self.done(io, "at travel height")
        elif self._stalled(io):
            self.fail(io, f"stalled: no progress rising at {_fmt(io.ee())}")


SKILLS = {"move_object": MoveObject, "hold": HoldStill, "survey": Survey}


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"
