"""The policies' `skills.ArmIO` over v1's `RealArm` (arm/real.py, wrapped unchanged), the real arm's
tolerances, and the day-one Config.

What `RealArm` gives and how it is used:
  - `command(goal, speed_cap, gripper, pitch)`: the goal goes through v1's Mover (workspace clamp,
    speed cap) in RealArm's own 20 Hz thread; pitch is streamed rate-limited at 0.6 rad/s. Every
    `goto` here asks for `pitch = pi/2` (straight down, the sim's grasp). Verify on hardware: v1
    verified 75 deg only.
  - yaw: RealArm has no yaw command; it streams `[0, pitch, 0]`. `OrientingDriver` sits between
    RealArm and the driver and rewrites that triple into the full angle-axis
    `orientation.down_orientation(yaw, pitch)`, rate-limiting the yaw. Verify on hardware.
  - gripper: the driver's position is one carriage's travel, 0-0.04 m; the policies speak width
    between the fingertips, so width = 2 x position and `max_width` = 0.08 (v1: "two 4 cm
    carriages"). RealArm closes BY FORCE whenever the position is under 0.036 (its paper-cup rule),
    so an opening narrower than 7.2 cm would be read as a close: openings are sent as at least 0.036
    per side, and `CLOSE_BELOW_W` decides what is a close. The neighbour-aware pre-grasp opening is
    therefore lost on the real arm (always >= 7.2 cm). Day-one change to real.py: an explicit mode.
  - `snapshot().holding`: v1's "the fingers stopped short of closed on something", set ~1.1 s after
    the close command (the force-close window), with no object identity. `held()` names the block the
    world said was between the fingers when the close was commanded, and drops it when the fingers
    close `DROP_Q` further than where they stopped (a slipped block; v1 never checked). Verify on
    hardware: the delay, and whether a slip shows in the carriage position.
  - `freeze(reason)` / `resume()`: the Mover freezes in place; RealArm also sends the measured pose
    once so the controller stops pushing on whatever stopped it.
  - RealArm's thread trips on its own (v1's EffortWatchdog, the 3 cm lag trip, the thermal governor)
    by freezing the Mover with a reason. `trip_reason()` surfaces each new one once, for the server
    to turn into a `safety_trip` event and a hold.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from robojev.arm import ArmSnapshot
from robojev.arm.real import RealArm
from robojev.config import DEFAULT, Config, Workspace
from robojev.protocol import HandObs
from robojev.robots.widowx_real.orientation import DOWN_PITCH, closing_yaw_of, down_orientation
from robojev.robots.widowx_real.world import WorldSource
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.server import LIMITS as SIM_LIMITS, TRAVEL_Z as SIM_TRAVEL_Z

ARM_ID = "arm_0"
BASE_FRAME = "base_link"          # the arm's base: +x forward, +y left, +z up, as the driver reports
GRIP_Q_MAX = 0.04                 # m per carriage, the driver's gripper range
MAX_WIDTH = 2 * GRIP_Q_MAX        # m between the fingertips fully open (verify on hardware: v1 config says 8 cm)
OPEN_Q_MIN = 0.036                # per side; RealArm closes by force below this, so every opening is at least it
CLOSE_BELOW_W = 0.044             # a commanded width under this is a close (4 cm block: close 3.2 cm, open >= 4.8 cm)
DROP_Q = 0.004                    # fingers this much narrower than where they stopped on the block = it left them
TRAVEL_ABOVE_TABLE = SIM_TRAVEL_Z - scene.TABLE_Z   # 0.12 m: the sim's travel height over its table
HELD_FREEZE = "held"              # the Mover freeze reason this adapter uses; any other reason is the arm's own trip


@dataclass(frozen=True)
class RealLimits:
    """The real arm's numbers beside the sim's `Limits`. All three are guesses to verify on hardware."""
    speed_cap: float = SIM_LIMITS.speed_cap     # m/s; also the Mover's hard cap via day_one_config
    yaw_rate: float = 1.0                       # rad/s for the yaw setpoint, half the sim's: the wrist roll is a real motor
    arrive_tol: float = 0.015                   # EE within this of the goal, setpoint there, = arrived (gravity sag)


REAL_LIMITS = RealLimits()

# The sim's Params with what the real arm needs loosened. settle_s must exceed RealArm's 1.1 s
# force-close window or the policy judges the grasp before `holding` can be true; xy/z tolerances
# allow for sag (every sim body has gravity compensation; the real arm does not). Verify on hardware.
REAL_PARAMS = replace(skills.Params(), xy_tol=0.012, z_tol=0.010, settle_s=1.3, close_s=2.5, timeout_s=20.0)


def day_one_config(table_z: float = scene.TABLE_Z, limits: RealLimits = REAL_LIMITS, base: Config = DEFAULT) -> Config:
    """v1's Config with the sim's workspace box moved onto the measured table (`table_z`, base frame,
    from the touch-off), the Mover's hard cap at the skills' speed cap, and RealArm's staging pose at
    travel height over the hover start. Everything else (effort trip 40 N, lag trip 3 cm, thermal
    limits) is v1's, which ran on this arm."""
    ws = SIM_LIMITS.workspace
    box = Workspace(x=ws.x, y=ws.y, z=(table_z + (ws.z[0] - scene.TABLE_Z), table_z + (ws.z[1] - scene.TABLE_Z)))
    motion = replace(base.motion, hard_speed_cap=limits.speed_cap, hover_start=(0.28, 0.0), safe_height=TRAVEL_ABOVE_TABLE)
    return replace(base, workspace=box, motion=motion, table_z=table_z)


class OrientingDriver:
    """Between RealArm and the driver: streams the full orientation without editing arm/real.py.

    RealArm sends `[x, y, z, 0, pitch, 0]` every tick while it is streaming (pitch-only, the form it
    was verified in). This proxy rewrites that triple into `down_orientation(yaw_sp, pitch)` with
    `yaw_sp` rate-limited toward the commanded yaw, and forwards everything else untouched: the
    blocking staging moves, and the "relax to the measured pose" command RealArm sends once when
    frozen, whose rotation is not of the pitch-only form. If RealArm ever streams a reached-but-wrong
    orientation instead (its `pitch_ok` False), nothing is rewritten and `rewrites` stays 0.
    Verify on hardware: the first yawed grasp."""

    def __init__(self, driver, yaw_rate: float, yaw0: float = math.pi / 2, clock: Callable[[], float] = time.perf_counter):
        self.driver = driver
        self.yaw_rate = yaw_rate
        self._clock = clock
        self._lock = threading.Lock()
        self.yaw_goal = self.yaw_sp = float(yaw0)     # closing axis along +y: zero wrist roll at azimuth 0
        self._last_t: float | None = None
        self.last_sent: tuple[float, ...] | None = None
        self.rewrites = 0
        self.passed = 0

    def __getattr__(self, name):
        return getattr(self.driver, name)

    def set_yaw(self, yaw: float) -> None:
        with self._lock:
            self.yaw_goal = float(yaw)

    def arrived(self) -> bool:
        with self._lock:
            return abs(skills.wrap(self.yaw_sp - self.yaw_goal)) < 1e-6

    def set_cartesian_positions(self, pose, space, goal_time: float, blocking: bool = False) -> None:
        pose = [float(v) for v in pose]
        if not blocking and pose[3] == 0.0 and pose[5] == 0.0:
            now = self._clock()
            dt = 0.0 if self._last_t is None else min(now - self._last_t, 0.075)
            self._last_t = now
            with self._lock:
                d = skills.wrap(self.yaw_goal - self.yaw_sp)
                step = self.yaw_rate * dt
                self.yaw_sp = self.yaw_goal if abs(d) <= step else self.yaw_sp + math.copysign(step, d)
                pose[3:6] = down_orientation(self.yaw_sp, pose[4])
            self.rewrites += 1
            self.last_sent = tuple(pose)
        else:
            self.passed += 1
        self.driver.set_cartesian_positions(pose, space, goal_time, blocking)


class RealArmIO:
    """skills.ArmIO and base.ServedArm for one real arm. Reads are the latest `RealArm.snapshot()`
    (20 Hz, so a 50 Hz policy sees each one two or three times); commands go to `RealArm.command`
    and the OrientingDriver. Runs under the server's lock."""

    def __init__(self, arm: RealArm, orient: OrientingDriver, world: WorldSource, cfg: Config,
                 limits: RealLimits = REAL_LIMITS, arm_id: str = ARM_ID, arrive_tol: float | None = None):
        self.arm, self.orient, self.world, self.cfg, self.limits = arm, orient, world, cfg, limits
        self.id = arm_id
        self.t = 0.0
        self.table_z = cfg.table_z
        self.travel_z = cfg.table_z + TRAVEL_ABOVE_TABLE
        self.max_width = MAX_WIDTH
        self.arrive_tol = limits.arrive_tol if arrive_tol is None else arrive_tol
        self.skill: skills.Skill | None = None
        self.stopped = False
        self.frozen_why: str | None = None
        self.speed = 0.0
        self.grip_width = 2 * (arm.snapshot().gripper_goal or GRIP_Q_MAX)    # last commanded width
        self._closing_on: str | None = None        # the block between the fingers when the close was commanded
        self._hold_q: float | None = None          # carriage position when `holding` first read true
        self.trip_seen: str | None = None

    # -- reads -----------------------------------------------------------------------------------------
    def snap(self) -> ArmSnapshot:
        return self.arm.snapshot()

    def ee(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.snap().ee)

    def ee_yaw(self) -> float:
        rot = self.snap().rot
        return self.orient.yaw_sp if rot is None else closing_yaw_of(rot)

    def arrived(self) -> bool:
        """The Mover's position and pitch setpoints and the yaw setpoint are at their goals and the EE is
        within `arrive_tol` of the goal: the real arm sags under gravity, so the sim's exact test would
        never pass. The pitch matters on the first move after staging: RealArm starts at the 0.5 rad
        hover pitch and reaches 90 deg at 0.6 rad/s, 1.8 s, longer than the approach itself."""
        m = self.arm.mover
        return (m.setpoint is not None and m.setpoint == m.goal and m.pitch == m.pitch_goal and self.orient.arrived()
                and math.dist(self.ee(), m.goal) <= self.arrive_tol)

    def width(self) -> float:
        return 2 * float(self.snap().gripper)

    def reachable(self, xyz) -> bool:
        return self.cfg.workspace.contains(xyz)

    def obj(self, oid: str) -> skills.ObjPose | None:
        return self.world.pose(oid)

    def objects(self) -> dict[str, skills.ObjPose]:
        return {oid: self.world.pose(oid) for oid in self.world.ids()}

    def where(self, oid: str) -> str:
        return self.world.where(oid, self._held_by())

    def place(self, pid: str) -> tuple[float, float] | None:
        return self.world.place_xy(pid, self._held_by())

    def hands(self) -> list[HandObs]:
        return list(self.world.hands)

    def _held_by(self) -> dict[str, str]:
        return {h: self.id for h in [self.held()] if h}

    def held(self) -> str | None:
        s = self.snap()
        if not s.holding:
            self._hold_q = None
            return None
        if self._closing_on is None:
            return None                            # closed on something the world did not list
        if self._hold_q is None:
            self._hold_q = s.gripper
        elif s.gripper < self._hold_q - DROP_Q:
            return None                            # the fingers closed on: the block is gone
        return self._closing_on

    # -- commands --------------------------------------------------------------------------------------
    def goto(self, xyz, speed: float, yaw: float | None = None) -> None:
        self.speed = min(float(speed), self.limits.speed_cap)
        if yaw is not None:
            self.orient.set_yaw(yaw)
        self.arm.command(tuple(float(v) for v in xyz), self.speed, pitch=DOWN_PITCH)

    def grip(self, width: float) -> None:
        w = max(0.0, min(MAX_WIDTH, float(width)))
        self.grip_width = w
        if w < CLOSE_BELOW_W:
            self._closing_on = self.world.under(self.ee())
            q = w / 2
        else:
            self._closing_on = None
            q = max(w / 2, OPEN_Q_MIN)
        goal = self.arm.mover.goal or self.ee()
        self.arm.command(goal, self.speed, gripper=q, pitch=DOWN_PITCH)

    def freeze(self) -> None:
        self.arm.freeze(HELD_FREEZE)

    def unfreeze(self) -> None:
        self.arm.resume()
        self.frozen_why = None

    # -- for the server --------------------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self.arm.mover.frozen

    @property
    def active(self) -> bool:
        return self.skill is not None and not self.skill.finished

    def trip_reason(self) -> str | None:
        """A freeze the arm's own thread applied (effort, lag, thermal) that has not been reported yet."""
        m = self.arm.mover
        if not m.frozen:
            self.trip_seen = None
            return None
        why = m.freeze_reason
        if why == HELD_FREEZE or why == self.trip_seen:
            return None
        self.trip_seen = why
        return why

    def mode(self) -> str:
        if self.stopped:
            return "stopped"
        if self.active:
            s = self.skill
            return "tripped" if s.state == "holding" and s.tripped else s.state
        if self.frozen and self.frozen_why:
            return "tripped" if self.frozen_why == "trip" else "holding"
        return "idle"

    def observation(self) -> dict:
        s = self.snap()
        return {"ee": list(s.ee), "yaw": self.ee_yaw(), "width": 2 * s.gripper, "holding": self.held(),
                "setpoint": list(s.setpoint), "lag": math.dist(s.ee, s.setpoint), "mode": self.mode(),
                "objects": self.world.poses(), "ext_force": list(s.ext_force), "rot": list(s.rot) if s.rot else None,
                "arm_status": s.status, "snapshot_t": s.t}

    def action(self) -> dict:
        m = self.arm.mover
        return {"goal": list(m.goal or self.ee()), "speed": self.speed, "yaw": self.orient.yaw_goal,
                "orient": list(self.orient.last_sent[3:6]) if self.orient.last_sent else None,
                "grip": self.grip_width, "frozen": self.frozen}
