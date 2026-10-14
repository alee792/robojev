"""The policies' `skills.ArmIO` over a `driver.FrankaDriver`, and the Franka's numbers.

Compared with the WidowX adapter (robots/widowx_real/arm.py) there is no Mover, no orientation
proxy and no carriage arithmetic: the driver takes a fingertip point, a yaw and a width directly,
and tracks them itself (franky's trajectory generator, or the fake's first-order tracker). So each
`goto` clamps the goal to the workspace box, caps the speed and hands it over; `arrived()` is the
driver reporting the motion finished with the fingertips within `arrive_tol` of the goal.

`held()` is bookkeeping plus the gripper, as on the WidowX: the block the world said was between
the fingers when the close was commanded, for as long as the fingers report stopped on something
(`FrankaState.grasped`) at about that block's width. A block that slips out shows as the fingers
closing further. No camera checks any of it.

A trip is the arm's own: `FrankaState.error` set (a collision reflex, a limit). The adapter freezes
itself (nothing is sent while frozen) and the server turns the error into a `safety_trip`;
`resume()` on the server calls the driver's `recover()` before unfreezing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from robojev.config import Workspace
from robojev.protocol import HandObs
from robojev.robots.franka.driver import FrankaDriver, FrankaState
from robojev.robots.widowx_real.world import WorldSource
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.server import TRAVEL_Z as SIM_TRAVEL_Z

ARM_ID = "arm_0"
BASE_FRAME = "fr3_link0"          # the arm's base (franka_description's name for the FR3; a Panda's is panda_link0): +x forward, +y left, +z up, as libfranka reports O_T_EE


@dataclass(frozen=True)
class FrankaLimits:
    """The Franka's numbers in one place. Every one is a starting guess to verify on the arm; the
    table height comes from blocks.json (`for_table`)."""
    x: tuple[float, float] = (0.25, 0.75)       # m from the base: nearer than 0.25 the elbow folds up behind a straight-
                                                # down wrist; the flange reaches 0.855, and 0.75 leaves the hand's reach margin
    y: tuple[float, float] = (-0.40, 0.40)      # across the table in front of the arm; the arm reaches further, the table does not
    table_z: float = scene.TABLE_Z              # the table top in the base frame, from the touch-off; the sim's until measured
    floor_above_table: float = 0.004            # the setpoint never goes under 4 mm above the table (the sim's rule)
    ceiling_above_table: float = 0.30           # high enough for a stack and the hand-over, under the "ready" pose
    travel_above_table: float = SIM_TRAVEL_Z - scene.TABLE_Z   # 0.12 m: the sim's travel height over its table
    speed_cap: float = 0.25                     # m/s for the fingertips, whatever a skill asks; franky's default Cartesian
                                                # limits allow far more, this is what a person next to the table is comfortable with
    max_width: float = 0.08                     # m between the Franka Hand's pads fully open
    arrive_tol: float = 0.008                   # fingertips within this of the goal, motion finished, = arrived (sag, if any)
    yaw_tol: float = 0.02                       # rad, the yaw setpoint counts as reached
    grasp_tol: float = 0.006                    # fingers stopped within this of the block's width = holding it
    grasp_force: float = 20.0                   # N for a close on a block: the Hand does 70 N; 20 N holds a wooden cube
    hover_xy: tuple[float, float] = (0.40, 0.0)  # where open() takes the fingertips, at travel height, before any skill

    @property
    def workspace(self) -> Workspace:
        return Workspace(x=self.x, y=self.y, z=(self.table_z + self.floor_above_table, self.table_z + self.ceiling_above_table))

    @property
    def travel_z(self) -> float:
        return self.table_z + self.travel_above_table

    def for_table(self, table_z: float) -> "FrankaLimits":
        return replace(self, table_z=float(table_z))


FRANKA_LIMITS = FrankaLimits()

# The sim's Params with what a real arm needs loosened: xy/z tolerances for whatever sag the arm
# shows, settle_s past the gripper's own loop (a Franka Hand move takes up to 0.8 s), timeout for a
# slower arm. Verify on hardware.
FRANKA_PARAMS = replace(skills.Params(), xy_tol=0.008, z_tol=0.006, settle_s=0.3, close_s=2.5, timeout_s=20.0)


class FrankaArmIO:
    """skills.ArmIO and base.ServedArm for one Franka arm over a FrankaDriver. Reads come from the
    state `refresh()` took at the start of the server's tick (one driver read per tick); commands
    go straight to the driver. Runs under the server's lock."""

    def __init__(self, driver: FrankaDriver, world: WorldSource, limits: FrankaLimits = FRANKA_LIMITS, arm_id: str = ARM_ID):
        self.driver, self.world, self.limits = driver, world, limits
        self.id = arm_id
        self.t = 0.0
        self.table_z = limits.table_z
        self.travel_z = limits.travel_z
        self.max_width = limits.max_width
        self.skill: skills.Skill | None = None
        self.stopped = False
        self.frozen_why: str | None = None
        self._frozen = False
        self._state: FrankaState | None = None
        self.goal: tuple[float, float, float] | None = None
        self.yaw_goal: float | None = None
        self.speed = 0.0
        self.grip_width = limits.max_width            # last commanded width
        self._closing_on: str | None = None           # the block between the fingers when the close was commanded
        self.error_seen: str | None = None            # the driver error the server has already reported

    # -- reads -----------------------------------------------------------------------------------------
    def refresh(self) -> FrankaState:
        self._state = self.driver.read()
        return self._state

    def snap(self) -> FrankaState:
        return self._state if self._state is not None else self.refresh()

    def ee(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.snap().xyz)

    def ee_yaw(self) -> float:
        return float(self.snap().yaw)

    def arrived(self) -> bool:
        s = self.snap()
        if self.goal is None or self._frozen or s.error is not None:
            return False
        yaw_ok = self.yaw_goal is None or abs(skills.wrap(s.yaw - self.yaw_goal)) <= self.limits.yaw_tol
        return not s.moving and math.dist(s.xyz, self.goal) <= self.limits.arrive_tol and yaw_ok

    def width(self) -> float:
        return float(self.snap().width)

    def reachable(self, xyz) -> bool:
        return self.limits.workspace.contains(xyz)

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
        if self._closing_on is None:
            return None
        s = self.snap()
        o = self.world.pose(self._closing_on)
        if o is None or not s.grasped or abs(s.width - o.size) > self.limits.grasp_tol:
            return None                              # the fingers closed past it: it is gone
        return self._closing_on

    # -- commands --------------------------------------------------------------------------------------
    def goto(self, xyz, speed: float, yaw: float | None = None) -> None:
        self.goal = self.limits.workspace.clamp(tuple(float(v) for v in xyz))
        self.speed = min(float(speed), self.limits.speed_cap)
        if yaw is not None:
            self.yaw_goal = float(yaw)
        if self.yaw_goal is None:
            self.yaw_goal = self.ee_yaw()
        if not self._frozen:
            self.driver.set_target(self.goal, self.yaw_goal, self.speed)

    def grip(self, width: float) -> None:
        w = max(0.0, min(self.limits.max_width, float(width)))
        self.grip_width = w
        under = self.world.under(self.ee())
        o = self.world.pose(under) if under else None
        if o is not None and w < o.size:
            self._closing_on = under                 # a close on a block: squeeze with force
            self.driver.set_gripper(w, self.limits.grasp_force)
        else:
            self._closing_on = None
            self.driver.set_gripper(w, None if w > 0.0 else self.limits.grasp_force)   # the push closes on nothing, firmly

    def freeze(self) -> None:
        """Stop where we are, keep the grip; nothing is sent until `unfreeze()`."""
        if not self._frozen:
            self._frozen = True
            self.driver.stop()

    def unfreeze(self) -> None:
        self._frozen = False
        self.frozen_why = None

    # -- for the server --------------------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def active(self) -> bool:
        return self.skill is not None and not self.skill.finished

    def new_error(self) -> str | None:
        """The driver's error, the first time it is seen; None once reported or once cleared."""
        err = self.snap().error
        if err is None:
            self.error_seen = None
            return None
        if err == self.error_seen:
            return None
        self.error_seen = err
        return err

    def mode(self) -> str:
        if self.stopped:
            return "stopped"
        if self.active:
            s = self.skill
            return "tripped" if s.state == "holding" and s.tripped else s.state
        if self._frozen and self.frozen_why:
            return "tripped" if self.frozen_why == "trip" else "holding"
        return "idle"

    def observation(self) -> dict:
        s = self.snap()
        return {"ee": list(s.xyz), "yaw": s.yaw, "width": s.width, "holding": self.held(), "moving": s.moving,
                "grasped": s.grasped, "error": s.error, "mode": self.mode(), "objects": self.world.poses(), "driver_t": s.t}

    def action(self) -> dict:
        return {"goal": list(self.goal or self.ee()), "speed": self.speed, "yaw": self.yaw_goal, "grip": self.grip_width,
                "frozen": self._frozen}
