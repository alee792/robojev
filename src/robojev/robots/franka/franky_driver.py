"""The real Franka driver: `FrankaDriver` (driver.py) over franky, for an FR3 on the current libfranka.

Written against franky 2.0.0's Python API (its names are the ones used here; `franky-control` is the
package, project extra `franka`). Nothing in this file has run against an arm yet: the pose maths,
the no-op and preemption rules and the error mapping are tested with a stand-in franky module
(tests/test_franky_driver.py); the constants a real arm fixes are the constructor's keyword
arguments, each marked VERIFY, and docs/franka-day-one.md says how to measure them.

What franky does for us: it holds the 1 kHz conversation with the arm (libfranka, on the RT control
box this process runs on), turns a target pose into a smooth trajectory (Ruckig), lets a new target
take over a running motion seamlessly, drives the gripper, stops, reports state and errors, and
clears a reflex. We never see a joint except in `park()`.

The mapping, call by call:
  set_target  Robot.move(CartesianMotion(CartesianState(RobotPose(Affine(...))),
              relative_dynamics_factor=RelativeDynamicsFactor(v, a, j)), asynchronous=True).
              The server repeats the same goal every tick: a repeat within `same_goal_m` and
              `same_yaw_rad` is a no-op here, a changed goal is a new motion (franky queues it and
              takes over). The velocity factor is speed / Robot.translation_velocity_limit; the
              acceleration and jerk factors are fixed gentle fractions so a new goal never jolts.
  set_gripper Gripper.move_async(width, speed) to open; Gripper.grasp_async(width, speed, force,
              epsilon_inner, epsilon_outer) to close on something. The policies command a close
              `squeeze` narrower than the block, so both epsilons are wide (`grasp_epsilon`, 4 cm):
              libfranka's own "grasped" then means "the fingers stopped short on something", which
              is what FrankaState.grasped promises. A repeat of the same width is a no-op.
  stop        Robot.stop(): the running motion ends within the arm's own limits; the grip is kept;
              the last goal is forgotten so the next set_target starts a new motion.
  recover     Robot.recover_from_errors(). The server calls it from resume() after a safety_trip.
  park        Robot.move(JointMotion(JointState(park_q)), asynchronous=False) at a low dynamics
              factor: blocking, from STOP's own thread. The grip is kept.
  read        Robot.state for O_T_EE (the end-effector frame as Desk has it), robot_mode and
              current_errors; Gripper.width and Gripper.is_grasped; Robot.is_in_control for
              `moving`. Any exception from a franky call lands in `FrankaState.error` instead of
              the policy loop.

Frames. Desk's end-effector frame for the Franka Hand is normally the point between the fingertips
(F_T_EE with z = 0.1034 m); `tip_offset` adds anything along the tool z axis that Desk does not
include (0 if Desk is set as above; VERIFY by the touch-off). The tool points straight down at every
target: the EE z axis is -z of the base. The closing axis (fingertip to fingertip) is the Franka
Hand's EE y axis (`closing_axis="y"`, VERIFY on the first yawed grasp: if the wrist turns the wrong
way, the sign of the yaw is off by pi/2 and this is the knob); `yaw` is that axis's direction in
the base xy plane, as skills.ArmIO.ee_yaw defines it. The target rotation is built as columns
[x_ee, y_ee, z_ee] = [(-sin y, cos y, 0), (cos y, sin y, 0), (0, 0, -1)], right-handed.
"""
from __future__ import annotations

import importlib
import math
import threading
import time
from types import ModuleType

import numpy as np

from robojev.robots.franka.driver import FRANKA_IP, FrankaState

READY_Q = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)   # Franka's "ready" pose, VERIFY it clears the tray


def down_matrix(xyz, yaw: float, tip_offset: float = 0.0, closing_axis: str = "y") -> np.ndarray:
    """The 4x4 base->EE transform for the tool straight down with the closing axis at `yaw`, the EE
    origin `tip_offset` back up the tool from the fingertip point (module doc)."""
    c, s = math.cos(yaw), math.sin(yaw)
    closing = np.array([c, s, 0.0])
    other = np.array([-s, c, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    if closing_axis == "y":
        x, y = other, closing
    elif closing_axis == "x":
        x, y = closing, -other            # keep x cross y = z
    else:
        raise ValueError(f"closing_axis must be 'x' or 'y', not {closing_axis!r}")
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    m[:3, 3] = np.asarray(xyz, dtype=float) - z * tip_offset     # the EE origin is above the tip along the tool
    return m


def read_pose(matrix: np.ndarray, tip_offset: float = 0.0, closing_axis: str = "y") -> tuple[tuple[float, float, float], float]:
    """(fingertip point, closing-axis yaw) from a base->EE 4x4."""
    m = np.asarray(matrix, dtype=float)
    r, p = m[:3, :3], m[:3, 3]
    tip = p + r[:, 2] * tip_offset
    axis = r[:, 1] if closing_axis == "y" else r[:, 0]
    return (float(tip[0]), float(tip[1]), float(tip[2])), float(math.atan2(axis[1], axis[0]))


def _errors_text(errors) -> str | None:
    """libfranka's error struct -> the names of the flags that are set, or None."""
    names = [n for n in dir(errors) if not n.startswith("_") and getattr(errors, n, False) is True]
    if names:
        return ", ".join(names)
    text = str(errors).strip()
    return None if text in ("", "[]") else text


class FrankyDriver:
    """driver.FrankaDriver over franky (module doc). Thread-safe for the server's use: one policy
    thread reads and commands, STOP's thread calls stop()/park()."""

    def __init__(self, ip: str = FRANKA_IP, *, speed_cap: float = 0.25, accel_factor: float = 0.2, jerk_factor: float = 0.2,
                 tip_offset: float = 0.0, closing_axis: str = "y", park_q=READY_Q, park_factor: float = 0.1,
                 torque_threshold: float = 20.0, force_threshold: float = 20.0,
                 gripper_speed: float = 0.1, grasp_epsilon: float = 0.04, squeeze_force_default: float = 20.0,
                 same_goal_m: float = 0.0005, same_yaw_rad: float = 0.005, home_gripper: bool = False,
                 franky_module: ModuleType | None = None):
        """VERIFY on the arm: tip_offset (touch-off), closing_axis (first yawed grasp), park_q (clears
        the tray from everywhere in the box), torque/force thresholds (a brush of a hand trips, a cube
        does not; franky's defaults are 20 Nm / 30 N), speed_cap (0.25 m/s next to a person).
        `home_gripper` runs the Hand's homing at connect (needed once after changing fingers)."""
        self.fk = franky_module or importlib.import_module("franky")
        self.ip = ip
        self.tip_offset, self.closing_axis = tip_offset, closing_axis
        self.park_q, self.park_factor = tuple(park_q), park_factor
        self.speed_cap, self.accel_factor, self.jerk_factor = speed_cap, accel_factor, jerk_factor
        self.gripper_speed, self.grasp_epsilon, self.squeeze_force_default = gripper_speed, grasp_epsilon, squeeze_force_default
        self.same_goal_m, self.same_yaw_rad = same_goal_m, same_yaw_rad
        self._lock = threading.RLock()
        self._goal: tuple[tuple[float, float, float], float] | None = None
        self._grip: tuple[float, float | None] | None = None      # last commanded (width, force)
        self._gfut = None
        self._fault: str | None = None                            # a franky call that raised, until the next read clears it
        self._t0 = time.monotonic()
        self.robot = self.fk.Robot(ip, default_torque_threshold=torque_threshold, default_force_threshold=force_threshold)
        self.gripper = self.fk.Gripper(ip)
        if home_gripper:
            self.gripper.homing()
        self.v_limit = float(self.robot.translation_velocity_limit) or 1.0

    # -- FrankaDriver ----------------------------------------------------------------------------------
    def read(self) -> FrankaState:
        with self._lock:
            try:
                st = self.robot.state
                tip, yaw = read_pose(st.O_T_EE.matrix, self.tip_offset, self.closing_axis)
                mode = getattr(st, "robot_mode", None)
                err = _errors_text(st.current_errors) if self.robot.has_errors else None
                if err is None and mode is not None and mode.name in ("Reflex", "UserStopped"):
                    err = {"Reflex": "reflex: the arm stopped itself and waits to be recovered",
                           "UserStopped": "user stopped: the activation device is released"}[mode.name]
                if err is None and self._fault:
                    err = self._fault
                moving = bool(self.robot.is_in_control)
                width = float(self.gripper.width)
                grasped = bool(self.gripper.is_grasped) and self._grip is not None and self._grip[1] is not None
                gmoving = self._gfut is not None and not self._gfut.wait(0.0)
                return FrankaState(t=time.monotonic() - self._t0, xyz=tip, yaw=yaw, width=width, moving=moving,
                                   error=err, grasped=grasped, gripper_moving=gmoving)
            except Exception as e:      # a lost connection or a libfranka exception: the arm is not ours to command
                return FrankaState(t=time.monotonic() - self._t0, xyz=(0.0, 0.0, 0.0), yaw=0.0, width=0.0, moving=False,
                                   error=f"driver: {e.__class__.__name__}: {e}")

    def set_target(self, xyz, yaw: float, speed: float) -> None:
        goal = ((float(xyz[0]), float(xyz[1]), float(xyz[2])), float(yaw))
        with self._lock:
            if self._goal is not None and math.dist(goal[0], self._goal[0]) <= self.same_goal_m \
                    and abs(_wrap(goal[1] - self._goal[1])) <= self.same_yaw_rad:
                return                                        # the same goal again: franky is already on it
            fk = self.fk
            v = max(0.01, min(1.0, min(speed, self.speed_cap) / self.v_limit))
            pose = fk.Affine(down_matrix(goal[0], goal[1], self.tip_offset, self.closing_axis))
            motion = fk.CartesianMotion(fk.CartesianState(fk.RobotPose(pose)),
                                        relative_dynamics_factor=fk.RelativeDynamicsFactor(v, self.accel_factor, self.jerk_factor))
            try:
                self.robot.move(motion, asynchronous=True)
                self._goal, self._fault = goal, None
            except Exception as e:                            # refused (error state, bad pose): shows in read().error
                self._fault = f"move refused: {e.__class__.__name__}: {e}"

    def set_gripper(self, width: float, force: float | None = None) -> None:
        cmd = (round(float(width), 4), None if force is None else float(force))
        with self._lock:
            if cmd == self._grip:
                return
            try:
                if force is None:
                    self._gfut = self.gripper.move_async(cmd[0], self.gripper_speed)
                else:
                    self._gfut = self.gripper.grasp_async(cmd[0], self.gripper_speed, cmd[1],
                                                          epsilon_inner=self.grasp_epsilon, epsilon_outer=self.grasp_epsilon)
                self._grip = cmd
            except Exception as e:
                self._fault = f"gripper refused: {e.__class__.__name__}: {e}"

    def stop(self) -> None:
        with self._lock:
            self._goal = None
            try:
                self.robot.stop()
            except Exception as e:
                self._fault = f"stop: {e.__class__.__name__}: {e}"

    def recover(self) -> None:
        with self._lock:
            self._fault = None
            self._goal = None
            try:
                self.robot.recover_from_errors()
            except Exception as e:
                self._fault = f"recover failed: {e.__class__.__name__}: {e}"

    def park(self) -> None:
        with self._lock:
            self._goal = None
            fk = self.fk
            try:
                self.robot.move(fk.JointMotion(fk.JointState(list(self.park_q)), relative_dynamics_factor=fk.RelativeDynamicsFactor(self.park_factor)),
                                asynchronous=False)
            except Exception as e:
                self._fault = f"park: {e.__class__.__name__}: {e}"

    def close(self) -> None:
        with self._lock:
            try:
                self.robot.stop()
            except Exception:
                pass


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
