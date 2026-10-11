"""A stand-in for the trossen_arm driver, so the real server runs here with no arm and no driver
package. It mimics exactly the surface arm/real.py's `RealArm` uses, with kinematics and nothing
else: a non-blocking Cartesian command interpolates the pose linearly over its goal_time (what the
controller does for goal_time in (0.001, 0.2]); a blocking one lands at once; the gripper carriage
moves toward its target at GRIP_RATE and stops on a block the world says is between the fingers
(half the block's width per side), in position mode and under a closing external effort alike; the
external efforts are whatever a test sets (`efforts`), so the effort watchdog can be tripped on
demand. Joint positions 0-5 are not computed (no FK here): zeros, except the gripper, joint 6.

`install_fake_trossen_arm()` registers a `trossen_arm` module with the enums `RealArm.start()` and
its I/O thread import (Mode, InterpolationSpace, Model, StandardEndEffector) when the real package
is not installed: `RealArm` imports it for those even when given a `driver_factory`. The stand-in's
`TrossenArmDriver` refuses to be built, so nothing can mistake it for the real driver.
"""
from __future__ import annotations

import enum
import importlib
import importlib.util
import sys
import time
import types
from typing import Callable

from robojev.robots.widowx_real.world import StaticWorld

GRIP_MAX = 0.04          # m of travel per carriage (the driver's 0-0.04; two carriages = 8 cm opening)
GRIP_RATE = 0.05         # m/s per carriage: 0.04 m in 0.8 s, inside RealArm's 1.1 s force-close window
STAGED_POSE = (0.20, 0.0, 0.30, 0.0, 0.5, 0.0)     # where the arm "is" after the STAGED joint move
SLEEP_POSE = (0.10, 0.0, 0.05, 0.0, 0.0, 0.0)      # ... and folded at SLEEP


class FakeTrossenDriver:
    """See the module docstring. `world` is a StaticWorld, a callable returning the current one (the
    conformance CLI swaps worlds between trials), or None (no blocks: the fingers close to zero)."""

    def __init__(self, world: StaticWorld | Callable[[], StaticWorld] | None = None, clock: Callable[[], float] = time.perf_counter):
        self._world: Callable[[], StaticWorld | None] = (lambda: None) if world is None else (world if callable(world) else (lambda: world))
        self._clock = clock
        self.pose = list(SLEEP_POSE)
        self.joints = [0.0] * 7
        self.efforts = [0.0] * 6             # test knob: Cartesian external efforts the driver reports
        self.temps = [35.0] * 7
        self.calls: list[tuple] = []         # (method, *args) of every mode, joint and lifecycle call, in order
        self.cleaned_up = False
        self._cart: tuple[list[float], list[float], float, float] | None = None   # from, to, t0, goal_time
        self._grip_mode = "position"
        self._grip_target = 0.0
        self._grip_effort = 0.0
        self._grip_t: float | None = None

    # -- lifecycle and modes ---------------------------------------------------------------------------
    def configure(self, model, end_effector, ip: str, clear_error: bool) -> None:
        self.calls.append(("configure", ip, clear_error))

    def cleanup(self) -> None:
        self.calls.append(("cleanup",))
        self.cleaned_up = True

    def set_all_modes(self, mode) -> None:
        self.calls.append(("set_all_modes", mode))

    def set_gripper_mode(self, mode) -> None:
        self.calls.append(("set_gripper_mode", mode))
        self._advance()
        self._grip_mode = "effort" if "effort" in str(mode) else "position"
        self._grip_target = self.joints[6]

    # -- commands --------------------------------------------------------------------------------------
    def set_all_positions(self, goal, goal_time: float, blocking: bool) -> None:
        """A joint-space move (staging and parking). The pose jumps to the matching constant."""
        self.calls.append(("set_all_positions", tuple(goal), goal_time, blocking))
        self.joints = [float(v) for v in goal]
        self.pose = list(STAGED_POSE if abs(goal[1]) > 0.1 else SLEEP_POSE)
        self._cart = None

    def set_cartesian_positions(self, goal, space, goal_time: float, blocking: bool) -> None:
        goal = [float(v) for v in goal]
        if blocking or goal_time <= 0:
            self._cart, self.pose = None, goal
        else:
            self._cart = (list(self._pose_now()), goal, self._clock(), goal_time)

    def set_gripper_position(self, position: float, goal_time: float, blocking: bool) -> None:
        self._advance()
        self._grip_mode, self._grip_target = "position", max(0.0, min(GRIP_MAX, float(position)))
        if blocking:
            self.joints[6] = self._grip_target

    def set_gripper_external_effort(self, effort: float, goal_time: float, blocking: bool) -> None:
        self._advance()
        self._grip_mode, self._grip_effort = "effort", float(effort)

    # -- reads -----------------------------------------------------------------------------------------
    def get_cartesian_positions(self) -> list[float]:
        return list(self._pose_now())

    def get_all_positions(self) -> list[float]:
        self._advance()
        return list(self.joints)

    def get_cartesian_external_efforts(self) -> list[float]:
        return list(self.efforts)

    def get_all_rotor_temperatures(self) -> list[float]:
        return list(self.temps)

    # -- kinematics ------------------------------------------------------------------------------------
    def _pose_now(self) -> list[float]:
        if self._cart is not None:
            frm, to, t0, T = self._cart
            a = min(1.0, (self._clock() - t0) / T)
            self.pose = [f + (t - f) * a for f, t in zip(frm, to)]
            if a >= 1.0:
                self._cart = None
        return self.pose

    def _advance(self) -> None:
        """Move the gripper carriage toward its target at GRIP_RATE, never past a block between the fingers."""
        now = self._clock()
        dt = 0.0 if self._grip_t is None else now - self._grip_t
        self._grip_t = now
        if self._grip_mode == "effort":
            target = 0.0 if self._grip_effort < 0 else GRIP_MAX if self._grip_effort > 0 else self.joints[6]
        else:
            target = self._grip_target
        q = self.joints[6]
        step = GRIP_RATE * dt
        if target < q:
            q_new = max(target, q - step)
            stop = self._block_stop()
            if stop is not None:
                q_new = max(q_new, min(q, stop))
        else:
            q_new = min(target, q + step)
        self.joints[6] = q_new

    def _block_stop(self) -> float | None:
        w = self._world()
        if w is None:
            return None
        oid = w.under(tuple(self._pose_now()[:3]))
        return None if oid is None else w.pose(oid).size / 2


class FakeDriverFactory:
    """A `driver_factory` for RealArm / RealRobotServer that builds a FakeTrossenDriver over `world`
    (a StaticWorld or a callable returning the current one) and keeps the last one built in
    `driver`, so a test can set its `efforts` or inspect its `calls`. Building one installs the
    trossen_arm stand-in when the real package is absent."""

    def __init__(self, world: StaticWorld | Callable[[], StaticWorld] | None = None):
        install_fake_trossen_arm()        # RealArm.start() imports trossen_arm before it calls the factory
        self.world = world
        self.driver: FakeTrossenDriver | None = None

    def __call__(self, ip: str) -> FakeTrossenDriver:
        self.driver = FakeTrossenDriver(self.world)
        self.driver.configure(None, None, ip, True)
        return self.driver


def real_driver_available() -> bool:
    """True when the real trossen_arm package is importable (not the stand-in). Tests of the real
    driver path skip on False."""
    m = sys.modules.get("trossen_arm")
    if m is not None:
        return not getattr(m, "__robojev_fake__", False)
    return importlib.util.find_spec("trossen_arm") is not None


def install_fake_trossen_arm() -> types.ModuleType:
    """Return the real `trossen_arm` if it is installed, else register and return the stand-in
    (enums only; its TrossenArmDriver cannot be built). Idempotent."""
    m = sys.modules.get("trossen_arm")
    if m is not None:
        return m
    if importlib.util.find_spec("trossen_arm") is not None:
        return importlib.import_module("trossen_arm")
    m = types.ModuleType("trossen_arm")
    m.__doc__ = "robojev stand-in for the trossen_arm driver package: enums only (robots/widowx_real/fake_driver.py)"
    m.__robojev_fake__ = True
    m.Mode = enum.Enum("Mode", "idle position velocity external_effort")
    m.InterpolationSpace = enum.Enum("InterpolationSpace", "joint cartesian")
    m.Model = enum.Enum("Model", "wxai_v0")
    m.StandardEndEffector = enum.Enum("StandardEndEffector", "wxai_v0_follower wxai_v0_leader")

    class TrossenArmDriver:
        def __init__(self, *a, **k):
            raise RuntimeError("the trossen_arm package is not installed; this is robojev's enum-only stand-in. "
                               "Pass a driver_factory (FakeDriverFactory for tests) or install robojev[real].")

    m.TrossenArmDriver = TrossenArmDriver
    sys.modules["trossen_arm"] = m
    return m
