"""A stand-in for a Franka driver, so the server runs here with no arm, no libfranka and no RT
kernel. It keeps the `driver.FrankaDriver` contract and nothing more: kinematics on a clock.

  - The fingertip point tracks the last `set_target` in a straight line at the commanded speed
    (capped at MAX_SPEED) and the yaw turns toward its target at YAW_RATE: a first-order tracker,
    which is what franky's online trajectory generator looks like from 50 Hz once the short
    acceleration ramp is ignored. A real arm overshoots nothing and sags a little; this one does
    neither, so a test that passes here says the layers above the driver work, not that the arm will.
  - The gripper width moves toward its target at GRIP_RATE and stops on a block the world says is
    between the fingers at the fingertips' height (`StaticWorld.under`); stopping short of the
    target on a block sets `grasped`, which is what the real driver must derive from libfranka's
    `is_grasped` with wide enough epsilons (driver.py).
  - Time comes from `clock` (wall time, or a `LockstepClock.now` in tests) and the state is
    advanced lazily on every read or command, so no thread is needed and lockstep tests are exact.
  - Faults: `fault_after_s` reports a collision reflex error that many seconds after the driver was
    built (the arm stops, drops its target, refuses motion until `recover()`); `inject_error()` does
    the same at will. `park()` lands at once, where the real one blocks for a joint move.
"""
from __future__ import annotations

import math
import time
from typing import Callable

from robojev.robots.franka.driver import FrankaState
from robojev.robots.widowx_real.world import StaticWorld
from robojev.robots.widowx_sim import skills

MAX_SPEED = 0.5            # m/s, above any cap the server would ask for: the fake never limits a test
YAW_RATE = 1.5             # rad/s for the yaw: a wrist joint, not instant
GRIP_RATE = 0.10           # m/s of width (the Franka Hand's default; it does the full 8 cm in 0.8 s)
MAX_WIDTH = 0.08           # m between the pads, Franka Hand
READY_XYZ = (0.307, 0.0, 0.487)    # fingertip point at the Panda's "ready" joint pose, where park() goes
REFLEX_TEXT = "cartesian_reflex: collision detected, the arm is stopped until recovered"


class FakeFrankaDriver:
    """See the module docstring. `world` is a StaticWorld, a callable returning the current one (the
    conformance CLI swaps worlds between trials) or None (no blocks: the fingers close to zero)."""

    def __init__(self, world: StaticWorld | Callable[[], StaticWorld] | None = None,
                 clock: Callable[[], float] = time.perf_counter, *, fault_after_s: float | None = None,
                 start_xyz: tuple[float, float, float] = READY_XYZ, start_yaw: float = math.pi / 2):
        self._world: Callable[[], StaticWorld | None] = (lambda: None) if world is None else (world if callable(world) else (lambda: world))
        self._clock = clock
        self.t0 = clock()
        self._t = self.t0
        self.xyz = tuple(float(v) for v in start_xyz)
        self.yaw = float(start_yaw)
        self.width = MAX_WIDTH
        self.target: tuple[float, float, float] | None = None
        self.yaw_target = self.yaw
        self.speed = 0.0
        self.grip_target = MAX_WIDTH
        self.grip_force: float | None = None
        self.grasped = False
        self.error: str | None = None
        self.fault_after_s = fault_after_s
        self.calls: list[tuple] = []          # every lifecycle call in order: stop, recover, park, close
        self.targets_sent = 0
        self.parked = False
        self.closed = False

    # -- the driver surface -----------------------------------------------------------------------------
    def read(self) -> FrankaState:
        self._advance()
        moving = (self.target is not None and math.dist(self.xyz, self.target) > 1e-9) or abs(skills.wrap(self.yaw - self.yaw_target)) > 1e-9
        g_moving = abs(self.width - self.grip_target) > 1e-9 and not self.grasped
        return FrankaState(self._t, self.xyz, self.yaw, self.width, moving, self.error, self.grasped, g_moving)

    def set_target(self, xyz, yaw: float, speed: float) -> None:
        self._advance()
        if self.error is not None:
            return                                        # libfranka refuses motion in error; the server sees it in read()
        self.target = tuple(float(v) for v in xyz)
        self.yaw_target = float(yaw)
        self.speed = min(float(speed), MAX_SPEED)
        self.targets_sent += 1

    def set_gripper(self, width: float, force: float | None = None) -> None:
        self._advance()
        target = max(0.0, min(MAX_WIDTH, float(width)))
        if target != self.grip_target:                    # a repeat of the same width every tick is a no-op
            self.grip_target, self.grasped = target, False
        self.grip_force = force

    def stop(self) -> None:
        self._advance()
        self.calls.append(("stop",))
        self.target, self.yaw_target = None, self.yaw

    def recover(self) -> None:
        self._advance()
        self.calls.append(("recover",))
        self.error = None
        self.fault_after_s = None

    def park(self) -> None:
        self._advance()
        self.calls.append(("park",))
        self.target, self.yaw_target = None, self.yaw
        self.xyz, self.parked = READY_XYZ, True

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True

    # -- test knobs ---------------------------------------------------------------------------------------
    def inject_error(self, text: str = REFLEX_TEXT) -> None:
        """What a collision reflex does: the arm stops and will not move until recover()."""
        self._advance()
        self.error = text
        self.target, self.yaw_target = None, self.yaw

    # -- kinematics ---------------------------------------------------------------------------------------
    def _advance(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._t)
        self._t = now
        if self.fault_after_s is not None and self.error is None and now - self.t0 >= self.fault_after_s:
            self.error = REFLEX_TEXT
            self.target, self.yaw_target = None, self.yaw
        if dt <= 0.0:
            return
        if self.target is not None:
            d = math.dist(self.xyz, self.target)
            step = self.speed * dt
            if d <= step:
                self.xyz = self.target
            else:
                k = step / d
                self.xyz = tuple(a + (b - a) * k for a, b in zip(self.xyz, self.target))
        dy = skills.wrap(self.yaw_target - self.yaw)
        ystep = YAW_RATE * dt
        self.yaw = self.yaw_target if abs(dy) <= ystep else self.yaw + math.copysign(ystep, dy)
        self._advance_gripper(GRIP_RATE * dt)

    def _advance_gripper(self, step: float) -> None:
        w, target = self.width, self.grip_target
        if target >= w:
            self.width = min(target, w + step)
            return
        w_new = max(target, w - step)
        stop = self._block_width()
        if stop is not None and stop > target:
            if w_new <= stop:
                w_new = min(w, stop)
                self.grasped = True
        self.width = w_new

    def _block_width(self) -> float | None:
        world = self._world()
        if world is None:
            return None
        oid = world.under(self.xyz)
        return None if oid is None else world.pose(oid).size
