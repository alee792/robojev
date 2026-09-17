"""Arm backends behind one interface.

Base frame everywhere in robojev: +x forward, +y left, +z up, metres. The EE orientation is fixed
pointing down; only position and gripper width are commanded. Each backend runs its own I/O thread
at its own rate and exposes thread-safe snapshots; the Jev loop never calls a driver directly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ArmSnapshot:
    t: float
    ee: tuple[float, float, float]                 # EE position, base frame
    gripper: float                                 # width, metres, 0 closed .. 0.04 open
    joints: tuple[float, ...] = ()
    ext_force: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Cartesian external force estimate (N)
    setpoint: tuple[float, float, float] = (0.0, 0.0, 0.0)   # what the backend last commanded
    goal: tuple[float, float, float] = (0.0, 0.0, 0.0)       # where the mover is heading
    speed_cap: float = 0.0
    frozen: bool = False
    status: str = "init"                           # init | staging | live | frozen | parking | stopped | error
    error: str | None = None


class ArmBackend(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> ArmSnapshot: ...
    def command(self, goal: tuple[float, float, float], speed_cap: float, gripper: float | None = None) -> None: ...
    def freeze(self, reason: str) -> None: ...
    def resume(self) -> None: ...


class Mover:
    """Velocity-capped setpoint that walks toward a goal. Runs inside the backend thread.

    Goals may jump every tick; the setpoint never does. Everything is clamped to the workspace
    box. A frozen mover holds its setpoint until resumed.
    """

    def __init__(self, workspace, hard_speed_cap: float):
        self.ws = workspace
        self.hard_cap = hard_speed_cap
        self.lock = threading.Lock()
        self.goal = None
        self.setpoint = None
        self.speed_cap = 0.0
        self.frozen = False
        self.freeze_reason = ""

    def init_at(self, p):
        with self.lock:
            self.setpoint = tuple(self.ws.clamp(p))
            self.goal = self.setpoint

    def set_goal(self, goal, speed_cap):
        with self.lock:
            self.goal = tuple(self.ws.clamp(goal))
            self.speed_cap = max(0.0, min(speed_cap, self.hard_cap))

    def freeze(self, reason):
        with self.lock:
            self.frozen, self.freeze_reason = True, reason
            if self.setpoint is not None:
                self.goal = self.setpoint

    def resume(self):
        with self.lock:
            self.frozen, self.freeze_reason = False, ""

    def step(self, dt: float):
        """Advance the setpoint by at most speed_cap*dt toward the goal. Returns the setpoint."""
        with self.lock:
            if self.setpoint is None or self.goal is None or self.frozen:
                return self.setpoint
            dx = [g - s for g, s in zip(self.goal, self.setpoint)]
            dist = (dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2) ** 0.5
            max_step = self.speed_cap * dt
            if dist <= max_step or dist < 1e-6:
                self.setpoint = self.goal
            else:
                k = max_step / dist
                self.setpoint = tuple(self.ws.clamp((self.setpoint[0] + dx[0] * k,
                                                     self.setpoint[1] + dx[1] * k,
                                                     self.setpoint[2] + dx[2] * k)))
            return self.setpoint


class EffortWatchdog:
    """Trips when |F_ext| deviates from a baseline learned at rest. Efforts are not zero at rest
    (idle read showed about -33, -6, -25 N), so only deviation is meaningful."""

    def __init__(self, trip_n: float, baseline_s: float):
        self.trip_n = trip_n
        self.baseline_s = baseline_s
        self.samples = []
        self.baseline = None
        self.t_start = None

    def reset(self):
        self.samples, self.baseline, self.t_start = [], None, None

    def update(self, force_xyz, now: float) -> float | None:
        """Returns the deviation in N once a baseline exists, else None."""
        f = force_xyz
        if self.baseline is None:
            if self.t_start is None:
                self.t_start = now
            self.samples.append(f)
            if now - self.t_start >= self.baseline_s and len(self.samples) >= 5:
                n = len(self.samples)
                self.baseline = tuple(sum(s[i] for s in self.samples) / n for i in range(3))
            return None
        d = ((f[0] - self.baseline[0]) ** 2 + (f[1] - self.baseline[1]) ** 2 + (f[2] - self.baseline[2]) ** 2) ** 0.5
        return d

    def tripped(self, deviation: float | None) -> bool:
        return deviation is not None and deviation > self.trip_n
