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
    rot: tuple[float, float, float] | None = None  # EE orientation, angle-axis, as the driver reports it
    holding: bool = False                          # something is between the closed fingers
    gripper_goal: float | None = None              # last commanded width (0 closed .. 0.04 open)


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

    def step(self, dt: float, max_dt: float = 0.075):
        """Advance the setpoint by at most speed_cap*min(dt, max_dt) toward the goal. The dt clamp
        keeps a late tick from turning into a large step. Returns the setpoint."""
        dt = min(dt, max_dt)
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
    """Trips when |F_ext| deviates from a slowly adapting baseline for several consecutive ticks.

    Efforts are not zero at rest (about (6, 1, 10) N at the hover start) and they wander by
    +-5 N in motion and jump ~16 N on a direction reversal (first_contact traces, 2026-09-17), so:
    the baseline is an EMA with a long time constant, the threshold sits above the reversal
    artifact, and a trip needs `persist` consecutive over-threshold ticks. This catches hard
    obstacles, not a paper cup."""

    def __init__(self, trip_n: float, baseline_s: float, tau_s: float = 3.0, persist: int = 3):
        self.trip_n = trip_n
        self.baseline_s = baseline_s
        self.tau = tau_s
        self.persist = persist
        self.samples = []
        self.baseline = None
        self.t_start = None
        self.t_last = None
        self.over = 0

    def reset(self):
        self.samples, self.baseline, self.t_start, self.t_last, self.over = [], None, None, None, 0

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
                self.t_last = now
            return None
        d = ((f[0] - self.baseline[0]) ** 2 + (f[1] - self.baseline[1]) ** 2 + (f[2] - self.baseline[2]) ** 2) ** 0.5
        self.over = self.over + 1 if d > self.trip_n else 0
        if d <= self.trip_n:   # adapt only while calm, so a real contact cannot be learned away
            dt = max(0.0, now - (self.t_last or now))
            a = min(1.0, dt / self.tau)
            self.baseline = tuple(self.baseline[i] + a * (f[i] - self.baseline[i]) for i in range(3))
        self.t_last = now
        return d

    def tripped(self, deviation: float | None) -> bool:
        return deviation is not None and self.over >= self.persist
