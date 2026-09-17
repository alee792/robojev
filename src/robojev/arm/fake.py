"""Kinematic fake: no physics, no driver. The setpoint is the pose. For tests and dry runs."""
from __future__ import annotations

import threading
import time

from robojev.arm import ArmSnapshot, Mover
from robojev.config import Config


class FakeArm:
    def __init__(self, cfg: Config, start_at=(0.25, 0.0, 0.20), rate_hz: float = 100.0):
        self.cfg = cfg
        self.mover = Mover(cfg.workspace, cfg.motion.hard_speed_cap)
        self.mover.init_at(start_at)
        self.gripper = 0.04
        self.rate = rate_hz
        self._stop = threading.Event()
        self._thread = None
        self._snap = ArmSnapshot(time.time(), tuple(start_at), self.gripper, status="init")
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        dt = 1 / self.rate
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            sp = self.mover.step(now - last)
            last = now
            with self._lock:
                self._snap = ArmSnapshot(time.time(), sp, self.gripper, setpoint=sp, goal=self.mover.goal,
                                         speed_cap=self.mover.speed_cap, frozen=self.mover.frozen, status="live",
                                         rot=self.cfg.motion.down_orientation)
            time.sleep(dt)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def snapshot(self) -> ArmSnapshot:
        with self._lock:
            return self._snap

    def command(self, goal, speed_cap, gripper=None):
        self.mover.set_goal(goal, speed_cap)
        if gripper is not None:
            self.gripper = max(0.0, min(0.04, gripper))

    def freeze(self, reason):
        self.mover.freeze(reason)

    def resume(self):
        self.mover.resume()
