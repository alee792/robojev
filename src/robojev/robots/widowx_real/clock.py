"""The real server's clock: wall time on the arm, or lockstep virtual time for tests.

`WallClock` is what runs on hardware: `now()` is `time.perf_counter()`, `sleep()` is `time.sleep()`.

`LockstepClock` makes a test independent of how fast the machine is. Virtual seconds move only
when the thread that built the clock (the owner: the test, through `RealRobotServer.run_until`)
sleeps; every other thread that sleeps on it (RealArm's I/O thread) blocks until virtual time
reaches its wake time. The owner's `sleep(d)` walks virtual time forward d seconds, stopping at each
due wake time and waiting until the woken thread has done its work and gone back to sleep (or
ended), so the arm runs exactly at `real_tick_hz` of virtual time and the policy exactly at
`policy_hz`, in the same interleaving on every run, however loaded the machine is.

RealArm (arm/real.py, not edited) reads its time from its module's `time`; a test puts it on the
clock by pointing `robojev.arm.real.time` at `clock.time` for the test's duration (the
`server` fixture in tests/test_robots_widowx_real.py does it with monkeypatch). The fake driver and
the OrientingDriver take the clock's `now` directly.

`release()` turns a lockstep clock back into a wall clock from where virtual time stands (sleepers
wake, sleeps are real): STOP and `close()` call it, since RealArm.stop() joins its I/O thread on the
wall clock and parks with blocking driver calls.
"""
from __future__ import annotations

import threading
import time
import types

STUCK_S = 30.0          # wall seconds a woken thread may take to come back to sleep before the owner gives up


class WallClock:
    stepped = False

    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))

    def adopt(self, thread: threading.Thread | None) -> None:
        pass

    def release(self) -> None:
        pass


WALL = WallClock()


class LockstepClock:
    """See the module docstring. `stepped`: the server ticks its policies in `run_until`/`run_for`
    (on the owner's thread) instead of in its own thread."""
    stepped = True

    def __init__(self, start: float = 1000.0):
        self._cv = threading.Condition()
        self._now = float(start)
        self._owner = threading.current_thread()
        self._asleep: dict[threading.Thread, float] = {}     # thread -> virtual wake time
        self._running: set[threading.Thread] = set()         # woken (or adopted) and not yet back asleep
        self._released_at: float | None = None               # wall time of release()
        self.time = types.SimpleNamespace(perf_counter=self.now, monotonic=self.now, sleep=self.sleep,
                                          time=lambda: 1.7e9 + self.now())   # a stand-in for the `time` module

    @property
    def released(self) -> bool:
        return self._released_at is not None

    def now(self) -> float:
        with self._cv:
            if self._released_at is not None:
                return self._now + (time.perf_counter() - self._released_at)
            return self._now

    def adopt(self, thread: threading.Thread | None) -> None:
        """Count `thread` as running from now on, so the owner's next step waits for it to sleep:
        closes the race between a thread's start and its first sleep."""
        with self._cv:
            if thread is not None and thread is not self._owner and thread not in self._asleep:
                self._running.add(thread)

    def sleep(self, seconds: float) -> None:
        if self.released:
            time.sleep(max(0.0, seconds))
        elif threading.current_thread() is self._owner:
            self._advance(max(0.0, seconds))
        else:
            self._wait(max(0.0, seconds))

    def release(self) -> None:
        with self._cv:
            if self._released_at is None:
                self._released_at = time.perf_counter()
                self._asleep.clear()
                self._running.clear()
                self._cv.notify_all()

    # -- internals ---------------------------------------------------------------------------------
    def _wait(self, seconds: float) -> None:
        th = threading.current_thread()
        with self._cv:
            self._running.discard(th)
            self._asleep[th] = self._now + seconds
            self._cv.notify_all()
            while th in self._asleep and self._released_at is None:
                self._cv.wait()

    def _advance(self, seconds: float) -> None:
        with self._cv:
            target = self._now + seconds
            while True:
                self._quiesce()
                if self._released_at is not None:
                    return
                due = min(self._asleep.values(), default=None)
                if due is None or due > target:
                    break
                self._now = max(self._now, due)
                for th, wake in list(self._asleep.items()):
                    if wake <= self._now:
                        del self._asleep[th]
                        self._running.add(th)
                self._cv.notify_all()
            self._now = target

    def _quiesce(self) -> None:
        """Wait (holding the condition) until no woken thread is still working."""
        deadline = time.monotonic() + STUCK_S
        while self._released_at is None:
            self._running = {th for th in self._running if th.is_alive()}
            if not self._running:
                return
            if time.monotonic() > deadline:
                names = ", ".join(th.name for th in self._running)
                raise RuntimeError(f"lockstep clock: {names} did not come back to sleep within {STUCK_S:.0f} s")
            self._cv.wait(0.01)
