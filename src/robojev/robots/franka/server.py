"""The Franka (Panda / FR3) behind the robot protocol: `FrankaRobotServer`, the same manifest shape,
policies and control surface as the WidowX servers, over a `driver.FrankaDriver`.

Layers, top down:
  FrankaRobotServer  base.SkillServer (status, start/hold/pause/resume/retarget/heartbeat, subscribe,
                     the watchdog, the argument checks, the events) plus: the clock, a 50 Hz policy
                     thread, `world()` from the world source, the arm's own errors surfaced as
                     `safety_trip`, resume() recovering the arm, STOP = stop then park in a thread,
                     and the pre-flight gate.
  FrankaArmIO        the policies' ArmIO over the driver (arm.py): goals, yaw, width, `held()`.
  FrankaDriver       franky over libfranka on the RT control box (a later file, driver.py says how),
                     or fake_driver.FakeFrankaDriver here.

Time and tests: as RealRobotServer. The server's clock is wall time since `open()`; `tick()` runs
one policy period and the thread `open()` starts calls it at `policy_hz`. Tests pass a
`LockstepClock` (robots/widowx_real/clock.py, imported): no thread is started, `run_until`/
`run_for` tick the policies themselves and step virtual time, and the fake driver reads the same
virtual time, so every outcome is the same however loaded the machine is.

Shared with widowx_real, and how: the clock and the world (StaticWorld, load_blocks) are imported
from there unchanged. The lifecycle methods (`run_for`, `run_until`, `_step`, `finished`,
`set_world`, `set_hands`, the policy thread) and the pre-flight record are DUPLICATED from
RealRobotServer rather than hoisted into base.SkillServer: they are a screenful, and moving them
would change the sim server's tick contract for no gain today. If a third real server appears, that
is the moment to lift them.

What is not the WidowX's:
  - The arm's trips are libfranka's: a collision reflex, a Cartesian or joint limit, a lost
    heartbeat with the controller. The driver reports them in `read().error`; each new one becomes
    a `safety_trip` and holds the skill; the arm then reads `tripped` until `resume()`, which calls
    `driver.recover()` first. A reflex that fires again on the next motion trips again.
  - STOP: every skill fails "stopped", the policy loop ends, `driver.stop()` halts the motion at
    once (one libfranka control step), and `driver.park()` moves to the safe pose in a thread so
    this returns at once. The grip is kept. Terminal.
  - `preflight()` must be called before `open()` with four human checks acknowledged (FCI active in
    Desk, the activation device in hand, the workspace clear, the control box on the RT kernel):
    none of them is observable from here. The acknowledgement is recorded with the operator's name.
  - `close()` stops the motion and releases the connection; the arm stays where it is (libfranka
    holds position when control ends). It does not park: STOP does.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from robojev import catalog
from robojev.protocol import ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, Recorder, RobotEvent, WorldState
from robojev.robots.franka.arm import ARM_ID, BASE_FRAME, FRANKA_LIMITS, FRANKA_PARAMS, FrankaArmIO, FrankaLimits
from robojev.robots.franka.driver import FrankaDriver
from robojev.robots.widowx_real.clock import WALL, LockstepClock, WallClock
from robojev.robots.widowx_real.world import WorldSource
from robojev.robots.widowx_sim import skills
from robojev.robots.widowx_sim.base import SkillServer

ROBOT = "franka-1"
POLICY_HZ = 50
HOME_TIMEOUT_S = 15.0           # the move to the hover start at open(): 0.5 m at the speed cap plus franky's ramps


class PreflightError(RuntimeError):
    """open() refused: the day-one checks were not acknowledged."""


@dataclass(frozen=True)
class Preflight:
    """What a person checked before the arm is commanded. All four block: each is something libfranka
    needs or the operator's safety needs, and none can be read from software here."""
    operator: str
    fci_active: bool                # FCI enabled in Desk, joints unlocked, no other client connected
    stop_in_hand: bool              # the external activation device / stop button is in the operator's hand
    workspace_clear: bool           # nothing and no one inside the workspace box; the table as blocks.json says
    rt_kernel: bool                 # the control box runs the PREEMPT_RT kernel and communication_test passed
    notes: str = ""
    t_wall: float = field(default_factory=time.time)

    def problems(self) -> list[str]:
        out = []
        if not self.operator.strip():
            out.append("no operator named")
        if not self.fci_active:
            out.append("FCI is not acknowledged active in Desk (joints unlocked, no other client)")
        if not self.stop_in_hand:
            out.append("the external activation device / stop button is not acknowledged in hand")
        if not self.workspace_clear:
            out.append("the workspace is not acknowledged clear")
        if not self.rt_kernel:
            out.append("the control box is not acknowledged on the RT kernel with communication_test passed")
        return out


class FrankaRobotServer(SkillServer):
    """protocol.RobotServer on a Franka arm. See the module docstring."""

    def __init__(self, driver_factory: Callable[[], FrankaDriver], world_source: WorldSource, recorder: Recorder | None = None,
                 *, limits: FrankaLimits = FRANKA_LIMITS, params: skills.Params = FRANKA_PARAMS, heartbeat_timeout_s: float = 0.5,
                 policy_hz: float = POLICY_HZ, robot: str = ROBOT, home_timeout_s: float = HOME_TIMEOUT_S,
                 clock: WallClock | LockstepClock = WALL):
        super().__init__(recorder, params, heartbeat_timeout_s)
        self.clock = clock
        self.limits, self.robot = limits, robot
        self.world_src = world_source
        self.world_view = world_source
        self.table_z, self.travel_z, self.max_width = limits.table_z, limits.travel_z, limits.max_width
        self._driver_factory = driver_factory
        self.home_timeout_s = home_timeout_s
        self.tick_s = 1.0 / policy_hz
        self.driver: FrankaDriver | None = None
        self.arms: dict[str, FrankaArmIO] = {}
        self.preflight_record: Preflight | None = None
        self._t0: float | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._parking: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle (not protocol)
    def preflight(self, operator: str, *, fci_active: bool, stop_in_hand: bool, workspace_clear: bool, rt_kernel: bool,
                  notes: str = "") -> Preflight:
        """Record the human checks. `open()` refuses until this has been called with all four true."""
        self.preflight_record = Preflight(operator, fci_active, stop_in_hand, workspace_clear, rt_kernel, notes)
        note = getattr(self.recorder, "note", None)
        if note is not None:
            note(kind="preflight", **asdict(self.preflight_record))
        return self.preflight_record

    @property
    def t(self) -> float:
        return 0.0 if self._t0 is None else self.clock.now() - self._t0

    def open(self) -> "FrankaRobotServer":
        """Connect, move the fingertips to the hover start at travel height, then start the policy
        thread. Raises PreflightError without the acknowledgement, RuntimeError if the arm reports an
        error or does not reach the hover start in `home_timeout_s`."""
        if self.preflight_record is None:
            raise PreflightError("preflight() has not been called: FCI, the stop button, the workspace and the RT kernel "
                                 "must be acknowledged by a person first")
        problems = self.preflight_record.problems()
        if problems:
            raise PreflightError("preflight: " + "; ".join(problems))
        if self.driver is not None:
            return self
        self._t0 = self.clock.now()
        self.driver = self._driver_factory()
        io = FrankaArmIO(self.driver, self.world_src, self.limits, ARM_ID)
        hx, hy = self.limits.hover_xy
        io.goto((hx, hy, self.travel_z), self.limits.speed_cap, yaw=io.ee_yaw())
        io.grip(self.limits.max_width)
        deadline = self.clock.now() + self.home_timeout_s
        while True:
            s = io.refresh()
            if s.error is not None:
                self.driver.close()
                self.driver = None
                raise RuntimeError(f"the arm reports an error before it moved: {s.error}; recover it in Desk and start again")
            if io.arrived():
                break
            if self.clock.now() > deadline:
                self.driver.stop()
                self.driver.close()
                self.driver = None
                raise RuntimeError(f"the arm did not reach the hover start ({hx:.2f}, {hy:.2f}, {self.travel_z:.2f}) within "
                                   f"{self.home_timeout_s:.0f} s: it is at {tuple(round(v, 3) for v in s.xyz)}")
            self.clock.sleep(self.tick_s)
        self.arms = {ARM_ID: io}
        if not self.clock.stepped:
            self._thread = threading.Thread(target=self._loop, name="franka-robot", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop the policy thread and the motion; release the driver. The arm stays put."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._parking is not None:
            self._parking.join(timeout=30.0)
        if self.driver is not None:
            if self._parking is None:
                self.driver.stop()
            self.driver.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            self.tick()
            next_t += self.tick_s
            ahead = next_t - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
            else:
                next_t = time.perf_counter()

    def tick(self) -> None:
        """One policy period. Events go out after the lock is released."""
        with self._lock:
            evs = self._tick()
        self._emit(evs)

    def run_for(self, seconds: float) -> None:
        if not self.clock.stepped:
            time.sleep(max(0.0, seconds))
            return
        end = self.clock.now() + seconds
        while self.clock.now() < end:
            self._step()

    def run_until(self, cond: Callable[[], bool], timeout_s: float) -> bool:
        """Wait until `cond()` holds or `timeout_s` has passed (on the server's clock). True if it held.
        On a stepped clock this ticks the policies itself, one period per check."""
        deadline = self.clock.now() + timeout_s
        while self.clock.now() < deadline:
            if cond():
                return True
            if self.clock.stepped:
                self._step()
            else:
                time.sleep(self.tick_s)
        return cond()

    def _step(self) -> None:
        if not self._stop_evt.is_set():
            self.tick()
        self.clock.sleep(self.tick_s)

    def finished(self, skill_id: str) -> Callable[[], bool]:
        return lambda: self._runs[skill_id].finished

    def set_world(self, world_source: WorldSource) -> None:
        """Swap the world between conformance trials (a person has re-laid the blocks). Refused
        while a skill runs."""
        with self._lock:
            if any(a.active for a in self.arms.values()):
                raise ValueError("a skill is running; the world cannot change under it")
            self.world_src = self.world_view = world_source
            for a in self.arms.values():
                a.world = world_source

    def set_hands(self, hands: list[HandObs]) -> None:
        """What world().hands and the policies' hands() report from now on (no hand perception yet)."""
        with self._lock:
            self.world_src.hands = list(hands)

    # ---------------------------------------------------------------- protocol: what this server adds
    def manifest(self) -> Manifest:
        ws = self.limits.workspace
        arm = ArmSpec(ARM_ID, BASE_FRAME, (ws.x, ws.y, ws.z), GripperSpec(max_width=self.max_width, can_grasp=True),
                      travel_z=self.travel_z,
                      description="Franka arm (Panda / FR3) over FCI, 7 joints plus the Franka Hand parallel gripper, "
                                  "grasping straight down; object positions are typed in and tracked by the server's own "
                                  "bookkeeping, not observed")
        return Manifest(self.robot, (arm,), skills.SPECS, catalog=catalog.CATALOG_VERSION)

    def world(self) -> WorldState:
        with self._lock:
            arms = {}
            for a in self.arms.values():
                if self._stopped:
                    a.refresh()                      # no tick refreshes it after STOP; report where the park left it
                x, y, z = a.ee()
                arms[a.id] = ArmObs(a.id, x, y, z, a.held(), a.width(), a.mode(), skill=a.skill.id if a.active else None)
            return WorldState(self.t, self.world_src.objects(self._held_by()), dict(self.world_src.places),
                              list(self.world_src.hands), arms)

    def resume(self, arm: str) -> None:
        """As the base, after recovering the arm from its error state: resume() is the only way out of
        a trip, and on a Franka the arm itself must be told it may move again."""
        with self._lock:
            a = self._arm(arm)
            if not a.stopped and a.snap().error is not None and self.driver is not None:
                self.driver.recover()
                a.refresh()
                a.error_seen = None
            super().resume(arm)

    def stop(self) -> None:
        """STOP: every skill fails "stopped", the policy loop ends, the motion stops at once and the
        arm parks (grip kept) in a thread so this returns at once. Terminal."""
        evs: list[RobotEvent] = []
        with self._lock:
            if self._stopped:
                return
            self._fail_all("stopped: STOP pressed; the arm is parking", evs)
            self._stop_evt.set()
            if self.driver is not None and self._parking is None:
                self.driver.stop()
                self._parking = threading.Thread(target=self._park, name="franka-robot-park", daemon=True)
                self._parking.start()
        self._emit(evs)

    def _park(self) -> None:
        note = getattr(self.recorder, "note", None)
        try:
            self.driver.park()
            if note is not None:
                note(kind="parked", t=self.t)
        except Exception as e:          # a parking move the arm refuses (in error) is reported, not raised into STOP
            if note is not None:
                note(kind="park_failed", t=self.t, text=str(e))

    # ---------------------------------------------------------------- internals
    def _tick(self) -> list[RobotEvent]:
        evs: list[RobotEvent] = []
        if self._stopped:
            return evs
        for a in self.arms.values():
            s = a.refresh()
            why = a.new_error()
            if why:
                self._trip(a, f"the arm reported {why}", {"error": s.error, "ee": list(s.xyz)}, evs)
        self._watchdog(evs)
        self._tick_skills(evs)
        for a in self.arms.values():
            self.world_src.track(a.ee(), a.held(), a.width(), self.params)
        return evs
