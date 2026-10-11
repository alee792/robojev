"""The real WidowX AI follower behind the robot protocol: `RealRobotServer`, with the same manifest
shape, policies and control surface as the sim (`SimRobotServer`), over v1's `RealArm`.

Layers, top down:
  RealRobotServer  base.SkillServer (status, start/hold/pause/resume/retarget/heartbeat, subscribe,
                   the watchdog, the argument checks, the events) plus: the wall clock, a 50 Hz policy
                   thread, `world()` from the world source, the arm's own trips surfaced as
                   `safety_trip`, STOP = `RealArm.stop()` (parks), and the pre-flight gate.
  RealArmIO        the policies' ArmIO over RealArm (arm.py): snapshot reads, Mover goals, the
                   OrientingDriver for yaw, gripper width <-> carriage position.
  RealArm          v1 (arm/real.py, not edited): its own 20 Hz I/O thread, the Mover (workspace clamp,
                   speed cap), the EffortWatchdog, the 3 cm lag trip, the thermal governor, parking.
  trossen_arm      the driver, or fake_driver.FakeTrossenDriver here.

Time: the server's clock is wall time since `open()`. `tick()` runs one policy period; the thread
`open()` starts calls it at `policy_hz`. `run_until`/`run_for` wait on the wall clock, so
conformance's trial loop reads the same on both servers.

What is not the sim's, and why:
  - `world()` is bookkeeping, not observation (world.StaticWorld): the arm is trusted to have done
    what it was told. Nothing here can see a dropped block that the gripper did not notice.
  - `stop()` returns at once and parks in a thread: `RealArm.stop()` joins its I/O thread and runs
    two blocking joint moves (6 s on hardware). The policy loop stops; status() keeps answering.
  - The arm's I/O thread dying (status "error") ends every skill failed with "stopped: ..." and the
    server reads stopped: RealArm has already parked itself.
  - `preflight()` must be called with the teleop link acknowledged off before `open()`: in the bay
    the leader's motions drive the follower unless the teleop process is down (docs/widowx-pair.md),
    and no code here can check that. The acknowledgement is recorded with the operator's name.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from robojev import catalog
from robojev.arm.real import RealArm
from robojev.config import Config
from robojev.protocol import ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, Recorder, RobotEvent, WorldState
from robojev.robots.widowx_real.arm import (ARM_ID, BASE_FRAME, MAX_WIDTH, REAL_LIMITS, REAL_PARAMS, TRAVEL_ABOVE_TABLE,
                                            OrientingDriver, RealArmIO, RealLimits)
from robojev.robots.widowx_real.world import WorldSource
from robojev.robots.widowx_sim import skills
from robojev.robots.widowx_sim.base import SkillServer

ROBOT = "widowx-bay-1"
POLICY_HZ = 50
LIVE_TIMEOUT_S = 20.0           # staging (two blocking moves) plus the effort baseline (1 s) on hardware


class PreflightError(RuntimeError):
    """open() refused: the day-one checks were not acknowledged."""


@dataclass(frozen=True)
class Preflight:
    """What a person checked before the arm is commanded (docs/day-one.md). `teleop_off` is the one
    that blocks: with the kit's teleop process running, the leader drives the follower."""
    operator: str
    teleop_off: bool
    estop_tested: bool = False
    workspace_set: bool = False
    notes: str = ""
    t_wall: float = field(default_factory=time.time)

    def problems(self) -> list[str]:
        out = []
        if not self.operator.strip():
            out.append("no operator named")
        if not self.teleop_off:
            out.append("the leader's teleop link is not acknowledged off: the leader would drive the follower")
        return out


class RealRobotServer(SkillServer):
    """protocol.RobotServer on the real follower. See the module docstring."""

    def __init__(self, ip: str, world_source: WorldSource, cfg: Config, recorder: Recorder | None = None,
                 driver_factory: Callable | None = None, *, heartbeat_timeout_s: float = 0.5,
                 params: skills.Params = REAL_PARAMS, limits: RealLimits = REAL_LIMITS, policy_hz: float = POLICY_HZ,
                 robot: str = ROBOT, arrive_tol: float | None = None, live_timeout_s: float = LIVE_TIMEOUT_S):
        super().__init__(recorder, params, heartbeat_timeout_s)
        self.ip, self.cfg, self.limits, self.robot = ip, cfg, limits, robot
        self.world_src = world_source
        self.world_view = world_source
        self.table_z, self.travel_z, self.max_width = cfg.table_z, cfg.table_z + TRAVEL_ABOVE_TABLE, MAX_WIDTH
        self._driver_factory = driver_factory or RealArm._default_factory
        self._arrive_tol = arrive_tol
        self.live_timeout_s = live_timeout_s
        self.tick_s = 1.0 / policy_hz
        self.arm: RealArm | None = None
        self.orient: OrientingDriver | None = None
        self.arms: dict[str, RealArmIO] = {}
        self.preflight_record: Preflight | None = None
        self._t0: float | None = None
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._parking: threading.Thread | None = None
        self._arm_ev_seen: tuple | None = None

    # ---------------------------------------------------------------- lifecycle (not protocol)
    def preflight(self, operator: str, *, teleop_off: bool, estop_tested: bool = False, workspace_set: bool = False,
                  notes: str = "") -> Preflight:
        """Record the human checks. `open()` refuses until this has been called with `teleop_off`."""
        self.preflight_record = Preflight(operator, teleop_off, estop_tested, workspace_set, notes)
        note = getattr(self.recorder, "note", None)
        if note is not None:
            note(kind="preflight", **asdict(self.preflight_record))
        return self.preflight_record

    @property
    def t(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def open(self) -> "RealRobotServer":
        """Connect, stage (RealArm.start: joint move to STAGED, blocking move to the hover start),
        wait for the effort baseline ("live"), then start the policy thread. Raises PreflightError
        without the acknowledgement, RuntimeError if the arm never reads live or could not reach the
        hover orientation (RealArm then never streams pitch, and nothing here could grasp)."""
        if self.preflight_record is None:
            raise PreflightError("preflight() has not been called: see docs/day-one.md")
        problems = self.preflight_record.problems()
        if problems:
            raise PreflightError("preflight: " + "; ".join(problems))
        if self.arm is not None:
            return self
        self._t0 = time.monotonic()

        def factory(ip: str):
            self.orient = OrientingDriver(self._driver_factory(ip), self.limits.yaw_rate)
            return self.orient

        self.arm = RealArm(self.cfg, self.ip, driver_factory=factory)
        self.arm.start()
        if not self.arm.pitch_ok:
            self.arm.stop()
            raise RuntimeError("the arm did not reach the hover orientation; RealArm would stream the reached one and never "
                               "the 90 deg pitch the grasps need: " + "; ".join(t for _, t in self.arm.events))
        deadline = time.monotonic() + self.live_timeout_s
        while self.arm.snapshot().status != "live":
            s = self.arm.snapshot()
            if s.status in ("error", "stopped"):
                raise RuntimeError(f"the arm's I/O thread ended during staging: {s.status} {s.error or ''}".strip())
            if time.monotonic() > deadline:
                self.arm.stop()
                raise RuntimeError(f"the arm did not read live within {self.live_timeout_s:.0f} s (status {s.status})")
            time.sleep(0.02)
        io = RealArmIO(self.arm, self.orient, self.world_src, self.cfg, self.limits, ARM_ID, self._arrive_tol)
        self.arms = {ARM_ID: io}
        self._record_arm_events()
        self._thread = threading.Thread(target=self._loop, name="real-robot", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the policy thread and park the arm (if STOP has not already)."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._parking is not None:
            self._parking.join(timeout=10.0)
        elif self.arm is not None and self.arm.driver is not None:
            self.arm.stop()

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
        time.sleep(max(0.0, seconds))

    def run_until(self, cond: Callable[[], bool], timeout_s: float) -> bool:
        """Wait (wall clock) until `cond()` holds or `timeout_s` has passed. True if it held."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(self.tick_s)
        return cond()

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
        """What world().hands and the policies' hands() report from now on (no hand perception on day one)."""
        with self._lock:
            self.world_src.hands = list(hands)

    # ---------------------------------------------------------------- protocol: what the real server adds
    def manifest(self) -> Manifest:
        ws = self.cfg.workspace
        arm = ArmSpec(ARM_ID, BASE_FRAME, (ws.x, ws.y, ws.z), GripperSpec(max_width=MAX_WIDTH, can_grasp=True),
                      travel_z=self.travel_z,
                      description=f"WidowX AI follower at {self.ip} over trossen_arm (v1 RealArm), 6 joints plus a parallel "
                                  "gripper, grasping straight down; object positions are typed in and tracked by the "
                                  "server's own bookkeeping, not observed")
        return Manifest(self.robot, (arm,), skills.SPECS, catalog=catalog.CATALOG_VERSION)

    def world(self) -> WorldState:
        with self._lock:
            held = self._held_by()
            arms = {}
            for a in self.arms.values():
                x, y, z = a.ee()
                arms[a.id] = ArmObs(a.id, x, y, z, a.held(), a.width(), a.mode(), skill=a.skill.id if a.active else None)
            return WorldState(self.t, self.world_src.objects(held), dict(self.world_src.places), list(self.world_src.hands), arms)

    def stop(self) -> None:
        """STOP: every skill fails "stopped", the policy loop ends, RealArm parks (STAGED then SLEEP,
        grip unchanged) in a thread so this returns at once. Terminal."""
        evs: list[RobotEvent] = []
        with self._lock:
            if self._stopped:
                return
            self._fail_all("stopped: STOP pressed; the arm is parking", evs)
            self._stop_evt.set()
            if self.arm is not None and self._parking is None:
                self._parking = threading.Thread(target=self.arm.stop, name="real-robot-park", daemon=True)
                self._parking.start()
        self._emit(evs)

    # ---------------------------------------------------------------- internals
    def _tick(self) -> list[RobotEvent]:
        evs: list[RobotEvent] = []
        if self._stopped:
            return evs
        for a in self.arms.values():
            s = a.snap()
            if s.status == "error":
                self._fail_all(f"stopped: the arm's I/O thread died ({s.error}); it has parked itself", evs)
                self._stop_evt.set()
                return evs
            why = a.trip_reason()                     # the arm's own thread froze the Mover: effort, lag, thermal
            if why:
                self._trip(a, why, {"ext_force": list(s.ext_force), "lag_m": math.dist(s.ee, s.setpoint)}, evs)
        self._watchdog(evs)
        self._tick_skills(evs)
        for a in self.arms.values():
            self.world_src.track(a.ee(), a.held(), a.width(), self.params)
        self._record_arm_events()
        return evs

    def _record_arm_events(self) -> None:
        """RealArm's own log lines (freezes, gripper stops, warnings) into the recorder as notes."""
        note = getattr(self.recorder, "note", None)
        if note is None or self.arm is None:
            return
        evs = list(self.arm.events)
        start = 0
        if self._arm_ev_seen is not None:
            try:
                start = evs.index(self._arm_ev_seen) + 1
            except ValueError:
                start = 0
        for t_wall, text in evs[start:]:
            note(kind="arm_event", t=self.t, t_wall=t_wall, text=text)
        if evs:
            self._arm_ev_seen = evs[-1]

