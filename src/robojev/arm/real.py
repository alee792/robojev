"""Real WidowX AI follower over the trossen_arm driver, streamed from its own thread.

Guarantees, in order of importance:
  * The arm parks (STAGED then SLEEP, the same sequence the bay's ArmSession uses) on stop, on
    any exception in the I/O thread, and on KeyboardInterrupt in the main thread.
  * Every commanded setpoint is inside the workspace box and moves no faster than the speed cap;
    the driver is only ever asked to reach a point at most (cap * goal_time) away.
  * An external-force deviation beyond the trip threshold freezes the setpoint; resume is manual.
  * The driver connection is exclusive: while this runs, nothing else may use the follower.

The setpoint is re-sent every tick with a short goal_time, so the controller interpolates linearly
between consecutive setpoints (goal_time in (0.001, 0.2] selects linear interpolation).
"""
from __future__ import annotations

import math
import threading
import time
import traceback

from robojev.arm import ArmSnapshot, EffortWatchdog, Mover
from robojev.config import Config

FOLLOWER_IP = "192.168.1.5"
SLEEP = [0.0] * 7
STAGED = [0.0, math.pi / 3, math.pi / 6, math.pi / 5, 0.0, 0.0, 0.0]
PARK = [(3.0, STAGED), (3.0, SLEEP)]


class RealArm:
    def __init__(self, cfg: Config, ip: str = FOLLOWER_IP, driver_factory=None, log=None):
        self.cfg = cfg
        self.ip = ip
        self._driver_factory = driver_factory or self._default_factory
        self.log = log
        self.mover = Mover(cfg.workspace, cfg.motion.hard_speed_cap)
        self._grip_mode = "position"
        self._relaxed = False
        self._temps: tuple[float, ...] = ()
        self._temp_n = 0
        self._resting = False
        self.pitch_ok = False   # True when the arm reached the staged orientation exactly: then pitch may be streamed
        self.watchdog = EffortWatchdog(cfg.safety.effort_trip_n, cfg.safety.effort_baseline_s, persist=cfg.safety.effort_persist_ticks)
        self._lag_over = 0
        self.driver = None
        self._gripper_goal = None
        self._gripper_sent = None
        self._gripper_t = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._snap = ArmSnapshot(time.time(), (0, 0, 0), 0.0, status="init")
        self.events = []  # (t, text) recent events for the dashboard
        self.last_step_m = 0.0
        self.stream_orient = list(cfg.motion.down_orientation)
        self.max_step_m = 0.0   # largest setpoint delta ever sent in one tick (speed-cap evidence)

    @staticmethod
    def _default_factory(ip):
        import trossen_arm
        d = trossen_arm.TrossenArmDriver()
        d.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_follower, ip, True)
        return d

    def _event(self, text):
        self.events.append((time.time(), text))
        self.events = self.events[-50:]
        if self.log:
            self.log.write("events", kind="arm", text=text)

    # -- lifecycle -----------------------------------------------------------------------
    def start(self):
        import trossen_arm
        self._set_status("connecting")
        self.driver = self._driver_factory(self.ip)
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self._set_status("staging")
        # from sleep (folded) to STAGED is a joint move; never start Cartesian streaming from sleep
        self.driver.set_all_positions(list(STAGED), 2.0, True)
        self.driver.set_gripper_position(0.04, 1.0, True)
        self._gripper_sent = 0.04
        # From STAGED, one slow blocking move to the hover-start pose, pointing down. Streaming
        # never starts from a pose whose orientation differs from the streamed one.
        start = list(self.cfg.motion.hover_start) + [self.cfg.table_z + self.cfg.motion.safe_height]
        start = list(self.cfg.workspace.clamp(start))
        survey = [0.0, self.cfg.motion.hover_pitch, 0.0]
        self.driver.set_cartesian_positions(start + survey,
                                            trossen_arm.InterpolationSpace.joint, 2.5, True)
        pose = list(self.driver.get_cartesian_positions())
        self.mover.init_at(pose[:3], self.cfg.motion.hover_pitch)
        want = survey
        got = pose[3:6]
        miss = math.dist(start, pose[:3]); rmiss = math.dist(want, got)
        self._event(f"hover start at {[round(v, 3) for v in pose[:3]]} rot {[round(v, 2) for v in got]}"
                    f" (asked {[round(v, 3) for v in start]}, miss {miss*1000:.0f} mm / {rmiss:.2f} rad)")
        # stream the orientation the arm actually reached, never one it could not: fighting an
        # unreachable orientation showed up as a 68 N phantom force on the first run
        self.stream_orient = got if rmiss > 0.05 else want
        self.pitch_ok = rmiss <= 0.05
        if rmiss > 0.05:
            self._event(f"WARNING: streaming the reached orientation {[round(v, 2) for v in got]} instead of {want}")
        if miss > 0.02:
            self._event(f"WARNING: hover start missed by {miss*1000:.0f} mm; pose may be unreachable")
        self.watchdog.reset()
        self._thread = threading.Thread(target=self._run, daemon=True, name="arm-io")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._park_and_release()

    def _park_and_release(self):
        if self.driver is None:
            return
        self._set_status("parking")
        import trossen_arm
        try:
            if self._grip_mode != "position":
                self.driver.set_gripper_mode(trossen_arm.Mode.position); self._grip_mode = "position"
        except Exception as e:
            self._event(f"gripper mode reset failed: {e!r}")
        try:
            for goal_time, goal in PARK:
                self.driver.set_all_positions(list(goal), goal_time, True)
        except Exception as e:
            self._event(f"park failed: {e}")
        finally:
            try:
                self.driver.cleanup()
            except Exception:
                pass
            self.driver = None
            self._set_status("stopped")

    # -- I/O thread ------------------------------------------------------------------------
    def _run(self):
        import trossen_arm
        dt = 1 / self.cfg.motion.real_tick_hz
        goal_time = self.cfg.motion.real_goal_time
        orient = list(self.stream_orient)
        space = trossen_arm.InterpolationSpace.cartesian
        next_t = time.perf_counter()
        last = next_t
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                pose = list(self.driver.get_cartesian_positions())
                joints = tuple(self.driver.get_all_positions())
                fx, fy, fz = list(self.driver.get_cartesian_external_efforts())[:3]
                dev = self.watchdog.update((fx, fy, fz), now)
                if self.watchdog.tripped(dev) and not self.mover.frozen:
                    self.mover.freeze(f"external force deviation {dev:.1f} N")
                    self._event(f"FROZEN: force deviation {dev:.1f} N (baseline {self.watchdog.baseline})")
                sp_now = self.mover.setpoint
                lag = math.dist(pose[:3], sp_now) if sp_now is not None else 0.0
                self._lag_over = self._lag_over + 1 if lag > self.cfg.safety.lag_trip_m else 0
                if self._lag_over >= self.cfg.safety.lag_persist_ticks and not self.mover.frozen:
                    self.mover.freeze(f"tracking lag {lag*1000:.0f} mm")
                    self._event(f"FROZEN: EE lags setpoint by {lag*1000:.0f} mm (blocked?)")
                if self.mover.frozen and not self._relaxed:
                    # stop fighting whatever stopped us: command the pose the arm is actually at, once
                    # (holding the old setpoint against an obstruction is how a motor overheats)
                    self.driver.set_cartesian_positions(pose, space, 0.5, False)
                    self._relaxed = True
                elif not self.mover.frozen:
                    self._relaxed = False
                # thermal governor: read rotor temperatures about once a second
                self._temp_n += 1
                if self._temp_n % 20 == 0:
                    try:
                        self._temps = tuple(float(v) for v in self.driver.get_all_rotor_temperatures())
                    except Exception as e:
                        self._event(f"temperature read failed: {e!r}")
                    hot = max(self._temps) if self._temps else 0.0
                    if hot > self.cfg.safety.temp_rest_c and not self._resting:
                        self._resting = True; self.mover.freeze(f"motor at {hot:.0f} C: resting")
                        self._event(f"RESTING: a rotor is at {hot:.0f} C (limit 95); holding until below {self.cfg.safety.temp_slow_c:.0f}")
                    elif self._resting and hot < self.cfg.safety.temp_slow_c:
                        self._resting = False; self.mover.resume(); self._event(f"rotors cooled to {hot:.0f} C: resuming")
                    self.mover.thermal_cap = 0.03 if hot > self.cfg.safety.temp_slow_c else None
                if self.watchdog.baseline is None or self.mover.frozen:
                    sp = self.mover.setpoint          # hold: nothing is sent while baselining or frozen
                    last = now
                else:
                    prev = self.mover.setpoint
                    sp = self.mover.step(now - last)
                    last = now
                    step_m = math.dist(prev, sp) if prev is not None else 0.0
                    self.last_step_m = step_m
                    self.max_step_m = max(self.max_step_m, step_m)
                    if self.pitch_ok and self.mover.pitch is not None:
                        orient = [0.0, float(self.mover.pitch), 0.0]   # angle-axis about base y
                    self.driver.set_cartesian_positions(list(sp) + orient, space, goal_time, False)
                g = self._gripper_goal
                if g is not None and g != self._gripper_sent:
                    if g < 0.036:
                        # close by force, not position: the fingers stop on the object and press with
                        # grip_force_n instead of crushing a paper cup toward a position target
                        if self._grip_mode != "effort":
                            self.driver.set_gripper_mode(trossen_arm.Mode.external_effort); self._grip_mode = "effort"
                        self.driver.set_gripper_external_effort(-abs(self.cfg.motion.grip_force_n), 0.3, False)
                    else:
                        if self._grip_mode != "position":
                            self.driver.set_gripper_mode(trossen_arm.Mode.position); self._grip_mode = "position"
                        self.driver.set_gripper_position(float(g), 0.5, False)
                    self._gripper_sent = g
                    self._gripper_t = time.perf_counter()
                # holding: asked to close, and the fingers stopped well short of closed
                g = self._gripper_goal
                # holding: asked to close (to any target) and the fingers stopped short of it on something
                holding = g is not None and g < 0.036 and joints[6] > 0.006 and (time.perf_counter() - self._gripper_t) > 0.5
                with self._lock:
                    self._snap = ArmSnapshot(time.time(), tuple(pose[:3]), joints[6], joints=joints,
                                             ext_force=(fx, fy, fz), setpoint=sp or tuple(pose[:3]),
                                             holding=holding, gripper_goal=g,
                                             goal=self.mover.goal or tuple(pose[:3]), speed_cap=self.mover.speed_cap,
                                             frozen=self.mover.frozen, rot=tuple(pose[3:6]), pitch=self.mover.pitch, temps=self._temps,
                                             status="frozen" if self.mover.frozen else ("live" if self.watchdog.baseline else "baselining"))
                next_t += dt
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.perf_counter()
        except Exception as e:
            self._event(f"arm thread error: {e!r}")
            traceback.print_exc()
            with self._lock:
                self._snap.status, self._snap.error = "error", repr(e)
            self._stop.set()
            self._park_and_release()

    def _set_status(self, s):
        with self._lock:
            self._snap.status = s

    # -- interface ---------------------------------------------------------------------------
    def snapshot(self) -> ArmSnapshot:
        with self._lock:
            return self._snap

    def command(self, goal, speed_cap, gripper=None, pitch=None):
        self.mover.set_goal(goal, speed_cap, pitch)
        if gripper is not None:
            self._gripper_goal = max(0.0, min(0.04, float(gripper)))

    def freeze(self, reason):
        self.mover.freeze(reason)
        self._event(f"freeze: {reason}")

    def resume(self):
        self.mover.resume()
        self._event("resume")


class RealArmReadOnly(RealArm):
    """Reads the real arm's pose in a thread; never sets a mode or sends a command. Commands are
    recorded in the snapshot's goal so the dashboard shows what *would* be sent."""

    @staticmethod
    def _default_factory(ip):
        import trossen_arm
        d = trossen_arm.TrossenArmDriver()
        d.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_follower, ip, True)   # a previous run's trip leaves an error on the controller; reading is harmless
        return d

    def start(self):
        self.driver = self._driver_factory(self.ip)
        pose = list(self.driver.get_cartesian_positions())
        self.mover.init_at(pose[:3])
        self._event(f"read-only: arm at {[round(v, 3) for v in pose[:3]]}, no commands will be sent")
        self._thread = threading.Thread(target=self._run, daemon=True, name="arm-ro")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.driver is not None:
            try:
                self.driver.cleanup()
            except Exception:
                pass
            self.driver = None
        self._set_status("stopped")

    def _run(self):
        dt = 1 / self.cfg.motion.real_tick_hz
        last = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                pose = list(self.driver.get_cartesian_positions())
                joints = tuple(self.driver.get_all_positions())
                fx, fy, fz = list(self.driver.get_cartesian_external_efforts())[:3]
                sp = self.mover.step(now - last)   # the mover walks, the arm does not
                last = now
                with self._lock:
                    self._snap = ArmSnapshot(time.time(), tuple(pose[:3]), joints[6], joints=joints, ext_force=(fx, fy, fz),
                                             setpoint=sp or tuple(pose[:3]), goal=self.mover.goal or tuple(pose[:3]),
                                             speed_cap=self.mover.speed_cap, frozen=self.mover.frozen, rot=tuple(pose[3:6]),
                                             status="live")
                time.sleep(max(0.0, dt - (time.perf_counter() - now)))
        except Exception as e:
            self._event(f"read-only thread error: {e!r}")
            with self._lock:
                self._snap.status, self._snap.error = "error", repr(e)
