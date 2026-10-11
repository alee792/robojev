"""The MuJoCo WidowX AI behind the robot protocol (protocol.py): one arm, real physics, the standard
skills from skills.py, a safety filter, a heartbeat watchdog and the recorder.

Layers, top down, as in docs/architecture.md:
  SimRobotServer  the protocol: manifest, world (MuJoCo ground truth), status, start/hold/pause/
                  resume/retarget/stop/heartbeat, subscribe. Owns the lock, the clock and the events.
  SimArm          one arm's ArmIO for the policies (skills.ArmIO) and its motor side: a velocity-
                  capped setpoint (v1's Mover, which also clamps to the workspace box), a rate-limited
                  yaw, v1's damped-least-squares IK tracking a straight-down orientation, the gripper
                  joint on its true 0-0.044 m range, grasp detection from contacts, and the lag trip.
  SimWorld        reads the scene (scene.py) out of MjData: object poses, resting/held/on-what, places.

Time: physics at the model's 2 ms step, policies every `POLICY_HZ`. With `realtime=False` nothing
runs by itself: `tick()` advances one policy period and `run_for()`/`run_until()` loop it, so tests
and trials are deterministic and run ~10x faster than wall time. With `realtime=True`, `open()`
starts a thread that does the same paced to the wall clock. The server's clock is sim time either
way; the heartbeat timeout is measured on it.

Choices the protocol leaves open, made here (report, do not settle):
  - start() refuses bad arguments by raising ValueError with the literal reason precondition() gives.
    No skill id, no event: a refusal is not a failure (the MCP tool-error path at K6).
  - start() on a busy arm replaces its skill: the protocol has no cancel, and a new plan that drops
    the current step must be able to start the next one. The replaced skill reads
    "cancelled: replaced by <id>" in status() and sends no event, since its caller did it.
  - The heartbeat watchdog arms on the first heartbeat(): a brain that never connected has not gone
    missing, and the trials and tests run without one. After a loss every arm holds; the brain
    reads status() and calls resume(). A fresh heartbeat does not resume anything.
  - STOP fails every skill with the catalog code "stopped", parks each arm at its home pose at
    crawl speed keeping its grip, and refuses every later start().
  - "on:<id>" means resting on that block's top with the centres within ON_TOL_XY (1.5 cm); a block
    overhanging further, falling or tipped on an edge reads "table". stack_on is judged by it.
  - The scene has no hands. world().hands is whatever set_hands() (a test-only hook) was last given,
    so hand_over's release is exercised by injecting a HandObs; perception fills it at L2.
"""
from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from robojev.arm import Mover
from robojev.config import Workspace

from robojev import catalog
from robojev.robots.widowx_sim import scene, skills
from robojev.protocol import (ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, ObjectObs, PlaceObs, Recorder, RobotEvent,
                       SkillStatus, WorldState)

ARM_ID = "arm_0"
ROBOT = "widowx-sim"
BASE_FRAME = "base_link"          # the arm's base body: +x forward, +y left, +z up, origin on the base plate
POLICY_HZ = 50
TRAVEL_Z = 0.10                   # 12 cm above the table: reachable straight down out to x = 0.42 (probe 2026-10-10)
HOME_XY = (0.22, 0.0)
GRIPPER_Q_MAX = 0.044             # m of travel per carriage; the fingertips meet at 0, so width = 2 q
MAX_WIDTH = 2 * GRIPPER_Q_MAX
HELD_TOL = 0.0025                 # |width - object size| within this, with both fingers touching = held
                                  # (an 8 mm squeeze reads 1.3 mm of penetration on a 4 cm cube)
ON_TOL_XY = 0.015                 # a block whose centre is within this of another's, resting on its top, is "on:" it
ON_TOL_Z = 0.006


@dataclass(frozen=True)
class Limits:
    """The safety filter's numbers. One place, so D1 points the real backend at the same ones."""
    workspace: Workspace = Workspace(x=(0.15, 0.42), y=(-0.25, 0.25), z=(scene.TABLE_Z + 0.004, 0.14))
    speed_cap: float = 0.15       # m/s, the EE setpoint never moves faster whatever a skill asks
    yaw_rate: float = 2.0         # rad/s for the yaw setpoint
    lag_trip_m: float = 0.03      # EE behind the setpoint by this much = something is in the way
    lag_persist_s: float = 0.02   # ... for this long (10 physics steps): a transient never trips


LIMITS = Limits()

_FINGER_BODIES = ("carriage_left", "carriage_right")


def _down(yaw: float) -> np.ndarray:
    """Rotation with the EE x axis (the finger direction) pointing down and the closing axis (EE y)
    along `yaw` in the xy plane."""
    xa = np.array([0.0, 0.0, -1.0])
    ya = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return np.stack([xa, ya, np.cross(xa, ya)], 1)


def _quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ------------------------------------------------------------------ the world


@dataclass
class _Body:
    id: str
    name: str
    label: str | None
    body: int
    geom: int
    qadr: int                    # free joint: 3 pos + 4 quat
    dadr: int                    # free joint: 6 vel
    size: float


class SimWorld:
    """Ground truth out of MjData for the blocks and the tray of a scene.SceneSpec. `hands` is the
    one thing not read from physics: the scene has no hands, so tests inject them (set_hands)."""

    def __init__(self, mujoco, model, data, spec: scene.SceneSpec):
        self.mujoco, self.model, self.data = mujoco, model, data
        self.table_z = scene.TABLE_Z
        self.hands: list[HandObs] = []
        self.blocks: dict[str, _Body] = {}
        for b in spec.blocks:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, b.name + "_free")
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b.name + "_g")
            tail = b.name.rsplit("_", 1)[-1]
            self.blocks[b.name] = _Body(b.name, b.name.replace("_", " "), tail if len(tail) == 1 or tail.isdigit() else None,
                                        bid, gid, model.jnt_qposadr[jid], model.jnt_dofadr[jid], b.size)
        self.geom_to_block = {h.geom: oid for oid, h in self.blocks.items()}
        self.places: dict[str, PlaceObs] = {}
        if spec.tray:
            t = spec.tray
            ids = []
            for i in range(t.n):
                x, y = t.slot_xy(i)
                tag = " (the leftmost slot)" if i == 0 else " (the rightmost slot)" if i == t.n - 1 else ""
                self.places[f"slot_{i}"] = PlaceObs(f"slot_{i}", f"tray slot {i}{tag}", "slot", x, y, 1, radius=t.slot / 2)
                ids.append(f"slot_{i}")
            cx = sum(self.places[i].x for i in ids) / len(ids)
            cy = sum(self.places[i].y for i in ids) / len(ids)
            self.places["tray"] = PlaceObs("tray", "the tray (any free slot)", "group", cx, cy, t.n, tuple(ids))

    # -- reads ---------------------------------------------------------------------------------------
    def pose(self, oid: str) -> skills.ObjPose | None:
        h = self.blocks.get(oid)
        if h is None:
            return None
        q = self.data.qpos[h.qadr:h.qadr + 7]
        return skills.ObjPose(float(q[0]), float(q[1]), float(q[2]), _quat_yaw(q[3:7]), h.size)

    def poses(self) -> dict[str, list[float]]:
        return {oid: [p.x, p.y, p.z, p.yaw] for oid in self.blocks for p in [self.pose(oid)]}

    def resting(self, oid: str) -> bool:
        """Sitting on the table: at its resting height and still."""
        h = self.blocks[oid]
        z = self.data.qpos[h.qadr + 2]
        v = self.data.qvel[h.dadr:h.dadr + 3]
        return abs(z - (self.table_z + h.size / 2)) < 0.005 and float(np.linalg.norm(v)) < 0.02

    def where(self, oid: str, held_by: dict[str, str]) -> str:
        """place id | "table" | "on:<object id>" | "gripper:<arm id>", from geometry. "on:" means
        resting on the other block's top, centred within ON_TOL_XY; anything else off the table
        (falling, tipped on an edge) reads "table"."""
        if oid in held_by:
            return f"gripper:{held_by[oid]}"
        p = self.pose(oid)
        if self.resting(oid):
            for pl in self.places.values():
                if pl.kind == "slot" and math.hypot(p.x - pl.x, p.y - pl.y) <= pl.radius:
                    return pl.id
            return "table"
        for other, h in self.blocks.items():            # above the table: on another block?
            if other == oid:
                continue
            q = self.pose(other)
            if (math.hypot(p.x - q.x, p.y - q.y) <= ON_TOL_XY
                    and abs(p.z - q.z - (h.size + p.size) / 2) <= ON_TOL_Z):
                return f"on:{other}"
        return "table"

    def objects(self, held_by: dict[str, str]) -> dict[str, ObjectObs]:
        out = {}
        for oid, h in self.blocks.items():
            p = self.pose(oid)
            out[oid] = ObjectObs(oid, h.name, p.x, p.y, p.z, self.where(oid, held_by), label=h.label, size=h.size)
        return out

    def place_xy(self, pid: str, held_by: dict[str, str]) -> tuple[float, float] | None:
        """Where to set something down for a place id: a slot's centre, or a group's first free slot."""
        pl = self.places.get(pid)
        if pl is None:
            return None
        if pl.kind == "group":
            taken = {self.where(oid, held_by) for oid in self.blocks}
            free = next((m for m in pl.members if m not in taken), None)
            return self.place_xy(free, held_by) if free else None
        return (pl.x, pl.y)

    def occupant(self, pid: str, held_by: dict[str, str]) -> str | None:
        return next((oid for oid in self.blocks if self.where(oid, held_by) == pid), None)

    def move(self, oid: str, x: float, y: float, yaw: float | None = None, z: float | None = None) -> None:
        """Teleport a block onto the table, or to centre height `z` (onto another block). A scripted
        person; tests. Not part of the protocol."""
        h = self.blocks[oid]
        q = self.data.qpos
        q[h.qadr:h.qadr + 3] = (x, y, self.table_z + h.size / 2 if z is None else z)
        if yaw is not None:
            q[h.qadr + 3:h.qadr + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qvel[h.dadr:h.dadr + 6] = 0
        self.mujoco.mj_forward(self.model, self.data)


# ------------------------------------------------------------------ the arm


class LagWatch:
    """Fires once when the EE has lagged the setpoint by more than `trip_m` for `persist_s`, and
    re-arms when the lag drops back under: one trip per obstacle, not one per physics step."""

    def __init__(self, trip_m: float, persist_s: float):
        self.trip_m, self.persist_s = trip_m, persist_s
        self.over_s = 0.0
        self.fired = False

    def update(self, lag: float, dt: float) -> bool:
        if lag <= self.trip_m:
            self.over_s, self.fired = 0.0, False
            return False
        self.over_s += dt
        if self.over_s >= self.persist_s and not self.fired:
            self.fired = True
            return True
        return False


class SimArm:
    """One arm: the policies' ArmIO and the motor loop under it. Everything here runs under the
    server's lock; the server calls `before_step`, steps the model, then `after_step`."""

    def __init__(self, mujoco, model, data, world: SimWorld, arm_id: str, limits: Limits, travel_z: float):
        self.mujoco, self.model, self.data, self.world = mujoco, model, data, world
        self.id = arm_id
        self.limits = limits
        self.t = 0.0
        self.table_z = world.table_z
        self.travel_z = travel_z
        self.max_width = MAX_WIDTH
        m = model
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(6)]
        self.qadr = [m.jnt_qposadr[j] for j in jids]
        self.dadr = [m.jnt_dofadr[j] for j in jids]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(6)]
        self.lo = np.array([m.jnt_range[j][0] for j in jids])
        self.hi = np.array([m.jnt_range[j][1] for j in jids])
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper")
        self.grip_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                       for j in ("left_carriage_joint", "right_carriage_joint")]
        self.finger_geoms: dict[int, str] = {}       # collision geom -> "left" | "right"
        for g in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if bn in _FINGER_BODIES and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.finger_geoms[g] = bn.split("_")[1]
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.mover = Mover(limits.workspace, limits.speed_cap)
        self.yaw_goal = self.yaw_sp = math.pi / 2     # closing axis along +y: zero wrist roll at azimuth 0
        self.speed = 0.0
        self.grip_width = MAX_WIDTH                   # last commanded width
        self.lag = LagWatch(limits.lag_trip_m, limits.lag_persist_s)
        self.lag_m = 0.0
        self.skill: skills.Skill | None = None
        self.frozen_why: str | None = None           # why the mover is frozen with no skill to say so
        self.stopped = False

    # -- staging --------------------------------------------------------------------------------------
    def stage(self, xyz, yaw: float = math.pi / 2, iters: int = 400) -> None:
        """Kinematic solve to a start pose: sets qpos directly, no dynamics (v1's pattern)."""
        R = _down(yaw)
        for _ in range(iters):
            self._ik_step(xyz, R, gain=0.6)
            self.data.qpos[self.qadr] = self.data.ctrl[self.act]
            self.mujoco.mj_forward(self.model, self.data)
        for q in self.grip_q:
            self.data.qpos[q] = GRIPPER_Q_MAX
        self.data.ctrl[self.grip_act] = GRIPPER_Q_MAX
        self.data.qvel[:] = 0
        self.mujoco.mj_forward(self.model, self.data)
        self.mover.init_at(self.ee())
        self.yaw_goal = self.yaw_sp = yaw

    # -- ArmIO ----------------------------------------------------------------------------------------
    def ee(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.data.site_xpos[self.site])

    def ee_yaw(self) -> float:
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return math.atan2(R[1, 1], R[0, 1])

    def arrived(self) -> bool:
        return self.mover.setpoint == self.mover.goal and abs(skills.wrap(self.yaw_sp - self.yaw_goal)) < 1e-6

    def width(self) -> float:
        return float(self.data.qpos[self.grip_q[0]] + self.data.qpos[self.grip_q[1]])

    def reachable(self, xyz) -> bool:
        return self.limits.workspace.contains(xyz)

    def obj(self, oid: str) -> skills.ObjPose | None:
        return self.world.pose(oid)

    def objects(self) -> dict[str, skills.ObjPose]:
        return {oid: self.world.pose(oid) for oid in self.world.blocks}

    def where(self, oid: str) -> str:
        return self.world.where(oid, self._held_by())

    def place(self, pid: str) -> tuple[float, float] | None:
        return self.world.place_xy(pid, self._held_by())

    def hands(self) -> list[HandObs]:
        return list(self.world.hands)

    def _held_by(self) -> dict[str, str]:
        return {h: self.id for h in [self.held()] if h}

    def held(self) -> str | None:
        """The block both finger sides touch while the width matches its size. Contacts come from
        the last physics step, so this is as fresh as anything the policy reads."""
        sides: dict[str, set[str]] = {}
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            for fg, og in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                side = self.finger_geoms.get(fg)
                oid = self.world.geom_to_block.get(og)
                if side and oid:
                    sides.setdefault(oid, set()).add(side)
        w = self.width()
        for oid, s in sides.items():
            if len(s) == 2 and abs(w - self.world.blocks[oid].size) <= HELD_TOL:
                return oid
        return None

    def goto(self, xyz, speed: float, yaw: float | None = None) -> None:
        self.mover.set_goal(xyz, speed)
        self.speed = min(speed, self.limits.speed_cap)
        if yaw is not None:
            self.yaw_goal = float(yaw)

    def grip(self, width: float) -> None:
        self.grip_width = max(0.0, min(MAX_WIDTH, float(width)))
        self.data.ctrl[self.grip_act] = self.grip_width / 2

    def freeze(self) -> None:
        self.mover.freeze("held")

    def unfreeze(self) -> None:
        self.mover.resume()
        self.frozen_why = None

    # -- motor loop -----------------------------------------------------------------------------------
    def _ik_step(self, target, R_target: np.ndarray, gain: float = 0.6, damping: float = 0.05) -> None:
        """One damped-least-squares step on position and orientation, written to the joint ctrl
        (position actuators). v1's solver; gain 0.6 keeps the EE within ~1.5 cm of a 0.15 m/s setpoint."""
        p = self.data.site_xpos[self.site]
        R = self.data.site_xmat[self.site].reshape(3, 3)
        ep = np.asarray(target) - p
        Re = R_target @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        self.mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
        e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        self.data.ctrl[self.act] = np.clip(self.data.qpos[self.qadr] + gain * dq, self.lo, self.hi)

    def before_step(self, dt: float) -> None:
        sp = self.mover.step(dt)
        if not self.mover.frozen:
            d = skills.wrap(self.yaw_goal - self.yaw_sp)
            step = self.limits.yaw_rate * dt
            self.yaw_sp = self.yaw_goal if abs(d) <= step else self.yaw_sp + math.copysign(step, d)
        self._ik_step(sp, _down(self.yaw_sp))

    def after_step(self, dt: float) -> str | None:
        """The lag trip. Returns the literal reason when it fires, else None."""
        sp = self.mover.setpoint
        self.lag_m = math.dist(self.ee(), sp)
        if self.lag.update(self.lag_m, dt):
            return f"EE {self.lag_m * 100:.1f} cm behind its setpoint at {skills.fmt_xyz(self.ee())}: something is in the way"
        return None

    # -- for the server ---------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.skill is not None and not self.skill.finished

    def mode(self) -> str:
        if self.stopped:
            return "stopped"
        if self.active:
            s = self.skill
            return "tripped" if s.state == "holding" and s.tripped else s.state
        if self.mover.frozen and self.frozen_why:        # held by request or by the filter, no skill to say so
            return "tripped" if self.frozen_why == "trip" else "holding"
        return "idle"                                     # a failed skill leaves the mover frozen: nothing to resume

    def observation(self) -> dict:
        return {"ee": list(self.ee()), "yaw": self.ee_yaw(), "width": self.width(), "holding": self.held(),
                "setpoint": list(self.mover.setpoint), "lag": self.lag_m, "mode": self.mode(),
                "objects": self.world.poses()}

    def action(self) -> dict:
        return {"goal": list(self.mover.goal), "speed": self.speed, "yaw": self.yaw_goal, "grip": self.grip_width,
                "frozen": self.mover.frozen}


# ------------------------------------------------------------------ the server


class SimRobotServer:
    """protocol.RobotServer on MuJoCo. See the module docstring for the clock and the choices."""

    def __init__(self, spec: scene.SceneSpec, realtime: bool = False, recorder: Recorder | None = None,
                 heartbeat_timeout_s: float = 0.5, params: skills.Params | None = None, limits: Limits = LIMITS,
                 robot: str = ROBOT):
        import mujoco
        self.mujoco = mujoco
        self.spec = spec
        self.model = scene.build(spec)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.realtime = realtime
        self.recorder = recorder
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.params = params or skills.Params()
        self.limits = limits
        self.robot = robot
        self.sim_world = SimWorld(mujoco, self.model, self.data, spec)
        self.arms = {ARM_ID: SimArm(mujoco, self.model, self.data, self.sim_world, ARM_ID, limits, TRAVEL_Z)}
        for a in self.arms.values():
            a.stage((HOME_XY[0], HOME_XY[1], TRAVEL_Z))
        self.dt = float(self.model.opt.timestep)
        self.steps_per_tick = max(1, round(1.0 / (POLICY_HZ * self.dt)))
        self.tick_s = self.steps_per_tick * self.dt
        self._lock = threading.RLock()
        self._runs: dict[str, skills.Skill] = {}
        self._ids = itertools.count(1)
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._n = 0                          # physics steps so far: the clock
        self._last_hb: float | None = None
        self._hb_lost = False
        self._stopped = False
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle and clock (not protocol)
    @property
    def t(self) -> float:
        return self._n * self.dt

    def open(self) -> "SimRobotServer":
        if self.realtime and self._thread is None:
            self._thread = threading.Thread(target=self._clock, name="sim-robot", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def _clock(self) -> None:
        t0 = time.perf_counter()
        k = 0
        while not self._stop_evt.is_set():
            self.tick()
            k += 1
            ahead = t0 + k * self.tick_s - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)

    def tick(self) -> None:
        """One policy period: the watchdog and every policy, then `steps_per_tick` physics steps with
        the safety filter after each. Events go out after the lock is released."""
        with self._lock:
            evs = self._tick()
        self._emit(evs)

    def run_for(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.tick_s))):
            self.tick()

    def run_until(self, cond: Callable[[], bool], timeout_s: float) -> bool:
        """Tick until `cond()` holds or `timeout_s` of sim time has passed. True if it held."""
        t_end = self.t + timeout_s
        while self.t < t_end:
            if cond():
                return True
            self.tick()
        return cond()

    def finished(self, skill_id: str) -> Callable[[], bool]:
        """A `run_until` condition: the skill is done or failed."""
        return lambda: self._runs[skill_id].finished

    # ---------------------------------------------------------------- protocol: manifest and resources
    def manifest(self) -> Manifest:
        ws = self.limits.workspace
        arm = ArmSpec(ARM_ID, BASE_FRAME, (ws.x, ws.y, ws.z), GripperSpec(max_width=MAX_WIDTH, can_grasp=True),
                      travel_z=TRAVEL_Z,
                      description="WidowX AI follower in MuJoCo, 6 joints plus a parallel gripper, grasping straight down")
        return Manifest(self.robot, (arm,), skills.SPECS, catalog=catalog.CATALOG_VERSION)

    def world(self) -> WorldState:
        with self._lock:
            held = self._held_by()
            arms = {}
            for a in self.arms.values():
                x, y, z = a.ee()
                arms[a.id] = ArmObs(a.id, x, y, z, a.held(), a.width(), a.mode(), skill=a.skill.id if a.active else None)
            return WorldState(self.t, self.sim_world.objects(held), dict(self.sim_world.places),
                              list(self.sim_world.hands), arms)

    def set_hands(self, hands: list[HandObs]) -> None:
        """Test-only hook, not protocol: what world().hands and the skills' ArmIO.hands() report
        from now on. The scene has no hands; hand_over is exercised by injecting one here."""
        with self._lock:
            self.sim_world.hands = list(hands)

    def status(self, skill_id: str) -> SkillStatus:
        with self._lock:
            run = self._runs.get(skill_id)
            if run is None:
                raise KeyError(f"no skill {skill_id}")
            return run.status()

    # ---------------------------------------------------------------- protocol: tools
    def precondition(self, arm: str, skill: str, args: dict) -> str | None:
        with self._lock:
            return self._why_not(arm, skill, args)

    def start(self, arm: str, skill: str, args: dict) -> str:
        with self._lock:
            why = self._why_not(arm, skill, args)
            if why:
                raise ValueError(why)
            a = self.arms[arm]
            sid = f"sk{next(self._ids)}"
            if a.active:
                old = a.skill
                old.state, old.reason, old.notified = "cancelled", f"cancelled: replaced by {sid}", True
            a.unfreeze()
            a.skill = skills.SKILLS[skill](sid, arm, args, self.params)
            self._runs[sid] = a.skill
            note = getattr(self.recorder, "note", None)       # outside protocol.Recorder; the JSONL recorder has it
            if note is not None:
                note(kind="skill_start", skill_id=sid, arm=arm, skill=skill, args=dict(args))
            return sid

    def hold(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.hold(a)

    def pause(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.pause(a)

    def resume(self, arm: str) -> None:
        with self._lock:
            a = self._arm(arm)
            if a.stopped:
                return
            if a.active:
                a.skill.resume(a)
            a.unfreeze()

    def retarget(self, skill_id: str) -> None:
        with self._lock:
            run = self._runs.get(skill_id)
            if run is None or run.finished:
                return
            run.retarget(self.arms[run.arm])

    def stop(self) -> None:
        evs = []
        with self._lock:
            if not self._stopped:
                self._stopped = True
                for a in self.arms.values():
                    if a.active:
                        s = a.skill
                        s.fail(a, "stopped: STOP pressed; every arm is parking")
                        s.notified = True
                        evs.append(self._ev("skill_failed", s.reason, skill_id=s.id, arm=a.id, object=s.args.get("object")))
                    a.stopped = True
                    a.unfreeze()
                    a.goto((HOME_XY[0], HOME_XY[1], TRAVEL_Z), 0.03)
        self._emit(evs)

    def heartbeat(self) -> None:
        with self._lock:
            self._last_hb, self._hb_lost = self.t, False

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            if callback in self._subs:
                self._subs.remove(callback)

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    # ---------------------------------------------------------------- internals
    def _arm(self, arm: str) -> SimArm:
        a = self.arms.get(arm)
        if a is None:
            raise ValueError(f"no arm {arm}")
        return a

    def _held_by(self) -> dict[str, str]:
        return {oid: a.id for a in self.arms.values() for oid in [a.held()] if oid}

    def _why_not(self, arm: str, skill: str, args: dict) -> str | None:
        """None, or a literal reason prefixed with a catalog.REASONS code."""
        a = self.arms.get(arm)
        spec = self.manifest().skill(skill)
        if a is None:
            return f"precondition: no arm {arm}"
        if spec is None:
            return f"precondition: {arm} has no skill {skill}"
        if a.stopped:
            return "precondition: the arm is stopped (STOP)"
        schema = spec.args_schema
        missing = [k for k in schema.get("required", []) if k not in args]
        if missing:
            return f"precondition: {skill} needs {', '.join(missing)}"
        extra = [k for k in args if k not in schema.get("properties", {})]
        if extra:
            return f"precondition: {skill} takes no argument {', '.join(extra)}"
        for k, v in args.items():
            prop = schema["properties"][k]
            kind = prop.get("type")
            if kind == "string" and not isinstance(v, str) or kind == "number" and not isinstance(v, (int, float)):
                return f"precondition: {k} must be a {kind}"
            if "enum" in prop and v not in prop["enum"]:
                return f"precondition: {k} must be one of {', '.join(prop['enum'])}, not {v!r}"
            if kind == "number" and not prop.get("minimum", -math.inf) <= v <= prop.get("maximum", math.inf):
                return f"precondition: {k} must be between {prop.get('minimum')} and {prop.get('maximum')}"
        held = self._held_by()
        if skill in ("pick_and_place", "stack_on", "hand_over"):
            why = self._grasp_why_not(a, args["object"], held)
            if why:
                return why
        if skill == "pick_and_place":
            oid, pid = args["object"], args["place"]
            pl = self.sim_world.places.get(pid)
            if pl is None:
                return f"precondition: there is no place {pid} in the world"
            occ = self.sim_world.occupant(pid, held)
            if pl.kind == "slot" and occ and occ != oid:
                return f"blocked: {pl.name} is occupied by {occ}"
            xy = self.sim_world.place_xy(pid, held)
            if xy is None:
                return f"blocked: {pl.name} has no free slot"
            if not a.reachable((xy[0], xy[1], TRAVEL_Z)):
                return f"unreachable: {pid} at ({xy[0]:.2f}, {xy[1]:.2f}) is outside the arm's workspace"
        if skill == "stack_on":
            oid, onto = args["object"], args["onto"]
            b = self.sim_world.pose(onto)
            if b is None:
                return f"not_in_view: there is no object {onto} in the world"
            if onto == oid:
                return f"precondition: {oid} cannot be stacked on itself"
            if onto in held:
                return f"precondition: {onto} is held by {held[onto]}"
            if not a.reachable((b.x, b.y, TRAVEL_Z)):
                return f"unreachable: {onto} at ({b.x:.2f}, {b.y:.2f}) is outside the arm's workspace"
            top = next((o for o in self.sim_world.blocks if o != oid and self.sim_world.where(o, held) == f"on:{onto}"), None)
            if top:
                return f"blocked: {onto} already has {top} on it"
            o = self.sim_world.pose(oid)
            if b.z + b.size / 2 + o.size + self.params.tip_height > TRAVEL_Z:      # the carried block would hit it
                return f"unreachable: the top of {onto} is {b.z + b.size / 2 - scene.TABLE_Z:.3f} m above the table, too high to stack on"
        if skill == "push":
            oid = args["object"]
            o = self.sim_world.pose(oid)
            if o is None:
                return f"not_in_view: there is no object {oid} in the world"
            if oid in held:
                return f"precondition: {oid} is held by {held[oid]}"
            if a.held():
                return f"precondition: the gripper is holding {a.held()}"
            start, limit = skills.push_path(o, args["direction"], args["distance"], self.params)
            push_z = scene.TABLE_Z + self.params.tip_height
            for xy, what in ((start, "starts"), (limit, "ends")):
                if not a.reachable((xy[0], xy[1], push_z)):
                    return (f"unreachable: pushing {oid} {args['distance'] * 100:.0f} cm {args['direction']} {what} at "
                            f"({xy[0]:.2f}, {xy[1]:.2f}), outside the arm's workspace")
        if skill == "hand_over":
            ox, oy = self.params.offer_xy
            if not a.reachable((ox, oy, self.params.offer_z)):
                return f"unreachable: the hand-over point ({ox:.2f}, {oy:.2f}) is outside the arm's workspace"
        return None

    def _grasp_why_not(self, a: SimArm, oid: str, held: dict[str, str]) -> str | None:
        """What stops this arm picking `oid` up: unknown, too wide, in another gripper, this gripper
        full of something else, out of reach. Already in this gripper is fine (the skill starts at lift)."""
        o = self.sim_world.pose(oid)
        if o is None:
            return f"not_in_view: there is no object {oid} in the world"
        if o.size + 0.004 > MAX_WIDTH:
            return f"precondition: {oid} is {o.size * 100:.1f} cm wide; the fingers open to {MAX_WIDTH * 100:.1f} cm"
        mine = a.held()
        if mine and mine != oid:
            return f"precondition: the gripper is holding {mine}"
        if held.get(oid, a.id) != a.id:
            return f"precondition: {oid} is held by {held[oid]}"
        if not a.reachable((o.x, o.y, TRAVEL_Z)):
            return f"unreachable: {oid} at ({o.x:.2f}, {o.y:.2f}) is outside the arm's workspace"
        return None

    def _ev(self, kind: str, text: str, **kw) -> RobotEvent:
        return RobotEvent(self.t, kind, text, **kw)

    def _emit(self, evs: list[RobotEvent]) -> None:
        for ev in evs:
            if self.recorder is not None:
                self.recorder.event(ev)
            for cb in list(self._subs):
                cb(ev)

    def _tick(self) -> list[RobotEvent]:
        evs: list[RobotEvent] = []
        t = self.t
        if self._last_hb is not None and not self._hb_lost and t - self._last_hb > self.heartbeat_timeout_s:
            self._hb_lost = True
            for a in self.arms.values():
                if a.stopped:
                    continue
                if a.active:
                    a.skill.hold(a)
                elif not a.mover.frozen:
                    a.freeze()
                    a.frozen_why = "heartbeat"
            evs.append(self._ev("heartbeat_lost", f"no heartbeat for {self.heartbeat_timeout_s:.1f} s: every arm is holding"))
        for a in self.arms.values():
            a.t = t
            s = a.skill
            if s is None:
                continue
            if not s.finished:
                s.tick(a)
                if self.recorder is not None:
                    self.recorder.frame(t, a.id, s.id, s.phase, a.observation(), a.action())
            if s.finished and not s.notified:       # finished in its tick, or outside it (retarget, STOP)
                s.notified = True
                if s.state == "done":
                    evs.append(self._ev("skill_done", f"{s.name} finished: {s.text}", skill_id=s.id, arm=a.id,
                                        object=s.args.get("object")))
                else:
                    evs.append(self._ev("skill_failed", s.reason, skill_id=s.id, arm=a.id, object=s.args.get("object"),
                                        data={"phase": s.phase}))
        for _ in range(self.steps_per_tick):
            for a in self.arms.values():
                a.before_step(self.dt)
            self.mujoco.mj_step(self.model, self.data)
            self._n += 1
            for a in self.arms.values():
                why = a.after_step(self.dt)
                if why and not a.stopped:
                    sid = None
                    if a.active:
                        a.skill.trip(a, why)
                        sid = a.skill.id
                    else:
                        a.freeze()
                        a.frozen_why = "trip"
                    evs.append(self._ev("safety_trip", f"the safety filter tripped on {a.id}: {why}; holding",
                                        skill_id=sid, arm=a.id, data={"lag_m": a.lag_m}))
        return evs
