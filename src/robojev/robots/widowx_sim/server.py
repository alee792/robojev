"""The MuJoCo WidowX AI behind the robot protocol (protocol.py): one arm or the facing pair of
docs/widowx-pair.md, real physics, the standard skills from skills.py, a safety filter, a heartbeat
watchdog and the recorder.

Layers, top down, as in docs/architecture.md:
  SimRobotServer  the protocol: manifest, world (MuJoCo ground truth), the clock, physics stepping,
                  the lag trip, the shared-zone lock, the arm-arm proximity trip and STOP. Everything
                  above the backend (status, start/hold/pause/resume/retarget/heartbeat, subscribe,
                  the watchdog, the argument checks, the events, the lock's rules) is base.SkillServer,
                  shared with the real server (robots/widowx_real).
  SimArm          one arm's ArmIO for the policies (skills.ArmIO) and its motor side: a velocity-
                  capped setpoint (v1's Mover, which also clamps to the workspace box), a rate-limited
                  yaw, v1's damped-least-squares IK tracking a straight-down orientation, the gripper
                  joint on its true 0-0.044 m range, grasp detection from contacts, and the lag trip.
                  In the pair each arm's model elements carry its prefix ("arm_1/joint_0") and its IK
                  moves only its own joints.
  SimWorld        reads the scene (scene.py) out of MjData: object poses, resting/held/on-what, places.

Frames (K5): the server's world, manifest and events speak the COMMON frame, arm_0's base frame,
which is also MuJoCo's world frame (`ArmSpec.base_frame` names it: "base_link" alone, "arm_0/
base_link" in the pair; each arm's workspace box is given in it). Each arm's ArmIO speaks the arm's
OWN base frame, as the real driver would: the policies run unchanged on either arm, "left" and
"toward_robot" mean the arm's own, the hand-over point is the far edge of the arm's own workspace,
and the arm's `reachable()` is its own box. `scene.BasePose` converts at the ArmIO boundary; every
goal a policy sets is converted to the common frame and solved by that arm's IK.

Time: physics at the model's 2 ms step, policies every `POLICY_HZ`. With `realtime=False` nothing
runs by itself: `tick()` advances one policy period and `run_for()`/`run_until()` loop it, so tests
and trials are deterministic and run ~10x faster than wall time. With `realtime=True`, `open()`
starts a thread that does the same paced to the wall clock. The server's clock is sim time either
way; the heartbeat timeout is measured on it.

Two arms, one physics thread: each tick runs every arm's policy, then the shared-zone lock
(base.ZoneLock on this file's geometry), then the proximity check, then the physics steps with each
arm's lag trip. The geometry, in each arm's own frame (the zone's centre line is at spacing / 2 from
either base): an arm is INSIDE the zone when its EE or setpoint is within ZONE_MARGIN of the strip;
ENTERING when the goal its running skill asked for lies there (so two arms carrying toward the
zone settle who goes first while still far apart); BEYOND its side line when its EE
or setpoint is closer to the centre line than KEEPOUT. KEEPOUT comes from the model: the wrist-roll
housing stands WRIST_REACH = 10.4 cm ahead of the fingertips at every straight-down pose, so two
facing EEs must stay 2 x 10.4 + ARM_GAP cm apart along the base line, and an arm at a zone spot
needs the other behind that line. The lock holds a skill with the status "waiting for the shared
zone, held by arm_1: ..." (at the strip's edge, or at its own side line while the other arm holds
the lock), backs a second requester off to its side line, and lets a waiter go when the holder is
out and the other arm is behind its line; an idle arm beyond its side line is moved back to it at
travel height (`_park`), so the zone always frees. The proximity check is the backstop under all of
this: each arm's footprint is the fingertips (a sphere of a held block's half-diagonal), the wrist
housing and the gripper housing as the model has them, and when the next setpoints would bring two
arms' footprints closer than ARM_GAP every arm closing the distance holds, the later-started skill
first, with one `safety_trip` event per held arm naming both arms (`data["arms"]`).

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
  - A block resting in the zone off every spot reads where "shared_zone"; the zone place's name says
    who holds the lock ("the shared zone (held by arm_0)"), PlaceObs having no claimed_by field.
    ObjectObs.claimed_by is the arm whose running skill names the object; another arm's grasp of a
    claimed object is refused.
  - An arm moving itself out of the zone with no skill reads mode "idle": it can be started at once.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

from robojev.arm import Mover
from robojev.config import Workspace

from robojev import catalog
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.base import LagWatch, SkillServer, ZoneLock
from robojev.protocol import ArmObs, ArmSpec, GripperSpec, HandObs, Manifest, ObjectObs, PlaceObs, Recorder, RobotEvent, WorldState

ARM_ID = scene.ARM_0
ROBOT = "widowx-sim"
BASE_FRAME = "base_link"          # the arm's base body: +x forward, +y left, +z up, origin on the base plate
POLICY_HZ = 50
TRAVEL_Z = 0.10                   # 12 cm above the table: reachable straight down out to x = 0.42 (probe 2026-10-10)
HOME_XY = (0.22, 0.0)
PAIR_HOME_XY = (0.10, 0.0)        # the pair's home, in each arm's own frame: behind its side line (0.12 at the default spacing)
GRIPPER_Q_MAX = 0.044             # m of travel per carriage; the fingertips meet at 0, so width = 2 q
MAX_WIDTH = 2 * GRIPPER_Q_MAX
HELD_TOL = 0.0025                 # |width - object size| within this, with both fingers touching = held
                                  # (an 8 mm squeeze reads 1.3 mm of penetration on a 4 cm cube)
ON_TOL_XY = 0.015                 # a block whose centre is within this of another's, resting on its top, is "on:" it
ON_TOL_Z = 0.006
ZONE_MARGIN = 0.035               # the fingers' footprint around the EE point along the table's x axis: a pad's
                                  # outer face is 1.4 cm past a fingertip, the jaws open to 7 cm, the closing axis
                                  # may be 45 deg off the strip: (3.5 + 1.4) cm * sin 45 deg
ARM_GAP = 0.04                    # the two arms' footprints never come closer than this
WRIST_REACH = 0.104               # the wrist-roll housing (link_5) stands this far ahead of the fingertips along the arm's
                                  # forward direction, 17 cm above them, at every straight-down pose (model, 2026-10-11)
KEEPOUT = 2 * WRIST_REACH + ARM_GAP + 0.012   # 0.26 m: how far from the zone's centre line an arm stays while the other
                                  # works there (both wrists, the gap, 1.2 cm of placement slack)


@dataclass(frozen=True)
class Limits:
    """The safety filter's numbers. One place, so D1 points the real backend at the same ones."""
    workspace: Workspace = Workspace(x=(0.15, 0.42), y=(-0.25, 0.25), z=(scene.TABLE_Z + 0.004, 0.14))
    speed_cap: float = 0.15       # m/s, the EE setpoint never moves faster whatever a skill asks
    yaw_rate: float = 2.0         # rad/s for the yaw setpoint
    lag_trip_m: float = 0.03      # EE behind the setpoint by this much = something is in the way
    lag_persist_s: float = 0.02   # ... for this long (10 physics steps): a transient never trips


LIMITS = Limits()
# The pair's box reaches closer to the base: at 0.65 m spacing an arm's own half is shallow, and the
# straight-down grasp is exact (0 mm, no self-contact) at x 0.10 at grasp and travel height (probe
# 2026-10-11; the wrist pitch saturates at x 0.08).
PAIR_LIMITS = Limits(workspace=Workspace(x=(0.10, 0.42), y=(-0.25, 0.25), z=(scene.TABLE_Z + 0.004, 0.14)))

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
    """Ground truth out of MjData for the blocks and the places of a scene.SceneSpec, in the common
    frame. `hands` is the one thing not read from physics: the scene has no hands, so tests inject
    them (set_hands). `arms` is set by the server, so `held_by()` sees every gripper."""

    def __init__(self, mujoco, model, data, spec: scene.SceneSpec):
        self.mujoco, self.model, self.data = mujoco, model, data
        self.table_z = scene.TABLE_Z
        self.hands: list[HandObs] = []
        self.arms: list[SimArm] = []
        self.blocks: dict[str, _Body] = {}
        labels = [scene.label_of(b) for b in spec.blocks]
        for b, label, letter in zip(spec.blocks, labels, scene.letter_names(labels)):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, b.name + "_free")
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, b.name + "_g")
            self.blocks[b.name] = _Body(b.name, scene.block_name(b, letter), label,
                                        bid, gid, model.jnt_qposadr[jid], model.jnt_dofadr[jid], b.size)
        self.geom_to_block = {h.geom: oid for oid, h in self.blocks.items()}
        self.zone = spec.zone()
        self.spec = spec
        self.places: dict[str, PlaceObs] = spec.places()

    # -- reads ---------------------------------------------------------------------------------------
    def ids(self):
        return self.blocks.keys()

    def held_by(self) -> dict[str, str]:
        return {oid: a.id for a in self.arms for oid in [a.held()] if oid}

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
        """place id | "table" | "on:<object id>" | "gripper:<arm id>", from geometry. A resting block is
        in a slot within its radius, else in the shared zone's strip, else on the table. "on:" means
        resting on the other block's top, centred within ON_TOL_XY; anything else off the table
        (falling, tipped on an edge) reads "table"."""
        if oid in held_by:
            return f"gripper:{held_by[oid]}"
        p = self.pose(oid)
        if self.resting(oid):
            for pl in self.places.values():
                if pl.kind == "slot" and math.hypot(p.x - pl.x, p.y - pl.y) <= pl.radius:
                    return pl.id
            if self.zone is not None and self.zone.contains(p.x, p.y):
                return self.zone.id
            return "table"
        for other, h in self.blocks.items():            # above the table: on another block?
            if other == oid:
                continue
            q = self.pose(other)
            if (math.hypot(p.x - q.x, p.y - q.y) <= ON_TOL_XY
                    and abs(p.z - q.z - (h.size + p.size) / 2) <= ON_TOL_Z):
                return f"on:{other}"
        return "table"

    def objects(self, held_by: dict[str, str], claimed_by: dict[str, str] | None = None) -> dict[str, ObjectObs]:
        out = {}
        for oid, h in self.blocks.items():
            p = self.pose(oid)
            out[oid] = ObjectObs(oid, h.name, p.x, p.y, p.z, self.where(oid, held_by), label=h.label, size=h.size,
                                 claimed_by=(claimed_by or {}).get(oid))
        return out

    def place_xy(self, pid: str, held_by: dict[str, str], arm: str | None = None) -> tuple[float, float] | None:
        """Where to set something down for a place id: a slot's centre, a group's or the zone's first
        free spot, or for the "table" place (scene.FreeSpots) a free spot on the table, in `arm`'s
        half when the scene has two arms."""
        pl = self.places.get(pid)
        if pl is None:
            return None
        if pl.kind in ("group", "zone"):
            taken = {self.where(oid, held_by) for oid in self.blocks}
            free = next((m for m in pl.members if m not in taken), None)
            return self.place_xy(free, held_by, arm) if free else None
        if pl.kind == "table":
            return self.free_spot(held_by, arm)
        return (pl.x, pl.y)

    def free_spot(self, held_by: dict[str, str], arm: str | None = None) -> tuple[float, float] | None:
        """The FreeSpots candidate (in `arm`'s half, if the scene has halves) farthest from every
        block not in a gripper and from every slot, if that is at least the clearance; else None."""
        fs = self.spec.free_spots
        if fs is None:
            return None
        cands = fs.candidates()
        if arm is not None and not self.spec.single:
            x0, x1 = self.spec.half(arm)
            cands = [c for c in cands if x0 <= c[0] <= x1]
        others = [(p.x, p.y) for oid in self.blocks if oid not in held_by for p in [self.pose(oid)]]
        others += [(pl.x, pl.y) for pl in self.places.values() if pl.kind == "slot"]
        best, best_d = None, 0.0
        for c in cands:
            d = min((math.hypot(c[0] - x, c[1] - y) for x, y in others), default=math.inf)
            if d > best_d:
                best, best_d = c, d
        return best if best is not None and best_d >= fs.clearance else None

    def occupant(self, pid: str, held_by: dict[str, str]) -> str | None:
        return next((oid for oid in self.blocks if self.where(oid, held_by) == pid), None)

    def move(self, oid: str, x: float, y: float, yaw: float | None = None, z: float | None = None) -> None:
        """Teleport a block onto the table, or to centre height `z` (onto another block), common
        frame. A scripted person; tests. Not part of the protocol."""
        h = self.blocks[oid]
        q = self.data.qpos
        q[h.qadr:h.qadr + 3] = (x, y, self.table_z + h.size / 2 if z is None else z)
        if yaw is not None:
            q[h.qadr + 3:h.qadr + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qvel[h.dadr:h.dadr + 6] = 0
        self.mujoco.mj_forward(self.model, self.data)


# ------------------------------------------------------------------ the arm


class SimArm:
    """One arm: the policies' ArmIO (in the arm's own base frame, `base`) and the motor loop under it
    (in MuJoCo's world frame, the common frame). Everything here runs under the server's lock; the
    server calls `before_step`, steps the model, then `after_step`. `prefix` is what the arm's model
    elements are named with ("" alone, "arm_1/" in the pair)."""

    def __init__(self, mujoco, model, data, world: SimWorld, arm_id: str, limits: Limits, travel_z: float,
                 prefix: str = "", base: scene.BasePose = scene.ORIGIN):
        self.mujoco, self.model, self.data, self.world = mujoco, model, data, world
        self.id = arm_id
        self.prefix, self.base = prefix, base
        self.limits = limits
        self.t = 0.0
        self.table_z = world.table_z
        self.travel_z = travel_z
        self.max_width = MAX_WIDTH
        m = model

        def name(n: str) -> str:
            return prefix + n
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name("ee_site"))
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name(f"joint_{i}")) for i in range(6)]
        self.qadr = [m.jnt_qposadr[j] for j in jids]
        self.dadr = [m.jnt_dofadr[j] for j in jids]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name(f"joint_{i}")) for i in range(6)]
        self.lo = np.array([m.jnt_range[j][0] for j in jids])
        self.hi = np.array([m.jnt_range[j][1] for j in jids])
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name("left_gripper"))
        self.grip_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name(j))]
                       for j in ("left_carriage_joint", "right_carriage_joint")]
        self.finger_geoms: dict[int, str] = {}       # collision geom -> "left" | "right"
        fingers = {name(b): b.split("_")[1] for b in _FINGER_BODIES}
        for g in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if bn in fingers and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.finger_geoms[g] = fingers[bn]
        def collision_geoms(body: str) -> list[int]:
            return [g for g in range(m.ngeom) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == name(body)
                    and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.wrist_geom = collision_geoms("link_5")[0]                                   # the wrist-roll housing: a cylinder
        self.bar_geom = max(collision_geoms("link_6"), key=lambda g: m.geom_size[g][1])    # the gripper housing: a 19 cm wide box
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.mover = Mover(limits.workspace, limits.speed_cap)
        self.yaw_goal = self.yaw_sp = math.pi / 2     # closing axis along +y: zero wrist roll at azimuth 0
        self.speed = 0.0
        self.grip_width = MAX_WIDTH                   # last commanded width
        self.goal_asked: tuple[float, ...] | None = None   # the last goal a policy asked for (arm frame): a freeze
                                                      # overwrites the Mover's goal, the lock still needs the intent
        self.lag = LagWatch(limits.lag_trip_m, limits.lag_persist_s)
        self.lag_m = 0.0
        self.skill: skills.Skill | None = None
        self.frozen_why: str | None = None           # why the mover is frozen with no skill to say so
        self.stopped = False

    # -- staging --------------------------------------------------------------------------------------
    def stage(self, xyz, yaw: float = math.pi / 2, iters: int = 400) -> None:
        """Kinematic solve to a start pose (arm frame): sets qpos directly, no dynamics (v1's pattern)."""
        target = self.base.to_common(xyz)
        R = _down(self.base.yaw_to_common(yaw))
        for _ in range(iters):
            self._ik_step(target, R, gain=0.6)
            self.data.qpos[self.qadr] = self.data.ctrl[self.act]
            self.mujoco.mj_forward(self.model, self.data)
        for q in self.grip_q:
            self.data.qpos[q] = GRIPPER_Q_MAX
        self.data.ctrl[self.grip_act] = GRIPPER_Q_MAX
        self.data.qvel[:] = 0
        self.mujoco.mj_forward(self.model, self.data)
        self.mover.init_at(self.ee())
        self.goal_asked = tuple(float(v) for v in xyz)
        self.yaw_goal = self.yaw_sp = yaw

    # -- ArmIO (arm frame) -----------------------------------------------------------------------------
    def ee_common(self) -> tuple[float, float, float]:
        """The EE point in the common frame, as world() reports it."""
        return tuple(float(v) for v in self.data.site_xpos[self.site])

    def ee(self) -> tuple[float, float, float]:
        return self.base.to_local(self.ee_common())

    def ee_yaw(self) -> float:
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return self.base.yaw_to_local(math.atan2(R[1, 1], R[0, 1]))

    def arrived(self) -> bool:
        return self.mover.setpoint == self.mover.goal and abs(skills.wrap(self.yaw_sp - self.yaw_goal)) < 1e-6

    def width(self) -> float:
        return float(self.data.qpos[self.grip_q[0]] + self.data.qpos[self.grip_q[1]])

    def reachable(self, xyz) -> bool:
        return self.limits.workspace.contains(xyz)

    def _local(self, p: skills.ObjPose | None) -> skills.ObjPose | None:
        if p is None or self.base is scene.ORIGIN:
            return p
        x, y = self.base.to_local((p.x, p.y))
        return skills.ObjPose(x, y, p.z, self.base.yaw_to_local(p.yaw), p.size)

    def obj(self, oid: str) -> skills.ObjPose | None:
        return self._local(self.world.pose(oid))

    def objects(self) -> dict[str, skills.ObjPose]:
        return {oid: self._local(self.world.pose(oid)) for oid in self.world.blocks}

    def where(self, oid: str) -> str:
        return self.world.where(oid, self.world.held_by())

    def place(self, pid: str) -> tuple[float, float] | None:
        xy = self.world.place_xy(pid, self.world.held_by(), self.id)
        return xy if xy is None or self.base is scene.ORIGIN else self.base.to_local(xy)[:2]

    def hands(self) -> list[HandObs]:
        if self.base is scene.ORIGIN:
            return list(self.world.hands)
        out = []
        for h in self.world.hands:
            x, y, z = self.base.to_local((h.x, h.y, h.z))
            vx, vy = self.base.rotate_to_local((h.vx, h.vy))
            out.append(HandObs(x, y, z, vx, vy, h.held_out))
        return out

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
        self.goal_asked = tuple(float(v) for v in xyz)
        self.mover.set_goal(xyz, speed)
        self.speed = min(speed, self.limits.speed_cap)
        if yaw is not None:
            self.yaw_goal = float(yaw)

    def drive(self, xyz, speed: float) -> None:
        """A move the server orders, not the skill (parking, backing off for the lock): the Mover
        runs even while the skill is held, and the skill's own intent (`goal_asked`) is kept."""
        self.mover.resume()
        self.mover.set_goal(xyz, speed)
        self.speed = min(speed, self.limits.speed_cap)

    def grip(self, width: float) -> None:
        self.grip_width = max(0.0, min(MAX_WIDTH, float(width)))
        self.data.ctrl[self.grip_act] = self.grip_width / 2

    def freeze(self) -> None:
        self.mover.freeze("held")

    def unfreeze(self) -> None:
        self.mover.resume()
        self.frozen_why = None

    @property
    def frozen(self) -> bool:
        return self.mover.frozen

    # -- motor loop -----------------------------------------------------------------------------------
    def _ik_step(self, target, R_target: np.ndarray, gain: float = 0.6, damping: float = 0.05) -> None:
        """One damped-least-squares step on position and orientation (common frame), written to the
        arm's joint ctrl (position actuators). v1's solver; gain 0.6 keeps the EE within ~1.5 cm of
        a 0.15 m/s setpoint."""
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
        self._ik_step(self.base.to_common(sp), _down(self.base.yaw_to_common(self.yaw_sp)))

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

    @property
    def running(self) -> bool:
        """a skill is moving this arm, or would be but for the shared-zone lock (and no one else holds it)"""
        return self.active and (self.skill.state == "running" or self.skill.holds.keys() == {"zone"})

    def setpoint_common(self) -> tuple[float, ...]:
        return self.base.to_common(self.mover.setpoint)

    def next_setpoint(self, tick_s: float) -> tuple[float, ...]:
        """Where the setpoint (arm frame) will be after one policy tick if nothing holds it: the
        Mover's step."""
        sp, goal = self.mover.setpoint, self.mover.goal
        d = [g - s for g, s in zip(goal, sp)]
        n = math.sqrt(sum(v * v for v in d))
        if self.mover.frozen or n < 1e-9:
            return tuple(sp)
        k = min(n, self.mover.speed_cap * tick_s) / n
        return tuple(s + k * v for s, v in zip(sp, d))

    def next_setpoint_common(self, tick_s: float) -> tuple[float, ...]:
        return self.base.to_common(self.next_setpoint(tick_s))

    def footprint(self, shift=(0.0, 0.0, 0.0)) -> list[tuple[np.ndarray, Callable[[np.ndarray], float]]]:
        """The proximity check's shapes (common frame), moved by `shift`, each as (centre, support
        function): the fingertips as a sphere of a held block's half-diagonal, the wrist-roll housing
        (link_5's cylinder, the part that reaches furthest toward a facing arm) and the gripper
        housing (link_6's box, 19 cm wide along the closing axis), as they stand in the model."""
        d = np.asarray(shift, dtype=float)
        oid = self.held()
        r = self.world.blocks[oid].size / math.sqrt(2) if oid else 0.0
        shapes: list[tuple[np.ndarray, Callable[[np.ndarray], float]]] = [(np.asarray(self.ee_common()) + d, lambda u, r=r: r)]
        for g in (self.wrist_geom, self.bar_geom):
            c = np.array(self.data.geom_xpos[g]) + d
            R = self.data.geom_xmat[g].reshape(3, 3)
            size = np.array(self.model.geom_size[g])
            if self.model.geom_type[g] == self.mujoco.mjtGeom.mjGEOM_CYLINDER:
                shapes.append((c, lambda u, a=R[:, 2], rr=size[0], h=size[1]: h * abs(float(a @ u)) + rr * math.sqrt(max(0.0, 1 - float(a @ u) ** 2))))
            else:
                shapes.append((c, lambda u, R=R, s=size: float(np.abs(R.T @ u) @ s)))
        return shapes

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
                "objects": {oid: [p.x, p.y, p.z, p.yaw] for oid, p in self.objects().items()}}

    def action(self) -> dict:
        return {"goal": list(self.mover.goal), "speed": self.speed, "yaw": self.yaw_goal, "grip": self.grip_width,
                "frozen": self.mover.frozen}


def footprint_gap(fa, fb) -> float:
    """A lower bound on the distance between two footprints (lists of (centre, support)): for each
    pair of shapes, the separation along the line of centres minus both supports along it (a
    separating-axis bound, exact for spheres, never more than the true distance)."""
    best = math.inf
    for ca, sa in fa:
        for cb, sb in fb:
            d = cb - ca
            n = float(np.linalg.norm(d))
            u = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            best = min(best, n - sa(u) - sb(u))
    return best


# ------------------------------------------------------------------ the server


class SimRobotServer(SkillServer):
    """protocol.RobotServer on MuJoCo. See the module docstring for the clock, the frames and the choices."""

    def __init__(self, spec: scene.SceneSpec, realtime: bool = False, recorder: Recorder | None = None,
                 heartbeat_timeout_s: float = 0.5, params: skills.Params | None = None, limits: Limits | None = None,
                 robot: str = ROBOT):
        """`limits` defaults to LIMITS alone and PAIR_LIMITS for the pair."""
        super().__init__(recorder, params, heartbeat_timeout_s)
        import mujoco
        self.mujoco = mujoco
        self.spec = spec
        self.model = scene.build(spec)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.realtime = realtime
        self.limits = limits or (LIMITS if spec.single else PAIR_LIMITS)
        self.robot = robot
        self.sim_world = SimWorld(mujoco, self.model, self.data, spec)
        self.world_view, self.travel_z, self.table_z, self.max_width = self.sim_world, TRAVEL_Z, scene.TABLE_Z, MAX_WIDTH
        self.home_xy = HOME_XY if spec.single else PAIR_HOME_XY
        self.arms = {arm: SimArm(mujoco, self.model, self.data, self.sim_world, arm, self.limits, TRAVEL_Z,
                                 scene.prefix_for(spec, arm), base)
                     for arm, base in spec.bases().items()}
        self.sim_world.arms = list(self.arms.values())
        for a in self.arms.values():
            a.stage((self.home_xy[0], self.home_xy[1], TRAVEL_Z))
        self.zone = self.sim_world.zone
        self.zone_lock = ZoneLock()
        if self.zone is not None:
            self.side_x = spec.spacing / 2 - KEEPOUT              # each arm's side line, in its own frame
            self.edge_x = spec.spacing / 2 - self.zone.width / 2 - ZONE_MARGIN   # where its fingers reach the strip
            if self.side_x < self.limits.workspace.x[0]:
                raise ValueError(f"spacing {spec.spacing:.2f} m is too close for a shared zone: while one arm works on the "
                                 f"zone's centre line the other must stand {KEEPOUT * 100:.0f} cm from it (2 x {WRIST_REACH * 100:.1f} cm "
                                 f"wrist housings + {ARM_GAP * 100:.0f} cm), at x {self.side_x:.3f} from its base, inside its "
                                 f"workspace (x >= {self.limits.workspace.x[0]:.2f}); spacing >= {2 * (KEEPOUT + self.limits.workspace.x[0]):.2f} m")
        self.dt = float(self.model.opt.timestep)
        self.steps_per_tick = max(1, round(1.0 / (POLICY_HZ * self.dt)))
        self.tick_s = self.steps_per_tick * self.dt
        self._n = 0                          # physics steps so far: the clock
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
        """One policy period: the watchdog and every policy, the zone lock and the proximity check,
        then `steps_per_tick` physics steps with the lag trip after each. Events go out after the
        lock is released."""
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

    def all_finished(self, *skill_ids: str) -> Callable[[], bool]:
        return lambda: all(self._runs[s].finished for s in skill_ids)

    # ---------------------------------------------------------------- protocol: what the sim adds
    @property
    def base_frame(self) -> str:
        """The common frame's name: arm_0's base body in the model."""
        return scene.prefix_for(self.spec, ARM_ID) + BASE_FRAME

    def manifest(self) -> Manifest:
        ws = self.limits.workspace
        specs = []
        for a in self.arms.values():
            where = "" if self.spec.single else f"; base at ({a.base.x:.2f}, {a.base.y:.2f}) heading {math.degrees(a.base.yaw):.0f} deg in {self.base_frame}"
            specs.append(ArmSpec(a.id, self.base_frame, a.base.box_to_common((ws.x, ws.y, ws.z)),
                                 GripperSpec(max_width=MAX_WIDTH, can_grasp=True), travel_z=TRAVEL_Z,
                                 description="WidowX AI follower in MuJoCo, 6 joints plus a parallel gripper, grasping straight down" + where))
        return Manifest(self.robot, tuple(specs), skills.SPECS, catalog=catalog.CATALOG_VERSION)

    def status(self, skill_id: str):
        """The skill's status with its heading in the common frame: a skill works in its arm's own
        frame (its ArmIO), and the protocol's resources speak the common one."""
        st = super().status(skill_id)
        a = self.arms.get(st.arm)
        if st.heading is not None and a is not None and a.base is not scene.ORIGIN:
            st.heading = a.base.to_common(st.heading)[:2]
        return st

    def world(self) -> WorldState:
        with self._lock:
            held = self._held_by()
            arms = {}
            for a in self.arms.values():
                x, y, z = a.ee_common()
                arms[a.id] = ArmObs(a.id, x, y, z, a.held(), a.width(), a.mode(), skill=a.skill.id if a.active else None)
            places = dict(self.sim_world.places)
            if self.zone is not None:
                places.update(self.zone.places(self.zone_lock.holder))
            return WorldState(self.t, self.sim_world.objects(held, self._claimed_by()), places,
                              list(self.sim_world.hands), arms)

    def set_hands(self, hands: list[HandObs]) -> None:
        """Test-only hook, not protocol: what world().hands and the skills' ArmIO.hands() report
        from now on (common frame). The scene has no hands; hand_over is exercised by injecting one."""
        with self._lock:
            self.sim_world.hands = list(hands)

    def move(self, oid: str, x: float, y: float, yaw: float | None = None) -> None:
        """Not protocol: a scripted person moves a block on the table (common frame). Nothing is
        announced; whoever plays the person reports the change as perception would."""
        with self._lock:
            self.sim_world.move(oid, x, y, yaw)

    def stop(self) -> None:
        evs: list[RobotEvent] = []
        with self._lock:
            if not self._stopped:
                self._fail_all("stopped: STOP pressed; every arm is parking", evs)
                for a in self.arms.values():
                    a.unfreeze()
                    a.goto((self.home_xy[0], self.home_xy[1], TRAVEL_Z), 0.03)
        self._emit(evs)

    # ---------------------------------------------------------------- internals
    def _tick(self) -> list[RobotEvent]:
        evs: list[RobotEvent] = []
        self._watchdog(evs)
        self._tick_skills(evs)
        if len(self.arms) > 1:
            if self.zone is not None:
                self._zone_tick()
            self._proximity(evs)
        for _ in range(self.steps_per_tick):
            for a in self.arms.values():
                a.before_step(self.dt)
            self.mujoco.mj_step(self.model, self.data)
            self._n += 1
            for a in self.arms.values():
                why = a.after_step(self.dt)
                if why:
                    self._trip(a, why, {"lag_m": a.lag_m}, evs)
        return evs

    # -- the shared-zone lock -------------------------------------------------------------------------
    def _zone_tick(self) -> None:
        """The lock's geometry for this tick (each arm in its own frame, see the module docstring), its
        rules (base.ZoneLock), its verdict on the skills, the yields, and the idle arms parked."""
        inside, entering, beyond, crossing = {}, {}, {}, {}
        for a in self.arms.values():
            sp_x, ee_x = a.mover.setpoint[0], a.ee()[0]
            intent_x = a.goal_asked[0] if a.goal_asked is not None else sp_x    # where the skill is taking the arm
            inside[a.id] = max(sp_x, ee_x) >= self.edge_x
            beyond[a.id] = max(sp_x, ee_x) > self.side_x + 1e-3
            entering[a.id] = not inside[a.id] and a.running and intent_x >= self.edge_x
            crossing[a.id] = not beyond[a.id] and a.running and intent_x > self.side_x + 1e-3
        waits = self.zone_lock.step(inside, entering, beyond)
        lock = self.zone_lock
        for a in self.arms.values():                         # the side line while another arm holds or waits
            if a.id not in waits and crossing[a.id]:
                if lock.holder not in (None, a.id):
                    waits[a.id] = f"held by {lock.holder}"
                elif any(q != a.id for q in lock.queue):
                    waits[a.id] = f"{next(q for q in lock.queue if q != a.id)} is waiting to enter it"
        for a in self.arms.values():
            if a.id in lock.yielding and a.skill.phase in ("approach", "carry"):
                waits[a.id] += ", backing off to its side line"
                a.drive(self._side_point(a), self.params.v_travel)
        self._apply_zone_waits(waits)
        for a in self.arms.values():
            if beyond[a.id] and not a.active and not a.frozen and not a.stopped:
                self._park(a)

    def _side_point(self, a: SimArm) -> tuple[float, float, float]:
        """Just behind the arm's side line, at its current y, at travel height (arm frame)."""
        return (self.side_x - 0.01, a.ee()[1], self.travel_z)

    def _park(self, a: SimArm) -> None:
        """An idle arm beyond its side line goes back behind it at travel height, so the zone frees
        for the other arm. Starting a skill on it overrides this at once."""
        a.drive(self._side_point(a), self.params.v_travel)

    # -- arm-arm proximity ----------------------------------------------------------------------------
    def arm_gap(self, a: SimArm, b: SimArm) -> float:
        """The gap between two arms' footprints now (fingertips, wrist housing, gripper housing)."""
        return footprint_gap(a.footprint(), b.footprint())

    def _proximity(self, evs: list[RobotEvent]) -> None:
        """Hold every arm that is closing the gap between two arms' footprints to under ARM_GAP,
        judged where the footprints will settle (each moved from its EE to its next setpoint, since
        the EE follows the setpoint within ~1.5 cm); later-started skill first. One event per held
        arm, naming both arms."""
        for a, b in combinations(self.arms.values(), 2):
            settled = {x: x.footprint(np.asarray(x.next_setpoint_common(self.tick_s)) - np.asarray(x.ee_common())) for x in (a, b)}
            now = self.arm_gap(a, b)
            g = min(now, footprint_gap(settled[a], settled[b]))
            if g >= ARM_GAP:
                continue
            closing = [x for x in (a, b) if not x.frozen
                       and footprint_gap(settled[x], (b if x is a else a).footprint()) < now - 1e-9]
            closing.sort(key=lambda x: int(x.skill.id[2:]) if x.active else 0, reverse=True)
            for x in closing:
                y = b if x is a else a
                self._trip(x, f"{x.id} and {y.id} would come within {ARM_GAP * 100:.0f} cm of each other ({max(g, 0) * 100:.1f} cm apart)",
                           {"arms": [x.id, y.id], "gap_m": g}, evs)
