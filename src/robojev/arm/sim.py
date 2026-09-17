"""MuJoCo backend: the WidowX AI follower from trossen_arm_mujoco, a table with a cup and a flat
object, physics at 500 Hz in a thread, a damped-least-squares IK tracking the streamed setpoint
with the same fixed orientation the real backend streams, and the wrist camera rendered (colour
and depth) so the real perception pipeline runs unchanged.

Conventions match the real arm: base frame +x forward, +y left, +z up; EE = `ee_site` (0.156 m
along link_6 x, the driver's t_flange_tool); orientation reported as angle-axis.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from robojev.arm import ArmSnapshot, Mover
from robojev.config import Config

SIM_DIR = Path(os.environ.get("TROSSEN_ARM_MUJOCO_DIR", Path.home() / "trossen_arm_mujoco"))
FOLLOWER_XML = SIM_DIR / "trossen_arm_mujoco/assets/wxai/wxai_follower.xml"
TABLE_Z = -0.02   # table surface in base frame, like the real bay (base plate sits 2 cm proud)

OBJECTS = [  # name, kind, xy, size, rgba
    ("cup", "cylinder", (0.37, -0.07), (0.04, 0.055), (0.93, 0.90, 0.85, 1)),      # radius, half-height -> 11 cm tall
    ("phone", "box", (0.30, 0.10), (0.035, 0.07, 0.012), (0.05, 0.05, 0.05, 1)),   # 7x14x2.4 cm (a thick phone / small book)
    ("hand", "box", (0.30, -0.60), (0.045, 0.06, 0.02), (0.85, 0.65, 0.55, 1)),     # 9x12x4 cm hand-sized intruder, parked out of view
]


def rot_y(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_to_angle_axis(R):
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return (0.0, 0.0, 0.0)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return tuple(float(x) for x in v * th)


def build_model():
    import mujoco
    spec = mujoco.MjSpec.from_file(str(FOLLOWER_XML))
    spec.modelname = "robojev_sim"
    w = spec.worldbody
    # a table plane under the arm base, a light, and the objects as mocap bodies (scriptable)
    w.add_light(pos=[0.3, 0, 1.2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.5, 0.01], pos=[0.3, 0, TABLE_Z - 0.01],
               rgba=[0.75, 0.6, 0.4, 1], contype=0, conaffinity=0)
    for name, kind, (x, y), size, rgba in OBJECTS:
        b = w.add_body(name=name, mocap=True)
        if kind == "cylinder":
            b.pos = [x, y, TABLE_Z + size[1]]
            b.add_geom(name=name + "_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size[0], size[1], 0], rgba=list(rgba),
                       contype=0, conaffinity=0)
        else:
            b.pos = [x, y, TABLE_Z + size[2]]
            b.add_geom(name=name + "_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), rgba=list(rgba), contype=0, conaffinity=0)
    # third-person camera for the dashboard, and an overhead "scene" camera like the boom D455
    w.add_camera(name="third", pos=[0.9, -0.7, 0.55], xyaxes=[0.6, 0.8, 0, -0.35, 0.26, 0.9])
    w.add_camera(name="overhead", pos=[0.33, 0.0, 0.80], xyaxes=[0, -1, 0, 1, 0, 0], fovy=60)
    return spec.compile()


class SimArm:
    def __init__(self, cfg: Config, log=None, realtime: bool = True):
        import mujoco
        self.mujoco = mujoco
        self.cfg = cfg
        self.log = log
        self.realtime = realtime
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(6)]
        self.qadr = [m.jnt_qposadr[j] for j in jids]
        self.dadr = [m.jnt_dofadr[j] for j in jids]
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}") for i in range(6)]
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper")
        self.grip_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_carriage_joint")]
        self.lo = np.array([m.jnt_range[j][0] for j in jids]); self.hi = np.array([m.jnt_range[j][1] for j in jids])
        self.mocap = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name, *_ in OBJECTS}
        self.R_target = rot_y(cfg.motion.down_orientation[1])
        self.mover = Mover(cfg.workspace, cfg.motion.hard_speed_cap)
        self.gripper_goal = 0.044     # sim ctrl units (0.022 closed .. 0.044 open)
        self.gripper_cmd = 0.04       # real units (0 .. 0.04)
        self.attached = None          # object name held (kinematic attach: the fingers have no collision geoms)
        self.half_h = {name: (size[1] if kind == "cylinder" else size[2]) for name, kind, _, size, _ in OBJECTS}
        self.lock = threading.Lock()          # guards data for renders
        self._snap = ArmSnapshot(time.time(), (0, 0, 0), 0.044, status="init")
        self._stop = threading.Event()
        self._thread = None
        self.events = []
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv))
        self.sim_time = 0.0
        self.last_step_m = 0.0
        self.max_step_m = 0.0

    # -- helpers ---------------------------------------------------------------------------------
    def _event(self, text):
        self.events.append((time.time(), text)); self.events = self.events[-50:]
        if self.log:
            self.log.write("events", kind="arm", text=text)

    def ee(self):
        p = self.data.site_xpos[self.site].copy(); R = self.data.site_xmat[self.site].reshape(3, 3).copy()
        return p, R

    def _ik_step(self, target, gain=0.5, damping=0.05):
        p, R = self.ee()
        ep = np.asarray(target) - p
        Re = self.R_target @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        self.mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
        e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
        q = np.clip(self.data.qpos[self.qadr] + gain * dq, self.lo, self.hi)
        self.data.ctrl[self.act] = q

    def _solve_to(self, target, iters=300):
        """Kinematic solve (used for staging): sets qpos directly, no dynamics."""
        for _ in range(iters):
            self._ik_step(target, gain=0.6)
            self.data.qpos[self.qadr] = self.data.ctrl[self.act]
            self.mujoco.mj_forward(self.model, self.data)

    def set_object(self, name, x, y):
        b = self.mocap[name]; mid = self.model.body_mocapid[b]
        with self.lock:
            self.data.mocap_pos[mid][0] = x; self.data.mocap_pos[mid][1] = y

    def object_xy(self, name):
        mid = self.model.body_mocapid[self.mocap[name]]
        return tuple(float(v) for v in self.data.mocap_pos[mid][:2])

    # -- lifecycle ---------------------------------------------------------------------------------
    def start(self):
        mj = self.mujoco
        with self.lock:
            mj.mj_forward(self.model, self.data)
            start = list(self.cfg.motion.hover_start) + [TABLE_Z + self.cfg.motion.safe_height]
            start = list(self.cfg.workspace.clamp(start))
            self._solve_to(start)
            self.data.qpos[self.grip_q] = 0.044; self.data.ctrl[self.grip_act] = 0.044
            self.data.qvel[:] = 0
            mj.mj_forward(self.model, self.data)
            p, R = self.ee()
            self.mover.init_at(tuple(p))
            self._event(f"sim staged at {[round(v, 3) for v in p]} (asked {[round(v, 3) for v in start]}, miss {1000*math.dist(p, start):.0f} mm)")
        self._thread = threading.Thread(target=self._run, daemon=True, name="sim")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self.lock:
            self._snap.status = "stopped"

    def _run(self):
        mj = self.mujoco
        dt = self.model.opt.timestep
        t0 = time.perf_counter(); n = 0
        last = t0
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                with self.lock:
                    prev = self.mover.setpoint
                    sp = self.mover.step(now - last)
                    last = now
                    if prev is not None and sp is not None:
                        self.last_step_m = math.dist(prev, sp); self.max_step_m = max(self.max_step_m, self.last_step_m)
                    if sp is not None:
                        self._ik_step(sp)
                    self.data.ctrl[self.grip_act] = self.gripper_goal
                    mj.mj_step(self.model, self.data)
                    n += 1
                    p, R = self.ee()
                    width = float(np.clip((self.data.qpos[self.grip_q] - 0.022) / 0.022 * 0.04, 0.0, 0.04))
                    # grasp model: closing with an object between the fingers attaches it to the EE;
                    # opening releases it onto the table where it is
                    if self.attached is None and self.gripper_cmd < 0.01 and width < 0.03:
                        for name, bid in self.mocap.items():
                            mid = self.model.body_mocapid[bid]; op = self.data.mocap_pos[mid]
                            if math.hypot(op[0] - p[0], op[1] - p[1]) < 0.035 and abs(op[2] - p[2]) < 0.05:
                                self.attached = name; self._event(f"grasped {name}"); break
                    if self.attached is not None:
                        mid = self.model.body_mocapid[self.mocap[self.attached]]
                        if self.gripper_cmd > 0.03:
                            self.data.mocap_pos[mid][2] = TABLE_Z + self.half_h[self.attached]
                            self._event(f"released {self.attached}"); self.attached = None
                        else:
                            self.data.mocap_pos[mid][:] = [p[0], p[1], p[2]]
                    holding = self.attached is not None and width < 0.03
                    # contact force on the arm approximated as zero (mocap objects do not collide)
                    lag = math.dist(p, sp) if sp is not None else 0.0
                    self._snap = ArmSnapshot(time.time(), tuple(float(v) for v in p), width,
                                             joints=tuple(float(v) for v in self.data.qpos[self.qadr]), setpoint=sp or tuple(p),
                                             holding=holding, gripper_goal=self.gripper_cmd,
                                             goal=self.mover.goal or tuple(p), speed_cap=self.mover.speed_cap,
                                             frozen=self.mover.frozen, status="frozen" if self.mover.frozen else "live",
                                             rot=R_to_angle_axis(R))
                    self.sim_time = n * dt
                if self.realtime:
                    ahead = t0 + n * dt - time.perf_counter()
                    if ahead > 0.0005:
                        time.sleep(ahead)
        except Exception as e:
            self._event(f"sim thread error: {e!r}")
            with self.lock:
                self._snap.status, self._snap.error = "error", repr(e)

    # -- interface ---------------------------------------------------------------------------------
    def snapshot(self) -> ArmSnapshot:
        with self.lock:
            return self._snap

    def command(self, goal, speed_cap, gripper=None):
        self.mover.set_goal(goal, speed_cap)
        if gripper is not None:
            self.gripper_cmd = max(0.0, min(0.04, float(gripper)))
            self.gripper_goal = 0.022 + self.gripper_cmd * (0.022 / 0.04)

    def freeze(self, reason):
        self.mover.freeze(reason); self._event(f"freeze: {reason}")

    def resume(self):
        self.mover.resume(); self._event("resume")


class SimCamera:
    """Renders one MuJoCo camera (colour + depth in metres). Same surface as CamClient.
    `third=True` also renders the third-person view for the dashboard."""

    def __init__(self, arm: SimArm, cam_name: str = "cam", width=640, height=480, third: bool = False):
        import mujoco
        self.arm, self.w, self.h, self.cam_name = arm, width, height, cam_name
        self.cam_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = math.radians(arm.model.cam_fovy[self.cam_id])
        fy = (height / 2) / math.tan(fovy / 2)
        self.info = {"width": width, "height": height, "fx": fy, "fy": fy, "ppx": width / 2, "ppy": height / 2,
                     "depth_scale": 1.0, "serial": f"sim-{cam_name}", "name": f"MuJoCo {cam_name}"}
        self.renderer = None   # GL contexts are thread-bound on macOS: created lazily on the calling thread
        self.want_third = third
        self.n = 0
        self.third_jpeg = None

    def extrinsic_fixed(self):
        """(R, t) of a fixed camera in base frame, from MuJoCo. MuJoCo cameras look along -z of
        their frame with +y up; optical: x right = cam x, y down = -cam y, z forward = -cam z."""
        with self.arm.lock:
            self.arm.mujoco.mj_forward(self.arm.model, self.arm.data)
            cp = self.arm.data.cam_xpos[self.cam_id].copy(); cR = self.arm.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        R = np.stack([cR[:, 0], -cR[:, 1], -cR[:, 2]], 1)
        return lambda pose6, R=R, t=cp: (R, t)

    def _ensure(self):
        if self.renderer is None:
            import mujoco
            self.renderer = mujoco.Renderer(self.arm.model, self.h, self.w)
            self.depth_renderer = mujoco.Renderer(self.arm.model, self.h, self.w); self.depth_renderer.enable_depth_rendering()
            self.third = mujoco.Renderer(self.arm.model, 240, 320) if self.want_third else None

    def frame(self):
        from robojev.perception.camclient import Frame
        import cv2
        self._ensure()
        with self.arm.lock:
            self.renderer.update_scene(self.arm.data, camera=self.cam_name)
            self.depth_renderer.update_scene(self.arm.data, camera=self.cam_name)
            rgb = self.renderer.render().copy()
            depth = self.depth_renderer.render().copy()
            third = None
            if self.third is not None:
                self.third.update_scene(self.arm.data, camera="third")
                third = self.third.render().copy()
        depth[~np.isfinite(depth)] = 0
        depth[depth > 3.0] = 0
        if third is not None:
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(third, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.third_jpeg = jpg.tobytes()
        self.n += 1
        return Frame(time.time(), self.n, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth.astype(np.float32))


class Scenario(threading.Thread):
    """Moves the mocap objects on a schedule so standing orders have something to bite on.
    `drift`: after `start_s`, the flat object slides from beside the cup to just in front of it
    (between the robot and the cup) over `dur_s`, then slides back, repeatedly."""

    def __init__(self, arm: SimArm, name: str = "static", start_s: float = 10.0, dur_s: float = 12.0):
        super().__init__(daemon=True, name="scenario")
        self.arm, self.kind, self.start_s, self.dur_s = arm, name, start_s, dur_s
        self.stop_evt = threading.Event()

    def run(self):
        if self.kind == "static":
            return
        if self.kind == "intruder":
            return self._intruder()
        t0 = time.time()
        cup = self.arm.object_xy("cup")
        a = (0.30, 0.10)                       # beside
        b = (cup[0] - 0.09, cup[1] + 0.01)    # right in front of the cup, toward the robot
        while not self.stop_evt.is_set():
            t = time.time() - t0
            if t < self.start_s:
                x, y = a
            else:
                ph = ((t - self.start_s) % (2 * self.dur_s)) / self.dur_s   # 0..2
                k = ph if ph <= 1 else 2 - ph
                k = 0.5 - 0.5 * math.cos(math.pi * k)
                x, y = a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
            self.arm.set_object("phone", x, y)
            time.sleep(0.05)

    def _intruder(self):
        """At start_s a hand slides in from the robot's right to sit between the gripper and the cup
        for 6 s, then leaves. At start_s + 16 s the cup is relocated 8 cm to the robot's left over
        2 s (someone moved it). Repeats every 40 s."""
        t0 = time.time()
        park = (0.30, -0.60)
        while not self.stop_evt.is_set():
            t = (time.time() - t0)
            cyc = (t - self.start_s) % 40.0 if t >= self.start_s else -1
            cup = self.arm.object_xy("cup")
            ee = self.arm.snapshot().ee
            # between the gripper and the cup, but a hand's width clear of both (11 cm from the cup)
            dx, dy = ee[0] - cup[0], ee[1] - cup[1]
            L = math.hypot(dx, dy) or 1.0
            between = (cup[0] + dx / L * 0.11, cup[1] + dy / L * 0.11)
            if 0 <= cyc < 2:          # slide in
                k = 0.5 - 0.5 * math.cos(math.pi * cyc / 2)
                self.arm.set_object("hand", park[0] + (between[0] - park[0]) * k, park[1] + (between[1] - park[1]) * k)
            elif 2 <= cyc < 8:        # stay between gripper and cup
                self.arm.set_object("hand", *between)
            elif 8 <= cyc < 10:       # leave
                k = 0.5 - 0.5 * math.cos(math.pi * (cyc - 8) / 2)
                hx, hy = self.arm.object_xy("hand")
                self.arm.set_object("hand", hx + (park[0] - hx) * k * 0.5, hy + (park[1] - hy) * k * 0.5)
            elif 10 <= cyc < 16:
                self.arm.set_object("hand", *park)
            elif 16 <= cyc < 18 and self.arm.attached != "cup":   # cup relocated (unless the arm holds it)
                if not hasattr(self, "_cup0") or self._cup0 is None:
                    self._cup0 = cup
                k = 0.5 - 0.5 * math.cos(math.pi * (cyc - 16) / 2)
                self.arm.set_object("cup", self._cup0[0], self._cup0[1] + 0.08 * k)
            elif cyc >= 18:
                self._cup0 = None
            time.sleep(0.05)
