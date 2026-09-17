"""Camera <-> base frame geometry for the wrist D405.

Chain: pixel + depth -> camera optical frame -> EE frame (fixed mount) -> base frame (arm pose).

Mount, from the follower's MJCF (wxai_follower.xml): site `camera_color_frame` in link_6 at
(0.05864, 0.009, 0.05427) with quat (0.9848, 0, 0.1736, 0) = +20 deg about y; the EE frame is
link_6 shifted 0.156 m along x (t_flange_tool from the driver agrees). Optical convention: cam z =
frame x (forward), cam x = -frame y (right), cam y = -frame z (down). Verified against the table
plane on 2026-09-17 (see scripts/probe_camera.py); adjust CAM_IN_EE_* if the plane comes out tilted.
"""
from __future__ import annotations

import math

import numpy as np

CAM_IN_EE_POS = np.array([0.05864 - 0.156062, 0.009, 0.05427])
CAM_PITCH_DEG = 20.0  # frame x tilted down by this much


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# frame->optical: columns are optical axes expressed in the (ROS-style) camera frame.
# optical x = -frame y, optical y = -frame z, optical z = frame x
R_FRAME_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # columns = optical axes in frame coords


def cam_to_ee() -> tuple[np.ndarray, np.ndarray]:
    """(R, t): p_ee = R @ p_opt + t."""
    R = rot_y(CAM_PITCH_DEG) @ R_FRAME_OPT
    return R, CAM_IN_EE_POS.copy()


def angle_axis_to_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def ee_to_base(pose6) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) from the driver's [x,y,z, rx,ry,rz] (angle-axis) EE pose."""
    return angle_axis_to_R(pose6[3:6]), np.asarray(pose6[:3], float)


def cam_to_base(pose6):
    Rce, tce = cam_to_ee()
    Reb, teb = ee_to_base(pose6)
    return Reb @ Rce, Reb @ tce + teb


class Intrinsics:
    def __init__(self, info: dict):
        self.fx, self.fy, self.ppx, self.ppy = info["fx"], info["fy"], info["ppx"], info["ppy"]
        self.w, self.h = info["width"], info["height"]

    def deproject(self, depth_m: np.ndarray, stride: int = 4):
        """Point cloud (N,3) in the optical frame plus the (u,v) pixel of each point. Skips depth 0.
        Distortion is ignored (D405 coefficients are small at the image centre)."""
        v, u = np.mgrid[0:self.h:stride, 0:self.w:stride]
        z = depth_m[::stride, ::stride]
        m = z > 0
        u, v, z = u[m], v[m], z[m]
        x = (u - self.ppx) / self.fx * z
        y = (v - self.ppy) / self.fy * z
        return np.stack([x, y, z], 1), np.stack([u, v], 1)

    def project(self, p_opt: np.ndarray):
        x, y, z = p_opt
        return (self.ppx + self.fx * x / z, self.ppy + self.fy * y / z)
