"""EE orientation for the trossen_arm driver: the policies' closing-axis yaw to and from the driver's
angle-axis (rotation vector) and back.

Frames. Base: +x forward, +y left, +z up. EE (trossen_arm_mujoco's `ee_site`, which the sim's
`_down()` uses too): +x along the fingers, +y the closing axis (from one fingertip to the other).
The driver reports and takes `[x, y, z, rx, ry, rz]` with `(rx, ry, rz)` the rotation vector of
R_base<-ee, the matrix whose columns are the EE axes in the base frame: v1 streamed `[0, pitch, 0]`
= Ry(pitch) and the fingers pitched down about base y (arm/real.py, verified at 75 deg).

A straight-down grasp with the closing axis along yaw psi is Rz(psi - pi/2) . Ry(pi/2): Ry(pi/2)
points the fingers down with the closing axis along base +y (psi = pi/2, v1's `[0, pi/2, 0]`), and
Rz turns it about the vertical. The same matrix is the sim's `_down(psi)`, so a yaw the policies
choose means the same thing on both backends.

Verify on hardware (D1): (1) that the driver's rotation triple is a rotation vector and not
roll-pitch-yaw (context/09 quotes a plugin doc saying RPY; the two coincide for a pure pitch, which
is all v1 ever streamed); (2) the sign of the yaw, visible at once as the wrist rolling the wrong way
for a yawed block; (3) that the controller accepts a 90 deg pitch streamed (v1 verified 75 deg).
"""
from __future__ import annotations

import math

import numpy as np

AngleAxis = tuple[float, float, float]
DOWN_PITCH = math.pi / 2          # fingers straight down


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def from_angle_axis(aa) -> np.ndarray:
    """Rodrigues: the rotation matrix of rotation vector `aa` (axis times angle, rad)."""
    v = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def to_angle_axis(R) -> AngleAxis:
    """The rotation vector of matrix `R`, angle in [0, pi]. At exactly pi the axis sign is a free
    choice (both give the same rotation); the one with the largest component positive is returned."""
    R = np.asarray(R, dtype=float)
    cos_th = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(cos_th)
    if th < 1e-9:
        return (0.0, 0.0, 0.0)
    if math.pi - th < 1e-6:
        # R = 2 k k^T - I: a column of (R + I) / 2 is k times its own component
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        if k[int(np.argmax(np.abs(k)))] < 0:
            k = -k
        return tuple(float(v) for v in k * math.pi)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return tuple(float(v) for v in k * th)


def down_matrix(closing_yaw: float, pitch: float = DOWN_PITCH) -> np.ndarray:
    """R_base<-ee = Rz(closing_yaw - pi/2) . Ry(pitch). At `pitch` = pi/2 the fingers point straight
    down and the closing axis lies along `closing_yaw` in the xy plane (the sim's `_down`). At a
    smaller pitch the fingers point down-and-forward toward azimuth closing_yaw - pi/2, which is zero
    wrist roll for an arm reaching along that azimuth; at closing_yaw = pi/2 it is v1's Ry(pitch)."""
    return rot_z(closing_yaw - math.pi / 2) @ rot_y(pitch)


def down_orientation(closing_yaw: float, pitch: float = DOWN_PITCH) -> AngleAxis:
    """The driver's rotation vector for `down_matrix(closing_yaw, pitch)`."""
    return to_angle_axis(down_matrix(closing_yaw, pitch))


def closing_yaw_of(aa) -> float:
    """Direction of the closing axis (EE y) in the base xy plane, rad: `atan2(R[1,1], R[0,1])`, the
    same read the sim takes off `site_xmat`. Undefined (0 returned) when the closing axis is vertical."""
    R = from_angle_axis(aa)
    return math.atan2(R[1, 1], R[0, 1])


def pitch_of(aa) -> float:
    """How far the fingers (EE x) point below the horizontal, rad: pi/2 straight down, 0 level."""
    R = from_angle_axis(aa)
    x = R[:, 0]
    return math.atan2(-x[2], math.hypot(x[0], x[1]))
