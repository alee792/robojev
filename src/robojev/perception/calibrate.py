"""Calibrate a fixed (overhead) depth camera to the arm base frame without a target board.

Unknowns: the camera's pose in base frame (6 DOF). Two constraints do it:
  1. The table plane, seen in the camera's own depth: fixes roll, pitch and height (3 DOF), given
     the table's base-frame height (from the wrist camera's plane fit).
  2. Objects that both cameras see: the wrist camera gives their base-frame xy; the fixed camera
     gives their xy in a table-aligned frame. A 2D rigid fit (yaw + x,y) closes the remaining 3 DOF.
     Two objects determine it; one object gives translation only (yaw assumed 0).

Usage in code: `ext = solve(depth_m, intr, table_z, fixed_dets_xy, wrist_xy)`; persist with
`save(path, R, t)` and load with `load(path)` -> extrinsic callable for Detector(extrinsic=...).
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np

from robojev.perception.detect import Detector, fit_plane
from robojev.perception.geometry import Intrinsics


def _align_to_plane(normal, d, table_z):
    """(R1, t1) mapping camera-frame points into a frame where the table is z = table_z and up is +z."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    # in the camera's optical frame the table normal points roughly toward the camera (-z); we want
    # the base-frame normal +z. Rotate n onto +z.
    z = np.array([0, 0, 1.0])
    if n @ z < 0:
        n = -n; d = -d
    v = np.cross(n, z); s = np.linalg.norm(v); c = float(n @ z)
    if s < 1e-9:
        R1 = np.eye(3)
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + K + K @ K * ((1 - c) / s**2)
    # a point on the plane maps to z = table_z: plane is n.p + d = 0 -> after R1, z' + d = 0
    t1 = np.array([0, 0, table_z + d])
    return R1, t1


def _rigid_2d(A, B):
    """Kabsch in 2D: R (2x2), t with B ~ R A + t."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1; Rm = Vt.T @ U.T
    t = cb - Rm @ ca
    return Rm, t


def solve(depth_m: np.ndarray, intr: Intrinsics, table_z: float, fixed_dets: list, wrist_ents: list):
    """fixed_dets: [(x, y, height)] in the *table-aligned* frame is computed here from `depth_m` and
    the plane; caller passes detections already in that frame via `detect_aligned`. wrist_ents:
    [(x, y, height)] in base frame. Returns (R, t, report)."""
    raise NotImplementedError  # use calibrate_from_frames


def detect_aligned(color, depth_m, intr: Intrinsics, table_z: float):
    """Plane-align a fixed camera's cloud and detect objects in that frame. Returns (dets, R1, t1, info)."""
    pts, _ = intr.deproject(depth_m, 4)
    pts = pts[(pts[:, 2] > 0.2) & (pts[:, 2] < 2.5)]
    nv, d, inl = fit_plane(pts, iters=80, thresh=0.01)
    if nv is None:
        return [], None, None, {"error": "no plane"}
    R1, t1 = _align_to_plane(nv, d, table_z)
    det = Detector(intr, workspace_xy=((-1.0, 1.0), (-1.0, 1.0)), extrinsic=lambda p6: (R1, t1), finger_mask=False,
                   table_z=table_z)
    # no arm mask here (pose unknown in this frame): pass an EE far away so the corridor is empty
    dets, info = det.run(color, depth_m, [5.0, 5.0, 5.0, 0, 0, 0])
    return dets, R1, t1, info | {"plane_inliers": int(inl.sum()), "normal": nv.tolist()}


def calibrate_from_frames(color, depth_m, intr: Intrinsics, table_z: float, wrist_ents: list[tuple[float, float, float]]):
    """wrist_ents: [(x, y, height)] in base frame from the wrist camera's tracker. Returns (R, t, report)."""
    dets, R1, t1, info = detect_aligned(color, depth_m, intr, table_z)
    if R1 is None:
        return None, None, info
    cands = [(d.base_xyz[0], d.base_xyz[1], d.height) for d in dets if d.height > 0.015]
    report = {"fixed_dets": [(round(x, 3), round(y, 3), round(h, 3)) for x, y, h in cands], "wrist": wrist_ents} | info
    if not cands or not wrist_ents:
        return None, None, report | {"error": "need objects seen by both cameras"}
    # assignment: try all injective maps wrist->fixed with height agreement, pick lowest residual
    best = None
    k = min(len(wrist_ents), len(cands))
    for wsub in itertools.combinations(range(len(wrist_ents)), k):
        for perm in itertools.permutations(range(len(cands)), k):
            if any(abs(wrist_ents[i][2] - cands[j][2]) > 0.05 for i, j in zip(wsub, perm)):
                continue
            A = [cands[j][:2] for j in perm]; B = [wrist_ents[i][:2] for i in wsub]
            if k == 1:
                Rm = np.eye(2); t = np.asarray(B[0]) - np.asarray(A[0])
            else:
                Rm, t = _rigid_2d(A, B)
            res = float(np.mean([np.linalg.norm(Rm @ np.asarray(a) + t - np.asarray(b)) for a, b in zip(A, B)]))
            if best is None or res < best[0]:
                best = (res, Rm, t, list(zip(wsub, perm)))
    if best is None:
        return None, None, report | {"error": "no height-consistent assignment"}
    res, Rm, t2, pairs = best
    Rz = np.eye(3); Rz[:2, :2] = Rm
    R = Rz @ R1
    t = Rz @ t1 + np.array([t2[0], t2[1], 0.0])
    yaw = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
    return R, t, report | {"residual_m": round(res, 4), "pairs": pairs, "yaw_deg": round(yaw, 2), "n_pairs": k,
                           "warning": "one object: yaw assumed 0" if k == 1 else None}


def save(path, R, t, report=None):
    json.dump({"R": np.asarray(R).tolist(), "t": np.asarray(t).tolist(), "report": report}, open(path, "w"), indent=1, default=str)


def load(path):
    o = json.load(open(path)); R, t = np.array(o["R"]), np.array(o["t"])
    return lambda pose6, R=R, t=t: (R, t)
