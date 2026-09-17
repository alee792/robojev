"""Depth-based tabletop detector: table plane -> points above it -> clusters -> entities.

Colour-agnostic (a white paper cup on a pale table is a bad colour target), which is why the
depth image is the primary cue. Each cluster gets a centroid in base frame, footprint, height and
a mean colour name; labels are assigned by the tracker (memory.py), not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from robojev.perception.geometry import angle_axis_to_R, Intrinsics, cam_to_base


@dataclass
class Detection:
    xyz: tuple[float, float, float]     # centroid, base frame (z = mid height)
    base_xyz: tuple[float, float, float]  # footprint centre on the table plane
    height: float
    width: float                          # footprint diameter (m)
    color_bgr: tuple[int, int, int]
    color_name: str
    n_points: int
    pixel: tuple[int, int]                # image centre of the blob
    partial: bool = False                 # blob touches the image border: its centroid is biased
    camera: str | None = None             # which camera produced it


COLOR_NAMES = [  # (name, hsv centre) rough buckets
    ("white", None), ("black", None), ("gray", None),
    ("red", 0), ("orange", 15), ("yellow", 28), ("green", 60), ("cyan", 90), ("blue", 115), ("purple", 140), ("pink", 165),
]


def color_name(bgr) -> str:
    hsv = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 170 else ("light gray" if v > 120 else "dark gray")
    if v < 90:
        return "dark " + _hue_name(h)
    if s < 90 and v > 150:
        return "pale " + _hue_name(h)
    return _hue_name(h)


def _hue_name(h: int) -> str:
    best, bd = "red", 999
    for name, hc in COLOR_NAMES:
        if hc is None:
            continue
        d = min(abs(h - hc), 180 - abs(h - hc))
        if d < bd:
            best, bd = name, d
    return best


def fit_plane(points: np.ndarray, iters: int = 60, thresh: float = 0.008, rng=np.random):
    """RANSAC plane. Returns (normal, d) with n.p + d = 0, normal pointing +z, and inlier mask."""
    best = (None, None, np.zeros(len(points), bool))
    n = len(points)
    if n < 50:
        return best
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nv)
        if norm < 1e-9:
            continue
        nv /= norm
        d = -nv @ p0
        dist = np.abs(points @ nv + d)
        inl = dist < thresh
        if inl.sum() > best[2].sum():
            best = (nv, d, inl)
    nv, d, inl = best
    if nv is None:
        return best
    # refine with SVD on inliers
    P = points[inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    nv = vt[-1]
    if nv[2] < 0:
        nv = -nv
    d = -nv @ c
    inl = np.abs(points @ nv + d) < thresh
    return nv, d, inl


class Detector:
    def __init__(self, intr: Intrinsics, workspace_xy=((0.10, 0.80), (-0.40, 0.40)),
                 min_height=0.02, max_height=0.30, stride=4, table_z: float | None = None,
                 extrinsic=None, finger_mask: bool = True):
        self.intr = intr
        self.ws = workspace_xy
        self.min_h, self.max_h = min_height, max_height
        self.stride = stride
        self.table_z = table_z      # if known, use it instead of fitting every frame
        self.last_plane = None      # (normal, d) in base frame
        self.extrinsic = extrinsic or cam_to_base   # pose6 -> (R, t) of the camera in base frame
        self.finger_mask = finger_mask               # wrist camera: the fingers are always in view

    def run(self, color: np.ndarray, depth_m: np.ndarray, pose6, holding: bool = False) -> tuple[list[Detection], dict]:
        pts_opt, uv = self.intr.deproject(depth_m, self.stride)
        # D405 valid range ~0.07..0.5+ m; drop far/noisy points
        m = (pts_opt[:, 2] > 0.07) & (pts_opt[:, 2] < 1.0)
        pts_opt, uv = pts_opt[m], uv[m]
        R, t = self.extrinsic(pose6)
        P = pts_opt @ R.T + t
        ee = np.asarray(pose6[:3], float)
        axis = R[:, 2]                                   # optical axis in base frame
        oblique = bool(self.finger_mask and axis[2] > -0.82)   # more than ~35 deg off vertical: a side view
        look_u = axis[:2] / max(1e-6, float(np.hypot(*axis[:2])))
        if holding:
            # the carried object rides with the gripper; code knows where it is, and its points must
            # not pollute other tracks (it dragged the phone's track 4 cm in pick-and-place run 2)
            # a cylinder around the gripper, but only down to ~10 cm below the EE point: flat objects
            # the arm flies over must keep their points (the 8 cm disc ate half the phone in run 3)
            carried = (np.hypot(P[:, 0] - ee[0], P[:, 1] - ee[1]) < 0.07) & (P[:, 2] > ee[2] - 0.12)
            P, uv = P[~carried], uv[~carried]
        if self.finger_mask:
            # the fingers, gripper body and wrist lie along the tool axis behind the fingertips
            # (the EE point): mask a cylinder from 2 cm ahead of the tips to 22 cm behind them.
            # Pointing down this is the old column above the EE; tilted for a side grasp it is the
            # slanted body that showed up as three "black cup-like objects" (real run 4).
            # Nothing ahead of the tips is masked, so a target under or in front of the gripper stays.
            d = angle_axis_to_R(pose6[3:6])[:, 0]
            v = P - ee
            along = v @ d
            perp = np.linalg.norm(v - np.outer(along, d), axis=1)
            # pads: 6.5 cm radius; further back the open carriages stick out ~9 cm sideways (they
            # were 'black cup-like objects' at +-9 cm behind the tips at the survey pose: real run 13)
            fingers = ((along > -0.04) & (along < 0.02) & (perp < 0.065)) | ((along > -0.22) & (along <= -0.04) & (perp < 0.11))
            P, uv = P[~fingers], uv[~fingers]
        else:
            # a fixed camera sees the whole arm: mask the base column and everything at or above
            # the EE height within a corridor from the base to the EE (the links are up there;
            # objects never are, because the z floor keeps the EE above the tallest object)
            base = np.hypot(P[:, 0], P[:, 1]) < 0.10
            seg = ee[:2]; L = np.linalg.norm(seg)
            if L > 1e-6:
                u = seg / L
                along = P[:, :2] @ u
                perp = np.abs(P[:, 0] * u[1] - P[:, 1] * u[0])
                corridor = (along > -0.05) & (along < L + 0.08) & (perp < 0.09) & (P[:, 2] > ee[2] - 0.03)
            else:
                corridor = np.zeros(len(P), bool)
            keep = ~(base | corridor)
            P, uv = P[keep], uv[keep]
        info = {"n_points": int(len(P))}
        if len(P) < 100:
            return [], info | {"error": "too few depth points"}
        if self.table_z is None:
            # fit the plane on points in the workspace footprint, lowish
            cand = (P[:, 0] > self.ws[0][0]) & (P[:, 0] < self.ws[0][1]) & (P[:, 1] > self.ws[1][0]) & (P[:, 1] < self.ws[1][1])
            nv, d, inl = fit_plane(P[cand]) if cand.sum() > 50 else (None, None, None)
            if nv is None:
                return [], info | {"error": "no table plane"}
            self.last_plane = (nv, d)
            info["plane_normal"] = [round(float(x), 3) for x in nv]
            info["plane_z_at_origin"] = float(-d / nv[2]) if abs(nv[2]) > 1e-6 else None
            info["plane_tilt_deg"] = float(np.degrees(np.arccos(min(1.0, abs(nv[2])))))
            height = P @ nv + d
        else:
            height = P[:, 2] - self.table_z
            info["plane_z_at_origin"] = self.table_z
        above = (height > self.min_h) & (height < self.max_h)
        inws = (P[:, 0] > self.ws[0][0]) & (P[:, 0] < self.ws[0][1]) & (P[:, 1] > self.ws[1][0]) & (P[:, 1] < self.ws[1][1])
        sel = above & inws
        info["n_above"] = int(sel.sum())
        if sel.sum() < 10:
            return [], info
        Q, uvq, hq = P[sel], uv[sel], height[sel]
        # a top-down camera sees flat tops, so touching objects of different height separate at a
        # height gap; a wrist camera sees sloped walls, where height splitting only fragments objects
        labels = _cluster_xy(Q[:, :2], cell=0.02, heights=None if self.finger_mask else hq, dh=0.035)
        dets = []
        for lab in np.unique(labels):
            k = labels == lab
            if k.sum() < 15:
                continue
            pts, px, hh = Q[k], uvq[k], hq[k]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            top = float(np.percentile(hh, 95))
            base_z = float(cx * 0 + (self.table_z if self.table_z is not None else -(self.last_plane[1] + self.last_plane[0][0] * cx + self.last_plane[0][1] * cy) / self.last_plane[0][2]))
            spread = np.percentile(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy), 90) * 2
            if oblique:
                # looking across the table, a blob's points are its near wall plus the inside of its
                # far wall: the centroid drifts away from the camera as the camera comes closer
                # (the cup "moved" 5 cm ahead of the advancing gripper, real run 4). The near edge
                # and the width across the view are what the camera measures well: place the
                # centre one half-width behind the near edge, along the look direction.
                along = pts[:, :2] @ look_u
                across = pts[:, 0] * look_u[1] - pts[:, 1] * look_u[0]
                width_across = float(np.percentile(across, 95) - np.percentile(across, 5))
                near = float(np.percentile(along, 5))
                mid_across = float(np.median(across))
                c = near + width_across / 2
                cx, cy = float(c * look_u[0] + mid_across * look_u[1]), float(c * look_u[1] - mid_across * look_u[0])   # along*u + across*(u1, -u0)
                spread = width_across
            u0, v0 = int(px[:, 0].mean()), int(px[:, 1].mean())
            margin = 3 * self.stride
            partial = bool(px[:, 0].min() < margin or px[:, 1].min() < margin
                           or px[:, 0].max() > self.intr.w - margin or px[:, 1].max() > self.intr.h - margin)
            if not self.finger_mask:
                # from a fixed camera the arm hides part of anything under it: a blob whose centre
                # sits within the arm's corridor is a partial view (its centroid drifts away from the arm)
                seg = ee[:2]; L = float(np.hypot(*seg))
                if L > 1e-6:
                    u = seg / L; along = float(np.array([cx, cy]) @ u); perp = abs(float(cx * u[1] - cy * u[0]))
                    if -0.02 < along < L + 0.10 and perp < 0.11:
                        partial = True
            patch = color[max(0, v0 - 6):v0 + 6, max(0, u0 - 6):u0 + 6].reshape(-1, 3)
            bgr = tuple(int(x) for x in np.median(patch, 0)) if len(patch) else (128, 128, 128)
            dets.append(Detection(xyz=(float(cx), float(cy), base_z + top / 2), base_xyz=(float(cx), float(cy), base_z),
                                  height=top, width=float(spread), color_bgr=bgr, color_name=color_name(bgr),
                                  n_points=int(k.sum()), pixel=(u0, v0), partial=partial))
        dets.sort(key=lambda d: -d.n_points)
        return dets, info


def _cluster_xy(xy: np.ndarray, cell: float, heights: np.ndarray | None = None, dh: float = 0.035) -> np.ndarray:
    """Grid-based connected components in the xy plane (8-neighbour), height-aware: two occupied
    cells join only if their top heights agree within `dh`, so a hand touching a cup stays a
    separate object (a flat grid merged hand+cup+phone into one 26 cm blob in the intruder run)."""
    g = np.floor(xy / cell).astype(int)
    g -= g.min(0)
    H, W = g[:, 0].max() + 1, g[:, 1].max() + 1
    occ = np.zeros((H, W), np.uint8)
    occ[g[:, 0], g[:, 1]] = 1
    if heights is None:
        n, comp = cv2.connectedComponents(occ, connectivity=8)
        return comp[g[:, 0], g[:, 1]]
    # plain components first, then split any component whose per-cell top heights show a clear gap
    top = np.full((H, W), -1.0)
    np.maximum.at(top, (g[:, 0], g[:, 1]), heights)
    n, comp = cv2.connectedComponents(occ, connectivity=8)
    out = np.zeros((H, W), int)
    next_id = 1
    for c in range(1, n):
        cells = np.argwhere(comp == c)
        tops = top[cells[:, 0], cells[:, 1]]
        order = np.argsort(tops)
        st = tops[order]
        gaps = np.diff(st)
        split_at = None
        if len(st) >= 8:
            k = int(np.argmax(gaps))
            if gaps[k] >= dh and 4 <= k + 1 <= len(st) - 4:
                split_at = st[k]
        if split_at is None:
            out[cells[:, 0], cells[:, 1]] = next_id; next_id += 1
        else:
            low = tops <= split_at
            # each side re-clustered on its own so two low objects on either side of a tall one stay apart
            for mask in (low, ~low):
                sub = np.zeros((H, W), np.uint8)
                sub[cells[mask, 0], cells[mask, 1]] = 1
                m, sc = cv2.connectedComponents(sub, connectivity=8)
                for q in range(1, m):
                    out[sc == q] = next_id; next_id += 1
    return out[g[:, 0], g[:, 1]]
