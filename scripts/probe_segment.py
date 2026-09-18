"""Grab one wrist frame, run the Segmenter on it, save a coloured overlay and time it.

No motion: the camera server is read-only, and --detect only *reads* the arm's cartesian pose.

Usage:
    .venv/bin/python scripts/probe_segment.py [out.jpg] [--model FastSAM-s.pt] [--conf 0.5]
                                              [--device auto] [--reps 20] [--detect]

--detect additionally reads the EE pose from the driver and runs Detector twice on the same frame,
with and without the segmenter, so you can see which blobs the colour masks split apart.
"""
import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

from robojev.perception.camclient import CamClient
from robojev.perception.geometry import Intrinsics
from robojev.perception.segment import Segmenter, SegmenterUnavailable

DEFAULT_OUT = ("/private/tmp/claude-501/-Users-anthonylee-code-robojev--claude-worktrees-"
               "widowx-doom-loop-fc1569/f358cce5-d87f-4631-9b8a-dff497cbf951/scratchpad/segment_overlay.jpg")

PALETTE = [(255, 64, 64), (64, 200, 64), (64, 64, 255), (255, 210, 0), (255, 0, 255), (0, 220, 220),
           (255, 140, 0), (160, 64, 255), (0, 140, 255), (140, 255, 64), (200, 80, 140), (80, 200, 200)]

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=DEFAULT_OUT)
ap.add_argument("--model", default="FastSAM-s.pt")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--iou", type=float, default=0.6)
ap.add_argument("--device", default="auto")
ap.add_argument("--reps", type=int, default=20, help="timed repetitions after warm-up")
ap.add_argument("--cam", default="http://127.0.0.1:8765")
ap.add_argument("--image", help="segment this saved frame instead of grabbing a live one")
ap.add_argument("--save-frame", help="also write the grabbed colour frame here")
ap.add_argument("--detect", action="store_true", help="also run Detector with/without the segmenter")
a = ap.parse_args()

if a.image:
    color = cv2.imread(a.image)
    if color is None:
        sys.exit(f"cannot read {a.image}")
    cam, f = None, None
    print(f"saved frame {a.image} colour {color.shape}")
else:
    cam = CamClient(a.cam)
    f = cam.frame()
    if f is None:
        sys.exit("no frame from the camera server")
    color = f.color
    print(f"frame {f.n} colour {color.shape}")
    if a.save_frame:
        cv2.imwrite(a.save_frame, color)

try:
    seg = Segmenter(model=a.model, device=a.device, conf=a.conf, iou=a.iou)
except SegmenterUnavailable as e:
    sys.exit(f"segmenter unavailable: {e}")
print(f"{seg.model_name} on {seg.device} (warm-up {seg.warmup_s * 1000:.0f} ms), "
      f"weights {seg.weights}")

t0 = time.perf_counter()
masks = seg.run(color)
first_ms = (time.perf_counter() - t0) * 1000

ts = []
for _ in range(max(1, a.reps)):
    t0 = time.perf_counter()
    seg.run(color)
    ts.append(time.perf_counter() - t0)
ts = np.array(ts)
med = float(np.median(ts))
print(f"{len(masks)} masks; first {first_ms:.0f} ms, median of {len(ts)} "
      f"{med * 1000:.0f} ms -> {1 / med:.2f} Hz (min {ts.min() * 1000:.0f}, max {ts.max() * 1000:.0f})")

for i, m in enumerate(masks):
    x1, y1, x2, y2 = m.bbox
    print(f"  mask{i:2d} {m.label:<10s} score {m.score:.2f} px {int(m.mask.sum()):6d} "
          f"bbox ({x1},{y1},{x2},{y2})")

# Draw the label image, not the raw stack: one colour per pixel, decided the same way the detector
# decides it (ascending score, so the more confident mask wins an overlap). Stacking the masks in
# list order instead would let a low-score background mask paint over the object that beat it.
lbl, _ = seg.label_image(color, masks)
vis = color.copy()
pal = np.array([(0, 0, 0)] + [PALETTE[i % len(PALETTE)] for i in range(len(masks))], np.uint8)
inside = lbl > 0
vis[inside] = (0.5 * vis[inside] + 0.5 * pal[lbl[inside]]).astype(np.uint8)
for i, m in enumerate(masks):
    won = lbl == i + 1
    area = int(won.sum())
    if area < 300:                      # a mask reduced to slivers by its neighbours: do not label it
        continue
    c = PALETTE[i % len(PALETTE)]
    ys, xs = np.nonzero(won)
    cv2.rectangle(vis, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), c, 1)
    txt = f"{i}:{m.label} {m.score:.2f}"
    org = (max(2, int(xs.mean()) - 30), int(np.median(ys)))
    cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
# the detector ignores everything below this line (finger pads, garbage depth at minimum range)
h = color.shape[0]
cv2.line(vis, (0, int(0.6 * h)), (vis.shape[1], int(0.6 * h)), (255, 255, 255), 1)
cv2.putText(vis, "detector drops below", (4, int(0.6 * h) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(a.out, vis)
print("wrote", a.out)

if a.detect:
    if f is None:
        sys.exit("--detect needs a live frame (drop --image)")
    if f.depth_m is None:
        sys.exit("--detect needs a depth camera")
    from robojev.perception.detect import Detector
    import trossen_arm as t                       # read-only: no motion command is ever sent
    d = t.TrossenArmDriver()
    d.configure(t.Model.wxai_v0, t.StandardEndEffector.wxai_v0_follower, "192.168.1.5", False)
    pose = list(d.get_cartesian_positions())
    d.cleanup()
    print("EE pose", [round(x, 4) for x in pose])
    intr = Intrinsics(cam.info)
    for tag, s in (("depth only", None), ("with masks", seg)):
        det = Detector(intr, segmenter=s)
        t0 = time.perf_counter()
        dets, info = det.run(f.color, f.depth_m, pose)
        print(f"\n{tag}: {len(dets)} detections in {(time.perf_counter() - t0) * 1000:.0f} ms  {json.dumps(info)}")
        for i, o in enumerate(dets):
            print(f"  obj{i}: xy=({o.base_xyz[0]:.3f},{o.base_xyz[1]:.3f}) h={o.height:.3f} w={o.width:.3f} "
                  f"{o.color_name} n={o.n_points} px={o.pixel} flat={o.flat} seg={o.seg_label}")
