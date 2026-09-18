"""Instance masks from the colour image, to split a depth blob that holds two touching objects.

The depth detector (detect.py) clusters points in xy, so a paper cup standing against a cardboard
box comes out as one 16 cm "boxy object". Colour sees the seam that depth cannot. This wraps an
ultralytics segmentation model and hands `Detector` a list of per-instance boolean masks; the
detector uses the mask id as the cluster label for every point whose pixel falls inside one.

Model choice (measured on a real wrist frame, 2026-09-17, scripts/probe_segment.py):
FastSAM-s, not yolo11n-seg. COCO has no class for a cardboard box or a mouse mat, so yolo11n-seg
found neither: at conf 0.25 it returned five masks, labelled the two gripper fingers "chair", found
only a cup on a desk in the far background, and missed both objects on the table; the near paper cup
only appeared at conf 0.10, as "chair"/"bowl" at 0.16. FastSAM is class-agnostic and gave the box
(score 0.94), the cup (0.90) and the mouse mat (0.60) as three separate masks, each the only mask
covering its object. The price is that it also segments the whole cluttered background, which costs
nothing here: the detector only looks up pixels of points that are already above the table plane and
inside the workspace, so background masks are never consulted.

Labels are therefore always "object" (FastSAM's only class). A COCO model would put a real name
there, which is why `Mask.label` and `Detection.seg_label` exist at all.
"""
from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass

import numpy as np

# where weights are cached; never inside the repo, and overridable for a sandboxed run
WEIGHTS_DIR = pathlib.Path(os.environ.get("ROBOJEV_WEIGHTS_DIR", pathlib.Path.home() / ".cache" / "robojev"))


@dataclass
class Mask:
    mask: np.ndarray                      # bool, HxW, True inside the instance
    label: str                            # class name, or "object" for a class-agnostic model
    score: float
    bbox: tuple[int, int, int, int]       # (x1, y1, x2, y2) pixels, x2/y2 exclusive


class SegmenterUnavailable(RuntimeError):
    """ultralytics missing, weights undownloadable, or the model failed its warm-up.

    Always raised from `Segmenter(...)`, never from `run()`: a robot loop that constructed a
    segmenter is entitled to assume it works.
    """


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError as e:                                # pragma: no cover - torch ships with ultralytics
        raise SegmenterUnavailable(f"torch is not installed: {e}") from e
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Segmenter:
    """Class-agnostic instance masks for a BGR frame.

        seg = Segmenter()                     # FastSAM-s on MPS if available, else CPU
        masks = seg.run(frame.color)          # list[Mask], highest score first

    Construction downloads the weights if needed and runs a warm-up inference, so any failure
    (no ultralytics, no network, a device that cannot run the model) surfaces here as
    SegmenterUnavailable. `run()` raises only on a malformed frame.
    """

    def __init__(self, model: str = "FastSAM-s.pt", device: str = "auto", conf: float = 0.5,
                 iou: float = 0.6, imgsz: int = 640, max_area_frac: float = 0.9,
                 min_area_px: int = 200, weights_dir: pathlib.Path | None = None):
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        # only a guard against a degenerate whole-frame mask: with smallest-wins painting a large
        # parent mask is harmless, because everything inside it takes its own label anyway
        self.max_area_frac = max_area_frac
        self.min_area_px = min_area_px          # drop specks that cannot hold enough depth points
        try:
            from ultralytics import FastSAM, YOLO
        except ImportError as e:
            raise SegmenterUnavailable(
                "ultralytics is not installed; install the seg extra: uv pip install -e '.[seg]'"
            ) from e

        wdir = pathlib.Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
        try:
            wdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SegmenterUnavailable(f"cannot create the weights directory {wdir}: {e}") from e
        # pass an absolute path so ultralytics downloads there; a bare name lands in the cwd
        path = pathlib.Path(model)
        self.weights = str(path if path.is_absolute() else wdir / path.name)
        self.model_name = path.name

        cls = FastSAM if "fastsam" in self.model_name.lower() else YOLO
        try:
            self.model = cls(self.weights)
        except Exception as e:                  # download failure, corrupt file, bad name
            raise SegmenterUnavailable(
                f"could not load segmentation weights {self.model_name} into {wdir}: "
                f"{type(e).__name__}: {e}"
            ) from e

        want = _pick_device(device)
        dummy = np.zeros((64, 64, 3), np.uint8)
        self.device = None
        errs = []
        for dev in ([want] if want == "cpu" else [want, "cpu"]):
            try:
                t0 = time.perf_counter()
                self.model.predict(dummy, device=dev, conf=self.conf, iou=self.iou,
                                   imgsz=self.imgsz, retina_masks=True, verbose=False)
                self.warmup_s = time.perf_counter() - t0
                self.device = dev
                break
            except Exception as e:              # an MPS op the model needs may be unimplemented
                errs.append(f"{dev}: {type(e).__name__}: {e}")
        if self.device is None:
            raise SegmenterUnavailable(
                f"{self.model_name} loaded but failed its warm-up inference — " + "; ".join(errs)
            )

    def run(self, color_bgr: np.ndarray) -> list[Mask]:
        """Instance masks for one BGR frame, highest score first."""
        if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got {None if color_bgr is None else color_bgr.shape}")
        h, w = color_bgr.shape[:2]
        # retina_masks keeps the masks at the input resolution instead of the letterboxed proto size
        res = self.model.predict(color_bgr, device=self.device, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, retina_masks=True, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            return []
        data = res.masks.data.cpu().numpy() > 0.5
        names = getattr(self.model, "names", None) or {}
        boxes = res.boxes
        scores = boxes.conf.cpu().numpy() if boxes is not None else np.ones(len(data))
        clsid = boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros(len(data), int)

        max_area = self.max_area_frac * h * w
        out: list[Mask] = []
        for i, m in enumerate(data):
            if m.shape != (h, w):               # a model that ignored retina_masks
                import cv2
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(m.sum())
            if area < self.min_area_px or area > max_area:
                continue
            ys, xs = np.nonzero(m)
            out.append(Mask(mask=m, label=str(names.get(int(clsid[i]), "object")),
                            score=float(scores[i]),
                            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
        out.sort(key=lambda k: -k.score)
        return out

    def label_image(self, color_bgr: np.ndarray, masks: list[Mask] | None = None) -> tuple[np.ndarray, list[Mask]]:
        """(HxW int32 label image, masks). 0 = no mask; otherwise 1-based index into `masks`.

        Painted largest first, so where masks overlap the *smallest* one wins the pixel. See
        `paint_order` in the module docstring for why that beats ordering by score.
        """
        if masks is None:
            masks = self.run(color_bgr)
        h, w = color_bgr.shape[:2]
        lbl = np.zeros((h, w), np.int32)
        for i in paint_order(masks):
            lbl[masks[i].mask] = i + 1
        return lbl, masks


def paint_order(masks) -> list[int]:
    """Indices of `masks` in the order they must be painted into a label image: largest area
    first, so the smallest mask covering a pixel wins it.

    FastSAM emits nested masks — a parent region plus the things inside it. On the real wrist frame
    a 58936 px mask contained the mouse mat (97% of it) and a gripper finger (96%) and outscored the
    mat by 0.01, so ordering by score merged mat and finger into one label: precisely the blob merge
    this module exists to undo. Smallest-wins is the fix, and it needs no area threshold. It is also
    less destructive overall: on that frame it erased 3 of 39 masks rather than 6, and the tight,
    high-scoring object masks (box 0.94, cup 0.90) keep 100% of their pixels under either rule, so
    nothing is lost by preferring the more specific mask.
    """
    return sorted(range(len(masks)), key=lambda j: -int(masks[j].mask.sum()))
