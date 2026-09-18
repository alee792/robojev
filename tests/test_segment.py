"""Colour instance masks, and the blob split they buy the depth detector.

The model tests need ultralytics and the downloaded weights and are skipped without them; the
important test (two touching boxes come apart) uses a stub segmenter and runs anywhere.
"""
import numpy as np
import pytest

from robojev.perception.detect import Detector, _cluster_xy
from robojev.perception.geometry import Intrinsics
from robojev.perception.segment import Mask, Segmenter, SegmenterUnavailable, paint_order

INFO = {"width": 320, "height": 240, "fx": 200.0, "fy": 200.0, "ppx": 160.0, "ppy": 120.0}


def _mask(h, w, x0, x1, y0, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# --------------------------------------------------------------------------- pure helpers

def test_paint_order_is_largest_first():
    small = Mask(_mask(20, 20, 0, 4, 0, 4), "object", 0.4, (0, 0, 4, 4))     # 16 px
    big = Mask(_mask(20, 20, 0, 10, 0, 10), "object", 0.9, (0, 0, 10, 10))   # 100 px
    # big is painted first despite its higher score, so the small nested mask keeps its pixels
    assert paint_order([small, big]) == [1, 0]


def test_label_image_gives_the_nested_mask_its_pixels():
    parent = Mask(_mask(20, 20, 0, 16, 0, 16), "object", 0.61, (0, 0, 16, 16))
    child = Mask(_mask(20, 20, 2, 6, 2, 6), "object", 0.60, (2, 2, 6, 6))
    lbl = np.zeros((20, 20), np.int32)
    for i in paint_order([parent, child]):
        lbl[[parent, child][i].mask] = i + 1
    assert lbl[3, 3] == 2 and lbl[10, 10] == 1      # child wins its own pixels, parent keeps the rest


# --------------------------------------------------------- Detector + a stub segmenter (no model)

class StubSegmenter:
    """Anything with .run(bgr) -> masks works; the Detector never imports ultralytics itself."""

    def __init__(self, masks):
        self.masks, self.calls = masks, 0

    def run(self, color):
        self.calls += 1
        return self.masks


def _two_touching_boxes():
    """A synthetic top-down scene: table at z=0 plus two adjacent 4 cm blocks that touch.

    Returns (intr, color, depth_m, pose6, the pixel column where the two blocks meet).
    """
    intr = Intrinsics(INFO)
    h, w = intr.h, intr.w
    cam_z = 0.60                                  # camera 60 cm above the table, looking straight down
    depth = np.full((h, w), cam_z, np.float32)
    # each block is 5 cm tall; together they span a continuous strip, so xy clustering merges them
    depth[100:150, 120:160] = cam_z - 0.05        # block A
    depth[100:150, 160:200] = cam_z - 0.05        # block B, sharing the edge at u=160
    color = np.zeros((h, w, 3), np.uint8)
    color[:, :] = (180, 180, 180)
    color[100:150, 120:160] = (60, 60, 200)
    color[100:150, 160:200] = (60, 200, 60)
    # identity extrinsic with the optical axis pointing down: p_base = R @ p_opt + t
    R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    t = np.array([0.40, 0.0, cam_z])
    return intr, color, depth, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], R, t, 160


def _detector(intr, R, t, segmenter=None):
    return Detector(intr, workspace_xy=((0.0, 1.0), (-0.5, 0.5)), min_height=0.02, max_height=0.30,
                    stride=2, table_z=0.0, extrinsic=lambda pose6: (R, t), finger_mask=False,
                    segmenter=segmenter)


def test_depth_alone_merges_two_touching_blocks():
    intr, color, depth, pose, R, t, _ = _two_touching_boxes()
    dets, info = _detector(intr, R, t).run(color, depth, pose)
    assert len(dets) == 1, f"expected the known merge bug, got {len(dets)}"
    assert dets[0].seg_label is None


def test_masks_split_two_touching_blocks():
    intr, color, depth, pose, R, t, seam = _two_touching_boxes()
    stub = StubSegmenter([Mask(_mask(intr.h, intr.w, 120, seam, 100, 150), "cup", 0.9, (120, 100, seam, 150)),
                          Mask(_mask(intr.h, intr.w, seam, 200, 100, 150), "box", 0.8, (seam, 100, 200, 150))])
    dets, info = _detector(intr, R, t, segmenter=stub).run(color, depth, pose)
    assert len(dets) == 2, f"masks should split the blob, got {len(dets)}"
    assert sorted(d.seg_label for d in dets) == ["box", "cup"]
    assert info["n_masks"] == 2 and info["n_masked_points"] > 0
    # the two centres are genuinely apart, not one blob reported twice. The seam is a column of
    # pixels, and with this extrinsic the image u axis maps to base x, so they separate in x.
    xs = sorted(d.base_xyz[0] for d in dets)
    assert xs[1] - xs[0] > 0.05, f"centres too close: {xs}"


def test_unmasked_points_still_cluster():
    """A mask over only one block leaves the other to the existing grid clustering."""
    intr, color, depth, pose, R, t, seam = _two_touching_boxes()
    stub = StubSegmenter([Mask(_mask(intr.h, intr.w, 120, seam, 100, 150), "cup", 0.9, (120, 100, seam, 150))])
    dets, info = _detector(intr, R, t, segmenter=stub).run(color, depth, pose)
    assert len(dets) == 2
    labels = sorted((d.seg_label or "-") for d in dets)
    assert labels == ["-", "cup"]                  # one from the mask, one from _cluster_xy


def test_segmentation_result_is_cached_per_frame():
    intr, color, depth, pose, R, t, seam = _two_touching_boxes()
    stub = StubSegmenter([Mask(_mask(intr.h, intr.w, 120, seam, 100, 150), "cup", 0.9, (120, 100, seam, 150))])
    det = _detector(intr, R, t, segmenter=stub)
    det.run(color, depth, pose)
    det.run(color, depth, pose)                    # same array object: must not re-run the model
    assert stub.calls == 1
    det.run(color.copy(), depth, pose)             # a new frame must
    assert stub.calls == 2


def test_mismatched_mask_shape_falls_back_to_depth_clustering():
    intr, color, depth, pose, R, t, seam = _two_touching_boxes()
    stub = StubSegmenter([Mask(_mask(intr.h + 7, intr.w, 120, seam, 100, 150), "cup", 0.9, (120, 100, seam, 150))])
    dets, info = _detector(intr, R, t, segmenter=stub).run(color, depth, pose)
    assert len(dets) == 1 and dets[0].seg_label is None      # merged again, but nothing crashed


def test_a_throwing_segmenter_does_not_take_the_loop_down():
    class Boom:
        def run(self, color):
            raise RuntimeError("model went away")

    intr, color, depth, pose, R, t, _ = _two_touching_boxes()
    dets, info = _detector(intr, R, t, segmenter=Boom()).run(color, depth, pose)
    assert len(dets) == 1 and "model went away" in info["seg_error"]


def test_cluster_xy_unchanged_without_a_segmenter():
    xy = np.array([[0.0, 0.0], [0.01, 0.0], [0.5, 0.5], [0.51, 0.5]])
    assert len(np.unique(_cluster_xy(xy, cell=0.02))) == 2


# ------------------------------------------------------------------- the real model (needs weights)

def _real_segmenter():
    pytest.importorskip("ultralytics")
    try:
        return Segmenter()
    except SegmenterUnavailable as e:
        pytest.skip(f"segmentation weights unavailable offline: {e}")


def test_segmenter_runs_on_a_synthetic_image():
    seg = _real_segmenter()
    assert seg.device in ("mps", "cpu", "cuda")
    # two solid rectangles on a plain background: shapes a class-agnostic model will happily split
    img = np.full((480, 640, 3), 200, np.uint8)
    img[200:340, 120:260] = (40, 40, 220)
    img[200:340, 380:520] = (40, 200, 40)
    masks = seg.run(img)
    assert masks, "expected at least one mask on a two-rectangle image"
    for m in masks:
        assert m.mask.dtype == bool and m.mask.shape == (480, 640)
        assert 0.0 <= m.score <= 1.0 and isinstance(m.label, str)
        x1, y1, x2, y2 = m.bbox
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
        assert m.mask[y1:y2, x1:x2].sum() == m.mask.sum()   # the bbox really bounds the mask
    assert [m.score for m in masks] == sorted((m.score for m in masks), reverse=True)
    # each rectangle should be covered, and by different masks
    lbl, _ = seg.label_image(img, masks)
    assert lbl[270, 190] != 0 and lbl[270, 450] != 0
    assert lbl[270, 190] != lbl[270, 450]


def test_segmenter_rejects_a_bad_frame():
    seg = _real_segmenter()
    with pytest.raises(ValueError):
        seg.run(np.zeros((480, 640), np.uint8))


def test_bad_weights_name_raises_at_construction():
    pytest.importorskip("ultralytics")
    with pytest.raises(SegmenterUnavailable):
        Segmenter(model="definitely-not-a-model-xyz.pt")
