"""V1 preview: one Decisions API call with a rendered camera frame attached, per scene-change event.

The MuJoCo WidowX scene has no person (the slice's hand is a perception override, not a body), so a
frame would never show one. Here a hand is DRAWN into the rendered frame only (a skin-coloured
forearm capsule and palm box added to the render scene, no physics) at a known distance from the
gripper, mid-carry. Per frame, one request: the slice's decision state with the hand facts removed
("judge the hand from the image"), a predicate "is a hand near the gripper" and the right-now choice.
The same request text-only (hand facts in the state, no image) is the baseline.

    uv run --frozen python experiments/v1_frame_preview.py [--reps 3]

Writes experiments/results/live/v1_<stamp>.jsonl and a few PNGs next to it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from robojev.brain.backends.decisions_api import DecisionsAPI
from robojev.brain.core.decision import RIGHT_NOW_CRITERIA
from robojev.robots.widowx_sim import scene
from robojev.robots.widowx_sim.server import ARM_ID, SimRobotServer

OUT = Path(__file__).resolve().parent / "results/live"
NEAR_M = 0.20
SKIN = np.array([0.80, 0.55, 0.45, 1.0], dtype=np.float32)


def mid_carry_server() -> SimRobotServer:
    blocks = [scene.Block("block_5", (0.26, 0.12)), scene.Block("block_1", (0.26, 0.0)), scene.Block("block_3", (0.26, -0.12))]
    s = SimRobotServer(scene.SceneSpec(blocks=blocks, tray=scene.Tray(n=3)), realtime=False).open()
    sk = s.start(ARM_ID, "pick_and_place", {"object": "block_1", "place": "slot_0"})
    assert s.run_until(lambda: s.status(sk).phase == "carry", 30.0)
    s.run_for(0.3)
    return s


def add_hand(scn, at: np.ndarray, toward: np.ndarray) -> None:
    """A forearm (capsule) reaching from outside the table toward `at`, ending in a palm (box)."""
    d = toward / np.linalg.norm(toward)
    def geom(kind, size, pos, mat):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(), SKIN)
        scn.ngeom += 1
    z = np.array([0, 0, 1.0])
    x = d
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    geom(mujoco.mjtGeom.mjGEOM_BOX, [0.045, 0.04, 0.012], at, R)                           # palm, 9 x 8 cm
    arm_c = at - d * 0.20                                                                  # forearm centre, 20 cm back
    Rc = np.stack([y, np.cross(d, y), d], axis=1)                                          # capsule axis along d
    geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [0.03, 0.15, 0], arm_c, Rc)


def render(s: SimRobotServer, hand=None, w=512, h=384) -> bytes:
    import cv2
    with mujoco.Renderer(s.model, h, w) as r:
        r.update_scene(s.data, camera="overhead")
        if hand is not None:
            add_hand(r.scene, *hand)
        img = r.render()
    ok, png = cv2.imencode(".png", img[:, :, ::-1])
    return png.tobytes()


def state(hand_text) -> dict:
    return {"event": {"kind": "scene change", "what_happened": "something changed near the arm"},
            "task": "Line the blocks up in the tray, lowest number on the left.",
            "robot": {"mode": "running the plan", "holding": "block 1", "new_plan_being_worked_out": "no"},
            "plan_position": {"current_step": "pick_and_place(block 1 -> slot 0 (the leftmost slot))",
                              "arm_in_step": "carrying block 1 at travel height toward slot 0"},
            "hand": hand_text,
            "camera": "The attached image is the overhead camera, looking straight down at the table; the robot's base is at the bottom of the image and the far side of the table, where a person stands, is at the top."
                      if hand_text == "judge from the camera image" else "no image"}


QUESTIONS = {
    "hand_near": {"type": "noul", "instructions": f"Is a person's hand within about {int(NEAR_M * 100)} cm of the robot's gripper?",
                  "criteria": {"true": "A person's hand is close to the gripper.", "false": "No hand, or a hand well away from the gripper."}},
    "right_now": {"type": "choice", "instructions": {"question": "Given `event`, what should the arm do right now?"},
                  "criteria": {k: RIGHT_NOW_CRITERIA[k] for k in ("carry_on", "hold", "pause", "back_off")}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    s = mid_carry_server()
    ee = np.array(s.arms[ARM_ID].ee())
    print(f"mid-carry: gripper at {np.round(ee, 3)}")
    be = DecisionsAPI(max_requests=200)
    be.warm()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    conds = [("none", None), (0.08, 0.08), (0.15, 0.15), (0.40, 0.40)]
    for rep in range(a.reps):
        for name, dist in conds:
            hand = None
            if dist is not None:
                ang = math.radians([-50, 0, 50][rep % 3])                               # from the person's side (+x), three bearings
                d = np.array([-math.cos(ang), -math.sin(ang), 0.0])                    # the direction the hand reaches in
                at = ee + np.array([dist * math.cos(ang), dist * math.sin(ang), 0.0])
                at[2] = max(at[2], scene.TABLE_Z + 0.06)
                hand = (at, d)
            png = render(s, hand)
            if rep == 0:
                (OUT / f"v1_{stamp}_{name}.png").write_bytes(png)
            truth = dist is not None and dist <= NEAR_M
            band = {None: "no person's hand in view", 0.08: {"distance_from_gripper": "very close"},
                    0.15: {"distance_from_gripper": "near", "coming_closer": "yes"}, 0.40: {"distance_from_gripper": "mid-range", "coming_closer": "no"}}[dist]
            img = be.answer(state("judge from the camera image"), QUESTIONS, images=[png])
            txt = be.answer(state(band), QUESTIONS)
            for mode, r in (("image", img), ("text", txt)):
                ans = r["answers"] or {}
                p = (ans.get("hand_near") or {}).get("noul")
                rows.append({"rep": rep, "hand_m": dist, "mode": mode, "truth_near": truth, "p_near": p,
                             "right_now": (ans.get("right_now") or {}).get("choice"), "latency_ms": r["latency_ms"],
                             "upstream_ms": r["meta"].get("upstream_ms"), "in_tok": r["in_tok"], "png_bytes": len(png) if mode == "image" else 0})
                print(f"rep {rep} hand {name!s:5} {mode:5}: p_near {p}  right_now {rows[-1]['right_now']}  {r['latency_ms']:.0f} ms  {r['in_tok']} tok")
    with (OUT / f"v1_{stamp}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    s.close()
    for mode in ("image", "text"):
        rs = [r for r in rows if r["mode"] == mode and r["p_near"] is not None]
        acc = sum((r["p_near"] >= 0.5) == r["truth_near"] for r in rs)
        lat = sorted(r["latency_ms"] for r in rs)
        print(f"{mode}: hand-near right {acc}/{len(rs)}; latency p50 {lat[len(lat) // 2]:.0f} max {lat[-1]:.0f} ms; "
              f"tokens {np.median([r['in_tok'] for r in rs]):.0f}")


if __name__ == "__main__":
    sys.exit(main())
