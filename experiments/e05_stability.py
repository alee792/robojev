"""E5: determinism under identical input, and answer flipping under small sensor jitter.
Jitter models a pose estimate wobbling by ±sigma each tick. Chatter = flips between consecutive ticks.
Cost: ~ (30 + 3*40) calls, tiny states ≈ negligible.
"""

import random
from collections import Counter

from common import Log, ask, make_client, render

log = Log("e05_stability")
rng = random.Random(5)

# Two objects nearly equidistant from the gripper: a deliberately borderline case.
BASE = {
    "gripper": (0.30, 0.00, 0.12),
    "objects": [
        {"id": "red cup A", "xyz": (0.42, 0.06, 0.02)},
        {"id": "blue block B", "xyz": (0.42, -0.065, 0.02)},
        {"id": "green ball C", "xyz": (0.25, 0.30, 0.02)},
    ],
}
QS = {
    "closest": {"type": "choice", "instructions": "Which object in `objects` is closest to the gripper?", "criteria": {"red cup A": None, "blue block B": None, "green ball C": None}},
    "approach": {"type": "choice", "instructions": "The task is to pick up the closest object. How should the gripper move right now?", "criteria": {"move_toward_red_cup_A": None, "move_toward_blue_block_B": None, "hold_still": None}},
    "in_reach": {"type": "choice", "instructions": "Is any object close enough to grasp without moving the arm? (under 3 cm)", "criteria": {"yes": None, "no": None}},
}


def jittered(sigma):
    s = {"gripper": BASE["gripper"], "objects": []}
    for o in BASE["objects"]:
        x, y, z = o["xyz"]
        s["objects"].append({"id": o["id"], "xyz": (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z)})
    return s


with make_client() as c:
    for fmt in ("numbers", "bands", "both"):
        # identical input
        state = render(BASE, fmt)
        seen = {k: Counter() for k in QS}
        probs = {k: set() for k in QS}
        for _ in range(10):
            r = ask(c, state, QS)
            log.write(exp="identical", fmt=fmt, body=r.body)
            for k in QS:
                a = r.body["answers"][k]
                seen[k][a["choice"]] += 1
                probs[k].add(tuple(round(v, 6) for v in a["probabilities"].values()))
        print(f"[{fmt}] identical x10: " + "; ".join(f"{k}: {dict(seen[k])} distinct_prob_vectors={len(probs[k])}" for k in QS))

        for sigma in (0.002, 0.005, 0.01):
            prev, flips, n = {}, Counter(), 0
            for _ in range(40):
                r = ask(c, render(jittered(sigma), fmt), QS)
                log.write(exp="jitter", fmt=fmt, sigma=sigma, body=r.body)
                if r.status != 200:
                    continue
                n += 1
                for k in QS:
                    ch = r.body["answers"][k]["choice"]
                    if k in prev and prev[k] != ch:
                        flips[k] += 1
                    prev[k] = ch
            print(f"[{fmt}] jitter σ={sigma*1000:.0f}mm, {n} ticks, flips between consecutive ticks: {dict(flips)}")
