"""E4: recover how `confidence` relates to `probabilities` (Choice + Score), across option counts.
Fits candidate formulas and reports max abs error for each.
Cost: ~40 calls x small state ≈ negligible.
"""

import math
import random

import numpy as np

from common import Log, ask, make_client, random_scene, render

log = Log("e04_confidence")
rng = random.Random(4)
rows = []  # (kind, probs list, confidence)


def candidates(p):
    p = np.array(sorted(p, reverse=True), dtype=float)
    n = len(p)
    nz = p[p > 0]
    H = -(nz * np.log(nz)).sum()
    out = {
        "max_prob": p[0],
        "margin_top2": p[0] - (p[1] if n > 1 else 0),
        "1-H/logn": 1 - H / math.log(n) if n > 1 else 1.0,
        "(pmax-1/n)/(1-1/n)": (p[0] - 1 / n) / (1 - 1 / n) if n > 1 else 1.0,
        "1-gini/(1-1/n)": 1 - (1 - (p**2).sum()) / (1 - 1 / n) if n > 1 else 1.0,
        "sum_p2": (p**2).sum(),
    }
    return out


with make_client() as c:
    for n_opts in (2, 3, 5, 8, 20):
        for trial in range(8):
            scene = random_scene(max(n_opts, 3), rng)
            state = render(scene, "both")
            labels = [o["id"] for o in scene["objects"]][:n_opts]
            qs = {
                "which": {"type": "choice", "instructions": rng.choice([
                    "Which object is closest to the gripper?",
                    "Which object would a person most likely reach for to drink from?",
                    "Which object is furthest to the left of the gripper?",
                    "Which object looks most fragile?",
                ]), "criteria": {l: None for l in labels}},
                "score": {"type": "score", "instructions": "How cluttered is the table?", "criteria": ["Empty or nearly empty", "A few objects", "Moderately cluttered", "Very cluttered"][: max(2, min(4, n_opts))]},
            }
            r = ask(c, state, qs)
            log.write(n_opts=n_opts, status=r.status, body=r.body)
            if r.status != 200:
                print("ERR", r.body)
                continue
            for k in ("which", "score"):
                a = r.body["answers"][k]
                rows.append((a["type"], len(a["probabilities"]), list(a["probabilities"].values()), a["confidence"]))

print(f"{len(rows)} answers")
names = list(candidates([0.5, 0.5]).keys())
for kind in ("choice", "score"):
    sub = [r for r in rows if r[0] == kind]
    if not sub:
        continue
    print(f"\n{kind}: max |candidate - confidence| over {len(sub)} answers")
    for nm in names:
        errs = [abs(candidates(p)[nm] - conf) for _, _, p, conf in sub]
        print(f"  {nm:<22} max={max(errs):.4f} mean={sum(errs)/len(errs):.4f}")
    print("  samples (n, sorted probs, conf):")
    for _, n, p, conf in sub[:6]:
        print("   ", n, [round(x, 3) for x in sorted(p, reverse=True)[:4]], round(conf, 4))
