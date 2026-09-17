"""E6 + E7: spatial judgments with numbers vs bands vs both, scored against geometry computed in code.
Also a Doom-style 'which X' Choice over entity labels (E10) at several scene sizes.
Cost: ~ 3 formats x 3 sizes x 25 scenes = 225 calls, small states ≈ $0.01.
"""

import random
from collections import defaultdict

from common import Log, ask, make_client, random_scene, rel, render

log = Log("e06_spatial")
rng = random.Random(6)
acc = defaultdict(lambda: [0, 0])  # (fmt, n, q) -> [correct, total]
conf_ok, conf_bad = defaultdict(list), defaultdict(list)

with make_client() as c:
    for n in (3, 10, 40):
        for trial in range(25):
            scene = random_scene(n, rng)
            g = scene["gripper"]
            rels = {o["id"]: rel(g, o["xyz"]) for o in scene["objects"]}
            labels = list(rels)
            truth = {
                "closest": min(labels, key=lambda l: rels[l]["dist"]),
                "leftmost": max(labels, key=lambda l: rels[l]["bearing"]),
                "first_is_left": "left" if rels[labels[0]]["bearing"] > 0 else "right",
                "first_within_25cm": "yes" if rels[labels[0]]["dist"] < 0.25 else "no",
            }
            # descriptive target: pick an object and describe it by color+kind (unique if possible)
            target = rng.choice(scene["objects"])
            desc = f"the {target['color']} {target['kind']}"
            matches = [o["id"] for o in scene["objects"] if o["color"] == target["color"] and o["kind"] == target["kind"]]
            for fmt in ("numbers", "bands", "both"):
                state = render(scene, fmt)
                qs = {
                    "closest": {"type": "choice", "instructions": "Which object in `objects` is closest to the gripper?", "criteria": {l: None for l in labels}},
                    "leftmost": {"type": "choice", "instructions": "Which object in `objects` has the most leftward bearing from the gripper?", "criteria": {l: None for l in labels}},
                    "first_is_left": {"type": "choice", "instructions": f"Is `{labels[0]}` to the left or to the right of the gripper?", "criteria": {"left": None, "right": None}},
                    "first_within_25cm": {"type": "choice", "instructions": f"Is `{labels[0]}` less than 25 cm from the gripper?", "criteria": {"yes": None, "no": None}},
                    "named": {"type": "choice", "instructions": f"The user said: \"pick up {desc}\". Which object in `objects` did they mean?", "criteria": {l: None for l in labels}},
                }
                r = ask(c, state, qs)
                log.write(n=n, fmt=fmt, truth=truth, desc=desc, matches=matches, status=r.status, body=r.body)
                if r.status != 200:
                    print("ERR", r.body)
                    continue
                ans = r.body["answers"]
                for q, t in list(truth.items()) + [("named", None)]:
                    a = ans[q]
                    ok = (a["choice"] in matches) if q == "named" else (a["choice"] == t)
                    if q == "named" and len(matches) > 1:
                        pass
                    acc[(fmt, n, q)][0] += ok
                    acc[(fmt, n, q)][1] += 1
                    (conf_ok if ok else conf_bad)[(fmt, q)].append(a["confidence"])

print("accuracy (correct/total) by format, scene size, question")
for q in ("closest", "leftmost", "first_is_left", "first_within_25cm", "named"):
    print(f"\n{q}")
    for n in (3, 10, 40):
        print(f"  n={n:>2}  " + "  ".join(f"{fmt}={acc[(fmt,n,q)][0]}/{acc[(fmt,n,q)][1]}" for fmt in ("numbers", "bands", "both")))
print("\nmean confidence when right vs wrong")
for (fmt, q), v in sorted(conf_ok.items()):
    bad = conf_bad.get((fmt, q), [])
    print(f"  {fmt:<8} {q:<18} right={sum(v)/len(v):.2f} (n={len(v)})  wrong={(sum(bad)/len(bad)) if bad else float('nan'):.2f} (n={len(bad)})")
