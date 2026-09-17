"""Brain-vs-guard measurements for a run: how often picks change tick to tick, how fast the loop
reacts to an intrusion, whether the task resumes, plus the usual latency/outcome summary.

    .venv/bin/python scripts/analyze.py runs/<name>
"""
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
def rows(name):
    p = run / f"{name}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []

answers, ticks, cmds, events = rows("answers"), rows("ticks"), rows("commands"), rows("events")
applied = [a for a in answers if a.get("outcome") == "applied"]
print(f"run {run.name}: {len(ticks)} ticks, {len(answers)} requests, outcomes {dict(Counter(a['outcome'] for a in answers))}")
lat = sorted(a["latency_ms"] for a in answers if a.get("latency_ms") is not None)
if lat:
    q = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]
    print(f"latency p50 {q(.5):.0f} p95 {q(.95):.0f} max {lat[-1]:.0f} ms; tokens/call {sum(a.get('input_tokens') or 0 for a in applied)/max(1,len(applied)):.0f}")

def pick(a):
    t = a.get("type")
    return a.get("choice") if t == "choice" else ("yes" if a.get("noul", 0) >= 0.5 else "no") if t == "noul" else round(a.get("score", 0))
changes = defaultdict(int); total = defaultdict(int); prev = {}
for a in applied:
    for k, v in a["answers"].items():
        p = pick(v)
        if k in prev:
            total[k] += 1; changes[k] += (p != prev[k])
        prev[k] = p
print("pick changed vs previous applied answer (fraction):")
for k in sorted(total):
    print(f"  {k:18s} {changes[k]/max(1,total[k]):.2f}  ({changes[k]}/{total[k]})")

# intrusion -> first evade/avoid command; uses sim truth if present (hand inside the workspace box)
def in_box(xy): return 0.15 < xy[0] < 0.45 and -0.25 < xy[1] < 0.25
def threatening(t):
    """The hand is inside the workspace AND within 12 cm of the gripper (the default order's zone)."""
    h = (t.get("obs", {}).get("truth") or {}).get("hand"); ee = t.get("obs", {}).get("ee")
    return bool(h and ee and in_box(h) and math.hypot(h[0] - ee[0], h[1] - ee[1]) < 0.12)
t_intr = next((t["t"] for t in ticks if threatening(t)), None)
if t_intr is None:
    t_intr = next((t["t"] for t in ticks if any("appeared" in (o.get("status") or "") for o in t.get("state", {}).get("objects", []))), None)
if t_intr:
    react = next((c["t"] for c in cmds if c["t"] >= t_intr and (c.get("reason", "").startswith(("avoid", "evade")) or c.get("prim") in ("back_off", "rise_away"))), None)
    print(f"intrusion at +{t_intr - ticks[0]['t']:.1f} s; first evasive command {'+%.2f s' % (react - t_intr) if react else 'never'}")
    after = [c for c in cmds if react and c["t"] > react + 8]
    resumed = next((c["t"] for c in after if c.get("prim") in ("move_above", "descend_to_grasp", "lift", "move_to_place", "lower_to_place") and c.get("prim_status") == "running"), None)
    print(f"task resumed after evasion: {'+%.1f s after intrusion' % (resumed - t_intr) if resumed else 'no'}")
seq = []
for c in cmds:
    key = (c.get("prim"), c.get("prim_status"))
    if not seq or seq[-1][0] != key:
        seq.append([key, 1])
    else:
        seq[-1][1] += 1
print("primitive sequence:", " -> ".join(f"{k[0]}({k[1]})" for k, n in seq if k[0] and n > 1))
for e in events:
    print(f"  event +{e['t'] - ticks[0]['t']:.1f}s {e.get('kind')} {e.get('text') or e.get('orders') or e.get('reason') or ''}"[:120])
