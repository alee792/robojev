"""Tables from the live slice recordings (experiments/results/live/slice_*.jsonl, written by
tests/test_slice.py with ROBOJEV_DECIDER / ROBOJEV_PLANNER set; see tests/live_models.py).

    uv run --frozen python experiments/live_slice_report.py
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

LIVE = Path(__file__).resolve().parent / "results/live"
GROUPS = ("right_now", "in_plan_fix", "route")
TIMES = ("text_to_hold_s", "hand_to_pause_s", "leave_to_resume_s", "wait_to_pause_s", "go_to_resume_s", "stop_to_parked_s")


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def f(x, nd=0):
    return "-" if x is None else f"{x:.{nd}f}"


def load():
    eps = []
    for p in sorted(LIVE.glob("slice_*.jsonl*")):
        ep = {"file": p.name, "decisions": [], "llm": [], "requests": []}
        for line in (gzip.open(p, "rt") if p.suffix == ".gz" else p.open()):
            r = json.loads(line)
            if r.get("type") != "note":
                continue
            k = r.get("kind")
            if k == "episode":
                ep.update(r)
            elif k == "decision":
                ep["decisions"].append(r["rec"])
            elif k == "llm":
                ep["llm"].append(r["rec"])
            elif k == "decision_request":
                ep["requests"].append(r["rec"])
        if "scenario" in ep:
            eps.append(ep)
    return eps


def main():
    eps = load()
    by = defaultdict(list)
    for e in eps:
        by[(e["decider"], e["planner"], e["scenario"])].append(e)
    print("## Per scenario (each run)")
    print("| decider | scenario | runs | outcome | wall s | " + " | ".join(TIMES) + " | first plan ms | replan ms | user-text right_now |")
    print("|---|---|---|---|---|" + "---|" * len(TIMES) + "---|---|---|")
    for (d, pl, sc), es in sorted(by.items()):
        m = [e["measured"] for e in es]
        cells = []
        for t in TIMES:
            v = [x.get(t) for x in m]
            cells.append(", ".join(f(x, 3) if x is not None else ("none" if t in x_keys(m) else "-") for x in v) if any(t in x for x in m) else "-")
        first = [round(sum(a["latency_ms"] for a in p["attempts"])) for e in es for p in e["llm"] if p["kind"] == "plan"]
        re = [round(sum(a["latency_ms"] for a in p["attempts"])) for e in es for p in e["llm"] if p["kind"] != "plan"]
        said = [x.get("said_right_now") or ",".join(t[1] for t in x.get("user_text_decisions", [])) for x in m]
        print(f"| {d} | {sc} | {len(es)} | {', '.join(e['outcome'] for e in es)} | {', '.join(f(x['wall_s'], 1) for x in m)} | "
              + " | ".join(cells) + f" | {', '.join(map(str, first))} | {', '.join(map(str, re)) or '-'} | {', '.join(s for s in said if s) or '-'} |")

    print("\n## Decision latency in the loop (ms, every decision the brain acted on)")
    print("| decider | n | p50 | p95 | max | shadow backend | shadow n | shadow p50 | shadow p95 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in sorted({e["decider"] for e in eps}):
        rq = [r for e in eps if e["decider"] == d for r in e["requests"]]
        lat = [r["primary"]["latency_ms"] for r in rq if r["primary"].get("answers") is not None]
        sh = [r["shadow"]["latency_ms"] for r in rq if (r.get("shadow") or {}).get("answers") is not None]
        other = {"jev": "decisions", "decisions": "jev"}.get(d, "-")
        print(f"| {d} | {len(lat)} | {f(pct(lat, 50))} | {f(pct(lat, 95))} | {f(max(lat) if lat else None)} | {other} | {len(sh)} | "
              f"{f(pct(sh, 50))} | {f(pct(sh, 95))} |")
        errs = [r for r in rq if r["primary"].get("answers") is None]
        if errs:
            print(f"|  errors | {len(errs)} | {errs[0]['primary'].get('meta')} |")

    print("\n## The same slice events, both backends (acting + shadow), against the oracle where it judged")
    print("| group | n paired | agree | Jev right | Decisions right | n with truth |")
    print("|---|---|---|---|---|---|")
    for g in GROUPS:
        n = agree = jr = dr = nt = 0
        for e in eps:
            for r in e["requests"]:
                sh = (r.get("shadow") or {}).get("answers")
                pa = r["primary"].get("answers")
                if not sh or not pa or g not in sh or g not in pa:
                    continue
                jev, dec = (pa, sh) if e["decider"] == "jev" else (sh, pa)
                n += 1
                agree += jev[g]["choice"] == dec[g]["choice"]
                T = r.get("truth") or {}
                if g in T and isinstance(T[g], list):
                    nt += 1
                    jr += jev[g]["choice"] in T[g]
                    dr += dec[g]["choice"] in T[g]
        if n:
            print(f"| {g} | {n} | {100 * agree / n:.0f}% | {f(100 * jr / nt if nt else None)}% | {f(100 * dr / nt if nt else None)}% | {nt} |")

    print("\n## Disagreements on user text and hands (slice)")
    for e in eps:
        for r in e["requests"]:
            if r["event"] not in ("user_text", "scene_change"):
                continue
            sh, pa = (r.get("shadow") or {}).get("answers"), r["primary"].get("answers")
            if not sh or not pa:
                continue
            jev, dec = (pa, sh) if e["decider"] == "jev" else (sh, pa)
            row = {g: (jev.get(g, {}).get("choice"), dec.get(g, {}).get("choice")) for g in GROUPS if g in jev or g in dec}
            if any(a != b for a, b in row.values()):
                T = r.get("truth") or {}
                print(f"- {e['decider']:9s} {e['scenario']:8s} {r['event_text'][:60]!r}: " +
                      "; ".join(f"{g} jev {a} / dec {b} (truth {T.get(g)})" for g, (a, b) in row.items()))


def x_keys(m):
    return set().union(*[set(x) for x in m])


if __name__ == "__main__":
    sys.exit(main())
