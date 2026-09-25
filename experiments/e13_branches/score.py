"""E13 scoring: ground truth relative to the plan (via the oracle), per-correction outcomes, tables and
the offline gate sweep. Everything after `truth()` works from logged records alone, so a live log can
be re-scored at any gate without new calls (`--replay`)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from common import pct

from . import oracle
from . import plan as P
from .router import decide
from .world import CATEGORIES, FAMILIES

SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def truth(task, correction, plan: dict, current: dict) -> dict:
    """Which route is right *for this plan*, and which prepared branches give a correct arrangement.

    route: chatter -> continue; pause -> pause; otherwise adjust if a prepared branch is correct
    (continue is also accepted when the current branch already is), else new_plan_needed."""
    tgt = oracle.target(task, correction)
    ok = {k: tgt.check(g) for k, g in P.branch_goals(plan).items()}
    cur = P.combo_key(plan, current)
    cat = correction.category
    if cat == "chatter":
        route, tk = ["continue"], cur
    elif cat == "pause":
        route, tk = ["pause"], cur
    elif ok.get(cur):
        route, tk = ["adjust", "continue"], cur
    elif any(ok.values()):
        route, tk = ["adjust"], next(k for k, v in ok.items() if v)
    else:
        route, tk = ["new_plan_needed"], None
    return {"route": route, "branch_ok": ok, "target_key": tk}


def outcome(rec: dict, gate: float) -> dict:
    """System outcome for one logged correction at a gate: path, correct (None = escalation not called), latency."""
    d = decide(rec["plan"], rec["current"], rec["jev"]["answers"], gate)
    jl = rec["jev"]["latency_ms"]
    if d.action in ("keep", "branch"):
        return {"d": d, "path": "jev", "correct": bool(rec["truth"]["branch_ok"].get(d.key)), "latency": jl}
    e = rec.get("esc")
    if not e:
        return {"d": d, "path": "escalated", "correct": None, "latency": None}
    return {"d": d, "path": "escalated", "correct": bool(e["valid"] and e["correct"]), "latency": jl + e["latency_ms"], "valid": e["valid"]}


def _f(ok, n):
    return f"{ok}/{n} ({100 * ok / n:.0f}%)" if n else "-"


def _lat(xs):
    xs = [x for x in xs if x is not None]
    return f"{pct(xs, 50):.0f}/{pct(xs, 95):.0f}" if xs else "-"


def print_planner(plans: list[dict], title: str):
    print(f"\n==================== {title}: planner (step 1) ====================")
    hdr = f"{'family':<14} {'tasks':>5} {'valid 1st':>10} {'valid +retry':>12} {'default ok':>11} {'coverable w/ branch':>20} {'params':>6} {'branches':>8}"
    print(hdr)
    print("-" * len(hdr))
    for fam in list(FAMILIES) + ["all"]:
        rs = [r for r in plans if fam == "all" or r["family"] == fam]
        if not rs:
            continue
        v = [r for r in rs if r["valid"]]
        cov = [c for r in v for c in r["coverage"]]
        print(f"{fam:<14} {len(rs):>5} {sum(r['valid_first'] for r in rs):>10} {len(v):>12} "
              f"{sum(r['default_ok'] for r in v):>11} {_f(sum(c['covered'] for c in cov), len(cov)):>20} "
              f"{(sum(r['n_params'] for r in v) / len(v)) if v else 0:>6.1f} {(sum(r['n_branches'] for r in v) / len(v)) if v else 0:>8.1f}")
    lat = [r["latency_ms"] for r in plans]
    print(f"latency ms p50/p95: {_lat(lat)}; tokens in/out per plan (mean): "
          f"{sum(r['in_tok'] for r in plans) / max(1, len(plans)):.0f}/{sum(r['out_tok'] for r in plans) / max(1, len(plans)):.0f}")
    errs = Counter(re.sub(r"^branch \d+: ", "", re.sub(r"'[^']*'|\[[^\]]*\]|\S+_\S+", "…", e)).split(" (")[0]
                   for r in plans for a in r["attempts"] for e in a["errors"])
    if errs:
        print("validation errors (all attempts): " + ", ".join(f"{k} x{v}" for k, v in errs.most_common(6)))


def print_jev(recs: list[dict], gate: float, title: str):
    print(f"\n==================== {title}: Jev router (step 2) ====================")
    print("route accuracy vs truth for this plan (continue / adjust / new_plan_needed / pause), by correction category")
    for cat in list(CATEGORIES) + ["all"]:
        rs = [r for r in recs if (cat == "all" or r["category"] == cat) and r["jev"]["answers"]]
        if not rs:
            continue
        ok = sum(r["jev"]["answers"]["route"]["choice"] in r["truth"]["route"] for r in rs)
        got = Counter(r["jev"]["answers"]["route"]["choice"] for r in rs)
        print(f"  {cat:<12} {_f(ok, len(rs)):>14}   answered: " + ", ".join(f"{k} {v}" for k, v in got.most_common()))
    # branch pick: cases where a prepared branch other than the current one is the right answer
    rs = [r for r in recs if r["truth"]["route"] == ["adjust"] and r["jev"]["answers"]]
    picks = Counter()
    for r in rs:
        d = decide(r["plan"], r["current"], r["jev"]["answers"], 0.0)
        picks["right branch" if d.action == "branch" and r["truth"]["branch_ok"].get(d.key) else
              "wrong branch" if d.action in ("branch", "keep") else f"escalated ({d.reason})"] += 1
    print(f"branch pick, ungated, where a prepared branch is the answer (n={len(rs)}): " + ", ".join(f"{k} {v}" for k, v in picks.most_common()))
    reasons = Counter(outcome(r, gate)["d"].reason for r in recs)
    print(f"decisions at gate {gate}: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))

    print("\ngate sweep (offline, same answers): Jev keeps (its accuracy) / escalates / its mistakes caught / end-to-end accuracy / latency p50/p95 ms")
    mistakes = [r for r in recs if (o := outcome(r, 0.0))["path"] == "jev" and not o["correct"]]
    for g in SWEEP:
        outs = [outcome(r, g) for r in recs]
        kept = [o for o in outs if o["path"] == "jev"]
        esc = [o for o in outs if o["path"] == "escalated"]
        caught = sum(outcome(r, g)["path"] == "escalated" for r in mistakes)
        known = [o for o in outs if o["correct"] is not None]
        e2e = f"{100 * sum(o['correct'] for o in known) / len(known):.0f}%" + ("" if len(known) == len(outs) else f" of {len(known)} known") if known else "-"
        print(f"  >={g:.1f}: keeps {100 * len(kept) / max(1, len(outs)):3.0f}% ({100 * sum(o['correct'] for o in kept) / max(1, len(kept)):3.0f}% right)"
              f"  escalates {100 * len(esc) / max(1, len(outs)):3.0f}%  caught {caught}/{len(mistakes)}  e2e {e2e:<16} latency {_lat([o['latency'] for o in outs])}")
    lat = [r["jev"]["latency_ms"] for r in recs if r["jev"]["answers"]]
    errs = sum(1 for r in recs if not r["jev"]["answers"])
    print(f"Jev latency ms p50/p95: {_lat(lat)}; errors {errs}/{len(recs)}")


def print_e2e(recs: list[dict], gate: float, title: str):
    print(f"\n==================== {title}: end to end per correction, gate {gate} ====================")
    hdr = f"{'category':<12} {'n':>4} {'system right':>14} {'jev ok':>6} {'jev bad':>7} {'esc ok':>6} {'esc bad':>7} {'p50/p95 ms':>12}   {'always-LLM right':>16} {'p50/p95 ms':>12}"
    print(hdr)
    print("-" * len(hdr))
    for cat in list(CATEGORIES) + ["all"]:
        rs = [r for r in recs if cat == "all" or r["category"] == cat]
        if not rs:
            continue
        outs = [outcome(r, gate) for r in rs]
        c = Counter((o["path"], o["correct"]) for o in outs)
        known = [o for o in outs if o["correct"] is not None]
        base = [r["esc"] for r in rs if r.get("esc") and r["esc"].get("baseline")]
        bl = f"{_f(sum(bool(e['valid'] and e['correct']) for e in base), len(base))}" if base else "-"
        print(f"{cat:<12} {len(rs):>4} {_f(sum(o['correct'] for o in known), len(known)):>14} {c[('jev', True)]:>6} {c[('jev', False)]:>7} "
              f"{c[('escalated', True)]:>6} {c[('escalated', False)]:>7} {_lat([o['latency'] for o in outs]):>12}   "
              f"{bl:>16} {_lat([e['latency_ms'] for e in base]):>12}")
    esc = [r["esc"] for r in recs if r.get("esc")]
    if esc:
        print(f"escalation calls: {len(esc)}; valid 1st try {sum(e['valid_first'] for e in esc)}, valid after retry {sum(e['valid'] for e in esc)}, "
              f"correct {sum(bool(e['valid'] and e['correct']) for e in esc)}; tokens in/out mean "
              f"{sum(e['in_tok'] for e in esc) / len(esc):.0f}/{sum(e['out_tok'] for e in esc) / len(esc):.0f}")
    dok = [r for r in recs if r["default_ok"]]
    if len(dok) != len(recs):
        outs = [outcome(r, gate) for r in dok]
        known = [o for o in outs if o["correct"] is not None]
        print(f"only tasks whose default branch was right: system {_f(sum(o['correct'] for o in known), len(known))}")


def by_family(recs: list[dict], gate: float):
    rows = defaultdict(list)
    for r in recs:
        rows[r["family"]].append(outcome(r, gate))
    print("system accuracy by task family: " + ", ".join(
        f"{f} {sum(o['correct'] for o in rows[f] if o['correct'] is not None)}/{len(rows[f])}" for f in FAMILIES if f in rows))


def print_all(plans, recs, gate, title):
    print_planner(plans, title)
    if recs:
        print_jev(recs, gate, title)
        print_e2e(recs, gate, title)
        by_family(recs, gate)
