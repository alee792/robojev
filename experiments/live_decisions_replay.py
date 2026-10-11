"""H3 / H4: the same decision requests to Jev and to OpenAI's Decisions API, judged against e12v2's oracle.

e12v2's live runs logged every decision request (state + questions, built by core/decision.py) with the
oracle's truth for that moment. This replays a stratified sample of them, unchanged, to both live
backends (robojev.brain.backends), so the two answer identical inputs, and also plants bad plans (a
plan_arrived request with one numbered block's step removed and the plan-check question for it added)
to see whether the plan check still catches them.

    uv run --frozen python experiments/live_decisions_replay.py --n 500 --plants 60

Writes experiments/results/live/replay_<stamp>.jsonl (one line per request: both answers, truth,
latencies) and prints the tables docs/live-results.md quotes.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from e12v2.eval.metrics import gate_sweep  # noqa: E402

from robojev.brain.backends.decisions_api import DecisionsAPI  # noqa: E402
from robojev.brain.backends.jev import JevHTTP  # noqa: E402
from robojev.brain.core.combine import Gates, combine, noul_conf  # noqa: E402
from robojev.brain.core.data import Event  # noqa: E402

LOGS = ["e12v2_20260927-143315.jsonl.gz", "e12v2_20260927-161449.jsonl.gz", "e12v2_20260925-093744.jsonl.gz",
        "e12v2_20260925-101602.jsonl.gz"]
GROUPS = ("right_now", "in_plan_fix", "route")
GATES = (0.5, 0.6, 0.7, 0.8, 0.9)


def load() -> list[dict]:
    seen, out = set(), []
    for name in LOGS:
        with gzip.open(ROOT / "results" / name, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") != "decision" or not str(r.get("arm", "")).startswith("jev") or not r.get("request") or not r.get("answers"):
                    continue
                # the full arm for everything; its ablations only for their plan checks (asked the same way, H4)
                if r["arm"] != "jev" and not any(k.startswith("check.") for k in r["request"]["questions"]):
                    continue
                key = json.dumps(r["request"], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"src": f"{name}:{r['scenario']}:{r['seed']}:{r['t']}", "scenario": r["scenario"], "event": r["event"]["kind"],
                            "request": r["request"], "truth": r["truth"], "jev_then": r["answers"]})
    return out


def sample(recs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Rare event kinds (up to 2/3 of n), every plan_arrived with a plan check, routine ones to fill up to n."""
    by = defaultdict(list)
    for r in recs:
        by[r["event"]].append(r)
    rare = [r for k, v in by.items() if k not in ("step_done", "plan_arrived") for r in v]
    rng.shuffle(rare)
    out = rare[: n * 2 // 3]
    checks = [r for r in by["plan_arrived"] if any(k.startswith("check.") for k in r["request"]["questions"])]   # H4: all of them
    rest = by["step_done"] + [r for r in by["plan_arrived"] if r not in checks]
    rng.shuffle(rest)
    return out + checks + rest[: max(0, n - len(out) - len(checks))]


def plant(recs: list[dict], k: int, rng: random.Random) -> list[dict]:
    """plan_arrived requests from sort scenarios (the task asks about every numbered block) with one
    block's step and done condition taken out of the plan: the plan check should say yes."""
    cands = [r for r in recs if r["event"] == "plan_arrived" and r["scenario"].startswith("sort_")
             and "numbered blocks" in r["request"]["state"]["task"] and r["request"]["state"].get("plan", {}).get("steps_to_run")]
    rng.shuffle(cands)
    out = []
    for r in cands:
        if len(out) >= k:
            break
        st = copy.deepcopy(r["request"]["state"])
        steps = st["plan"]["steps_to_run"]
        victims = [s for s in steps if "(block " in s]
        if len(victims) < 2:
            continue
        v = rng.choice(victims)
        name = v.split("(", 1)[1].split(" →")[0].split(" ->")[0].strip()          # "block 7"
        if not name.startswith("block ") or any(name in m for m in st["plan"]["objects_the_plan_never_mentions"]):
            continue
        st["plan"]["steps_to_run"] = [s for s in steps if s != v]
        st["plan"]["done_when"] = [c for c in st["plan"]["done_when"] if name.replace(" ", "_") + " " not in c + " "]
        st["plan"]["objects_the_plan_never_mentions"] = st["plan"]["objects_the_plan_never_mentions"] + [name]
        qs = copy.deepcopy(r["request"]["questions"])
        qs[f"check.{name.replace(' ', '_')}"] = {
            "type": "noul",
            "instructions": f"Is {name} something `task` asks about (something to move, arrange, hand over or keep away from somewhere) but `plan` leaves out?",
            "criteria": {"true": f"The task covers {name} (by name, number, colour or as one of a group it describes) and the plan has no step, done condition or constraint for it.",
                         "false": f"The task does not ask for anything about {name}; the plan is right to leave it alone."}}
        truth = {**r["truth"], f"check.{name.replace(' ', '_')}": True}
        out.append({**r, "src": r["src"] + ":planted", "planted": name, "request": {"state": st, "questions": qs}, "truth": truth,
                    "jev_then": None})
    return out


def right(answers: dict | None, rec: dict) -> dict:
    """{group: (choice, confidence, right)} as e12v2's metrics count it (only where the truth was offered)."""
    out = {}
    if not answers:
        return out
    T, qs = rec["truth"], rec["request"]["questions"]
    for g in GROUPS:
        if g in answers and g in qs and g in T:
            offered = list(qs[g]["criteria"])
            if not [t for t in T[g] if t in offered]:
                continue
            out[g] = (answers[g].get("choice"), float(answers[g].get("confidence") or 0.0), answers[g].get("choice") in T[g])
    return out


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--plants", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    recs = load()
    todo = sample(recs, a.n, rng) + plant(recs, a.plants, rng)
    print(f"{len(recs)} unique logged requests; replaying {len(todo)} ({Counter(r['event'] for r in todo)}), "
          f"{sum('planted' in r for r in todo)} planted")
    jev, dec = JevHTTP(max_requests=len(todo) + 5), DecisionsAPI(max_requests=len(todo) + 5)
    jev.warm(), dec.warm()

    def one(rec):
        st, qs = rec["request"]["state"], rec["request"]["questions"]
        return {**rec, "jev": jev.answer(st, qs), "decisions": dec.answer(st, qs)}

    out_path = ROOT / "results/live" / f"replay_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool, out_path.open("w") as f:
        done = []
        for i, r in enumerate(pool.map(one, todo)):
            f.write(json.dumps(r) + "\n")
            done.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    print(f"wrote {out_path}")
    report(done)


def report(done: list[dict]) -> None:
    print("\n## Errors and latency (replay, %d requests in parallel)" % len(done))
    for b in ("jev", "decisions"):
        ok = [r for r in done if r[b]["answers"] is not None]
        lat = [r[b]["latency_ms"] for r in ok]
        up = [r[b]["meta"].get("upstream_ms") for r in ok if r[b]["meta"].get("upstream_ms")]
        print(f"{b:10s} errors {len(done) - len(ok)}/{len(done)}  latency p50 {pct(lat, 50):.0f} p95 {pct(lat, 95):.0f} max {max(lat):.0f} ms;"
              f" server p50 {pct(up, 50):.0f} p95 {pct(up, 95):.0f} ms; refusals {sum(len(r[b]['meta'].get('refusals', [])) for r in done)}")

    both = [r for r in done if r["jev"]["answers"] and r["decisions"]["answers"] and "planted" not in r]
    print("\n## Agreement and accuracy (same requests; truth = e12v2's oracle where it offered an answer)")
    print("| group | n | Jev = Decisions | Jev right | Decisions right | Jev (Sept log) right |")
    print("|---|---|---|---|---|---|")
    for g in GROUPS:
        n = agree = jr = dr = then_r = then_n = 0
        for r in both:
            ja, da = right(r["jev"]["answers"], r), right(r["decisions"]["answers"], r)
            if g in ja and g in da:
                n += 1
                agree += ja[g][0] == da[g][0]
                jr += ja[g][2]
                dr += da[g][2]
                th = right(r["jev_then"], r)
                if g in th:
                    then_n += 1
                    then_r += th[g][2]
        print(f"| {g} | {n} | {100 * agree / n:.1f}% | {100 * jr / n:.1f}% | {100 * dr / n:.1f}% | {100 * then_r / max(then_n, 1):.1f}% |")

    print("\n## The gate on each one's confidence (e12v2's gate_sweep: right-now, and in-plan fixes when routed stay_local)")
    for label, b, conf in (("Jev", "jev", None), ("Decisions (formula)", "decisions", "confidence_formula"),
                           ("Decisions (API field)", "decisions", "confidence_api")):
        arecs = []
        for r in both:
            ans = copy.deepcopy(r[b]["answers"])
            if conf:
                for g in GROUPS:
                    if g in ans and ans[g].get(conf) is not None:
                        ans[g]["confidence"] = ans[g][conf]
            rec = right(ans, r)
            if rec:
                arecs.append(rec)
        print(f"\n{label}")
        print("| gate | answers | wrong | caught | caught % | escalated/fallback % | acted-on accuracy % |")
        print("|---|---|---|---|---|---|---|")
        for row in gate_sweep(arecs, GATES):
            print(f"| {row['gate']:.1f} | {row['answers']} | {row['wrong']} | {row['caught']} | {row['caught_pct']:.0f} | "
                  f"{row['escalated_pct']:.1f} | {row['acted_accuracy']:.1f} |")

    diffs = [abs(a["confidence_api"] - a["confidence_formula"]) for r in both for g, a in r["decisions"]["answers"].items()
             if a.get("type") == "choice" and a.get("confidence_api") is not None]
    print(f"\nDecisions API confidence field vs (p_max-1/n)/(1-1/n): n {len(diffs)}, |diff| p50 {pct(diffs, 50):.3f} "
          f"p95 {pct(diffs, 95):.3f} max {max(diffs):.3f}; equal within 0.01: {100 * sum(d <= 0.01 for d in diffs) / len(diffs):.1f}%")

    print("\n## H4: the plan check (plan_arrived; a plan is sent back if any check Noul is >= 0.5 or unsure, |2p-1| < 0.5)")
    print("| backend | good plans | sent back | % | planted bad plans | caught | % |")
    print("|---|---|---|---|---|---|---|")
    for b in ("jev", "decisions"):
        good = back = bad = caught = 0
        for r in done:
            ans = r[b]["answers"]
            ks = [k for k in r["request"]["questions"] if k.startswith("check.")]
            if not ans or not ks:
                continue
            sent = any(float(ans.get(k, {}).get("noul", 0.5)) >= 0.5 or noul_conf(float(ans.get(k, {}).get("noul", 0.5))) < 0.5 for k in ks)
            if "planted" in r:
                bad += 1
                caught += sent
            elif not any(r["truth"].get(k) for k in ks):
                good += 1
                back += sent
        print(f"| {b} | {good} | {back} | {100 * back / max(good, 1):.0f}% | {bad} | {caught} | {100 * caught / max(bad, 1):.0f}% |")
    print("\n## H4 as e12v2 counted it: a new plan the oracle says to keep (route truth stay_local) that combine() at the")
    print("default gates sends back to the LLM anyway (plan-check problem, a replan route, or stay_local under 0.8)")
    print("| backend | good plans | sent back | % | of which unsure stay_local | confident replan | plan check | planted bad plans | caught | % |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    class V:
        hand = None

        @staticmethod
        def name(oid):
            return oid.replace("_", " ")
    for b in ("jev_then", "jev", "decisions"):
        good = back = low = conf = chk = bad = caught = 0
        for r in done:
            if r["event"] != "plan_arrived" or "route" not in r["request"]["questions"]:   # not the no-router ablation
                continue
            ans = r[b] if b == "jev_then" else r[b]["answers"]
            if not ans:
                continue
            c = combine(Event(0, 0, "plan_arrived", ""), ans, V(), Gates())
            sent = c.route != "stay_local"
            if "planted" in r:
                bad += 1
                caught += sent
            elif "stay_local" in (r["truth"].get("route") or []):
                good += 1
                back += sent
                if sent:
                    chk += bool(c.problems)
                    low += not c.problems and c.route_raw == "stay_local"
                    conf += not c.problems and c.route_raw != "stay_local"
        name = {"jev_then": "jev (Sept log)", "jev": "jev (now)", "decisions": "decisions"}[b]
        print(f"| {name} | {good} | {back} | {100 * back / max(good, 1):.0f}% | {low} | {conf} | {chk} | {bad or ''} | "
              f"{caught if bad else ''} | {f'{100 * caught / bad:.0f}%' if bad else ''} |")
    then = [r for r in done if r["jev_then"] and any(k.startswith("check.") for k in r["request"]["questions"]) and "planted" not in r]
    tb = sum(any(float(r["jev_then"].get(k, {}).get("noul", 0.5)) >= 0.5 or noul_conf(float(r["jev_then"].get(k, {}).get("noul", 0.5))) < 0.5
                 for k in r["request"]["questions"] if k.startswith("check.")) for r in then)
    print(f"| jev (Sept log, same requests) | {len(then)} | {tb} | {100 * tb / max(len(then), 1):.0f}% | | | |")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        report([json.loads(line) for line in (gzip.open(sys.argv[2], "rt") if sys.argv[2].endswith(".gz") else open(sys.argv[2]))])
    else:
        main()
