"""Offline: what the brain rules added after the live re-run would have changed in the recorded runs.

docs/live-results.md ("Live re-run after the fixes") found four rule gaps, and the loop now has
rules for them (src/robojev/brain/loop.py "Liveness", core/combine.py "A routine step_done"). None
of them can be fully replayed without the models (a changed question gets a new answer), so this
counts, from the recorded decisions alone, how many of each pattern the runs had and which of them
the new rules settle in code:

  routine      step_done decisions the 0.8 gate escalated ("low confidence") with no fix and no
               plan-check problem: now kept local on the decider's choice (core/combine.routine).
  paused_idle  paused_idle re-asks that left the arm paused: a hand pause now ends in code one idle
               period after the hand has gone; any other pause asks the user after the second re-ask.
  spare        plan_arrived send-backs whose only problems were "unsure" plan-check answers: the
               plan check now asks about counts per label (core/decision.unmentioned_groups); the
               answer is the model's, so this is the size of the pattern, not a result.
  words        user_text decisions the gate escalated from an unsure stay_local: those lines joined
               `user_words`; now only the decider's own planner/ask_user routes do.
  replan_fail  failed replans after which the hold was released (the old plan ran): now retried once,
               then the holding arms pause and the user is asked.

Reads experiments/results/battery/*/ep_*.jsonl and experiments/results/live/slice_*.jsonl.gz.

    uv run --frozen python experiments/replay_rules.py            # prints the table
    uv run --frozen python experiments/replay_rules.py --write    # also writes results/battery/replay_rules.txt
    uv run --frozen python experiments/replay_rules.py --after 20261013-095000 --write
        # only recordings stamped after that (one live run's), written to replay_rules_after_<stamp>.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robojev.brain.rules import is_transient  # noqa: E402

ROOT = Path(__file__).resolve().parent / "results"


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def decisions_and_llm(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """-> (decision records with `combined`, planner records, brain_log notes), whichever log shape."""
    decs, llm, log = [], [], []
    for r in records(path):
        rec = r.get("rec", r)
        kind = r.get("kind") or r.get("type")
        if kind in ("decision", "decisions") and isinstance(rec, dict) and "combined" in rec:
            decs.append(rec)
        elif kind == "llm" and isinstance(rec, dict) and "ok" in rec:
            llm.append(rec)
        elif kind == "brain_log" and isinstance(rec, dict):
            log.append(rec)
    return decs, llm, log


def classify(decs: list[dict], llm: list[dict], log: list[dict]) -> Counter:
    c = Counter()
    for d in decs:
        cb = d["combined"]
        ev, reason, route, fix = d["event"], cb.get("reason", ""), cb.get("route"), cb.get("fix")
        escalated = route == "fast_llm" and (reason == "low confidence" or reason.startswith("weakest confidence"))
        if ev == "step_done":
            c["step_done"] += 1
            if route in ("fast_llm", "capable_llm"):
                c["step_done replans"] += 1
            if escalated and fix == "none" and not cb.get("problems"):
                c["routine: now local"] += 1
        elif ev == "paused_idle":
            c["paused_idle re-asks"] += 1
            if cb.get("right_now") in ("carry_on", "pause"):
                c["paused_idle: stayed paused"] += 1
                if "hand" in d.get("text", ""):
                    c["paused_idle: hand pause (code resumes)"] += 1
                else:
                    c["paused_idle: other pause (user asked at the 2nd)"] += 1
        elif ev == "plan_arrived" and route == "fast_llm" and reason == "plan check":
            c["plan sent back"] += 1
            if all("(unsure)" in p for p in cb.get("problems", [])):
                c["spare: unsure-only send-back"] += 1
        elif ev == "user_text" and escalated:
            said = d.get("text", "").split('the user typed: "', 1)[-1].rstrip('"')
            if not is_transient(said):                     # a transient line never joined (rules.is_transient)
                c["words: unsure line joined user_words"] += 1
    for r in llm:
        if r.get("kind") == "replan" and not r.get("ok"):
            c["replan_fail: hold released"] += 1
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--after", default=None, help="only recordings whose stamp (YYYYMMDD-HHMMSS) is later; "
                    "runs are grouped by decider, so without it several live runs' files are counted together")
    a = ap.parse_args(argv)

    def keep(stamp: str) -> bool:
        return a.after is None or stamp > a.after
    runs: dict[str, list[Path]] = defaultdict(list)
    for p in sorted((ROOT / "battery").glob("2026*-sim-*/ep_*.jsonl")):
        if not keep(p.parent.name.split("-sim-")[0]):
            continue
        runs["battery " + p.parent.name.split("-sim-")[1]].append(p)
    for p in sorted((ROOT / "live").glob("slice_*_2026101[23]-*.jsonl.gz")):
        if not keep(p.name.rsplit("_", 1)[1].split(".")[0]):
            continue
        runs["slice " + p.name.split("_")[1].split("-")[0]].append(p)
    lines = ["| run | files | step_done | step_done replans | routine: now local | paused_idle re-asks | stayed paused | hand (code resumes) | other (user asked) | plans sent back | unsure-only (spare) | words: unsure lines | failed replans |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_episode: dict[str, list[tuple[str, Counter]]] = defaultdict(list)
    for name, paths in runs.items():
        tot = Counter()
        for p in paths:
            c = classify(*decisions_and_llm(p))
            per_episode[name].append((p.name, c))
            tot += c
        lines.append(f"| {name} | {len(paths)} | {tot['step_done']} | {tot['step_done replans']} | {tot['routine: now local']} | "
                     f"{tot['paused_idle re-asks']} | {tot['paused_idle: stayed paused']} | {tot['paused_idle: hand pause (code resumes)']} | "
                     f"{tot['paused_idle: other pause (user asked at the 2nd)']} | {tot['plan sent back']} | {tot['spare: unsure-only send-back']} | "
                     f"{tot['words: unsure line joined user_words']} | {tot['replan_fail: hold released']} |")
    out = ["# Recorded runs under the rules added after the live re-run (offline count; module doc has the columns)", ""] + lines + [""]
    for name, eps in per_episode.items():
        stuck = [(n, c["paused_idle re-asks"]) for n, c in eps if c["paused_idle: stayed paused"] >= 2]
        if stuck:
            out.append(f"- {name}: episodes with 2+ paused_idle re-asks that stayed paused: " + ", ".join(f"{n} ({k})" for n, k in stuck))
    text = "\n".join(out) + "\n"
    print(text)
    if a.write:
        out_path = ROOT / "battery" / ("replay_rules.txt" if a.after is None else f"replay_rules_after_{a.after}.txt")
        out_path.write_text(text)
        print("wrote", out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
