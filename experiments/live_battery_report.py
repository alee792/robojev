"""What the brain's log says in live battery runs (experiments/results/battery/<stamp>-sim-<decider>/),
beside the judge's table: plans sent back and why, escalations, decision latency, the reading guard.

    uv run --frozen python experiments/live_battery_report.py DIR [DIR ...]

A plan sent back is a `plan_arrived` decision routed off stay_local. The battery has no oracle, so a
sent-back plan is counted "flag on a spare letter" when every object the plan check flagged carries
a letter the word in force needs fewer of than the table holds (leaving one out is what a right plan
does), and "flag on a needed letter" otherwise. That is a reading of the flag, not a grade of the plan.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from robojev.brain.rules import TRANSIENT_RE, is_transient

WAITISH = re.compile(r"\b(wait|waiting|pause|paused|hang on|hold on|moment|sec|second|go on|resume|continue|carry on)\b", re.I)


def label(oid: str) -> str:
    m = re.match(r"letter_([A-Z])\d*$", oid)
    return m.group(1) if m else "?"


def load(path: Path) -> dict:
    ep = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        ep[r["type"]].append(r if r["type"] in ("episode", "judge") else r["rec"])
    return ep


def word_at(ep, t: float) -> str:
    word = ep["episode"][0]["word"]
    for f in ep["fired"]:
        if f["kind"] == "new_word" and f.get("what") in (None, "text") and f["t_s"] <= t:
            e = ep["episode"][0]["events"][f["index"]]
            word = e.get("word") or word
    return word


def table_count(ep, letter: str) -> int:
    e0 = ep["episode"][0]
    n = e0["pool"].count(letter)
    n += sum(1 for f in ep["fired"] if f["kind"] == "letter_added" and f.get("what") == "added" and (f.get("object") or "").startswith(f"letter_{letter}"))
    return n


def main(dirs: list[str]) -> None:
    for d in dirs:
        d = Path(d)
        files = sorted(d.glob("ep_*.jsonl"))
        if not files:
            continue
        name = files[0].stem.split("_", 3)[-1]
        plans = sent = spare = needed = firsts = first_sent = 0
        kept = loops = trusted = misfit = stale = 0
        esc = Counter()
        lat = []
        per_ep = []
        waits = []
        for f in files:
            ep = load(f)
            j = ep["judge"][0]
            said = []
            ep_sent = 0
            for rec in ep["brain_log"]:
                n = rec["note"]
                kept += n == "reading_kept"
                loops += n == "loop"
                trusted += n == "plan_trusted"
                misfit += n == "plan_misfit"
                stale += n == "plan_stale"
                if n == "event" and rec["event"] == "user_text":
                    said.append(re.sub(r'^the user typed: "(.*)"$', r"\1", rec["text"]))
                if n == "plan_applied":
                    reading = (rec.get("plan") or rec.get("diff") or {}).get("reading") or ""
                    if any(is_transient(s) for s in said) or any(WAITISH.search(s) for s in said):
                        if WAITISH.search(reading) or TRANSIENT_RE.search(reading):
                            waits.append((f.name, reading))
            for dec in ep["decisions"]:
                c = dec["combined"]
                if dec.get("latency_ms") is not None:
                    lat.append(dec["latency_ms"])
                if c["route"] != "stay_local":
                    esc[(dec["event"], c["reason"])] += 1
                if dec["event"] != "plan_arrived":
                    continue
                first = dec["text"].startswith("the first plan")
                plans += not first
                firsts += first
                if c["route"] == "stay_local":
                    continue
                sent += not first
                first_sent += first
                ep_sent += 1
                checked = [k.split(".", 1)[1] for k in c["conf"] if k.startswith("check.")]
                flagged = [o for o in checked if any(_name_hit(o, p) for p in c["problems"])]
                w = word_at(ep, dec["t_event"])
                if flagged and all(w.count(label(o)) < table_count(ep, label(o)) for o in flagged):
                    spare += 1
                else:
                    needed += 1
            per_ep.append((j["index"], j["word"], "".join(x or "_" for x in j["row"]), j["outcome"], j["planner_calls"], ep_sent))
        print(f"== {name} ({d.name})")
        print(f"new plans {plans}; sent back {sent} ({100 * sent / plans if plans else 0:.0f}%): flag on a spare letter {spare}, on a needed letter or other {needed} (first plans: {first_sent} of {firsts} sent back, counted in the flag split)")
        print(f"replan-loop limit hit {loops}; plans trusted by the loop rule {trusted}; misfits {misfit}; stale {stale}; reading_kept {kept}")
        if lat:
            q = statistics.quantiles(lat, n=20)
            print(f"decisions {len(lat)}: p50 {statistics.median(lat):.0f} ms, p95 {q[18]:.0f} ms, max {max(lat):.0f} ms")
        print("off stay_local, by event and reason:", dict(esc.most_common()))
        for w in waits:
            print("  reading after a wait-ish line:", w)
        for row in per_ep:
            print("  ep %d %-6s %-6s %-8s planner %d sent back %d" % row)


_ORD = {"second": "2", "third": "3", "fourth": "4"}


def _name_hit(oid: str, problem: str) -> bool:
    """Whether a plan-check problem line ("the second letter A may be ...") names this object id."""
    m = re.match(r"the (?:(second|third|fourth) )?letter ([A-Z])\b", problem)
    if not m:
        return False
    suffix = _ORD.get(m.group(1), "") if m.group(1) else ""
    return oid == f"letter_{m.group(2)}{suffix}"


if __name__ == "__main__":
    main(sys.argv[1:])
