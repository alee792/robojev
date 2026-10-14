"""H4, offline: the recorded plan_arrived decisions re-routed by the brain's new rule (no network).

docs/live-results.md section 3 measured that the generic route question on `plan_arrived` sent 49%
(Jev) / 66% (Decisions API) of good new plans back to the planner, while the plan-check Nouls flagged
none of them. core/combine.py now routes a new plan by the plan check alone (docs/v2.md: "a new plan
is presumed good"). This replays the recorded requests and BOTH deciders' recorded answers through the
brain's combine() + enforce_gate() as they are now, and through a replica of the old rule, so the two
are compared on identical answers:

  - experiments/results/live/replay_*.jsonl.gz: the Mac's replay of e12v2's September requests to both
    live backends (185 good new plans with a route question, 60 planted bad plans), plus Jev's
    September answers to the same requests (`jev_then`);
  - experiments/results/live/slice_*.jsonl.gz: every plan_arrived decision of the live slice, with the
    acting backend's answer and the shadow backend's answer to the same request.

Truth is e12v2's oracle as logged: a good plan is one whose route truth includes stay_local; a planted
plan is one with a block's step removed and its plan-check question added. The old-rule replica is
checked against the numbers the Mac's report printed (91/185, 123/185, 60/60, 59/60) before anything
else is reported.

    uv run --frozen python experiments/replay_plan_route.py            # prints the tables
    uv run --frozen python experiments/replay_plan_route.py --write    # also writes results/live/replay_plan_route.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from robojev.brain.core.combine import Gates, combine, noul_conf
from robojev.brain.core.data import Event
from robojev.brain.rules import enforce_gate

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "results" / "live"
DECIDERS = ("jev", "decisions")
LLM = ("fast_llm", "capable_llm")
PUBLISHED = {("replay", "jev_then"): (91, 185, None, None), ("replay", "jev"): (91, 185, 60, 60),
             ("replay", "decisions"): (123, 185, 59, 60)}          # docs/live-results.md section 3


class _View:
    """What combine() reads of the view for a plan_arrived request: no hand, names for the problems."""
    hand = None

    @staticmethod
    def name(oid):
        return oid.replace("_", " ")


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def load_replay(path: Path) -> list[dict]:
    """-> [{src, decider -> answers, truth, questions, planted}] for every plan_arrived request with a route question."""
    out = []
    with _open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["event"] != "plan_arrived" or "route" not in r["request"]["questions"]:
                continue
            answers = {b: r[b]["answers"] for b in DECIDERS}
            answers["jev_then"] = r.get("jev_then")
            out.append({"src": r["src"], "questions": r["request"]["questions"], "truth": r["truth"],
                        "planted": "planted" in r, "answers": answers})
    return out


def load_slice(paths: list[Path]) -> list[dict]:
    """The live slice's plan_arrived decision requests; the acting and shadow backends by the file's tag."""
    out = []
    for path in paths:
        tag = path.name.split("_", 2)[1]                 # slice_<decider>-<planner>_<test>...
        acting = tag.split("-")[0]
        other = {"jev": "decisions", "decisions": "jev"}[acting]
        with _open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") != "decision_request" or r["rec"]["event"] != "plan_arrived":
                    continue
                rec = r["rec"]
                answers = {acting: rec["primary"].get("answers"), other: (rec.get("shadow") or {}).get("answers"), "jev_then": None}
                out.append({"src": f"{path.name}:{rec['t']}", "questions": rec["questions"], "truth": rec["truth"],
                            "planted": False, "answers": answers})
    return out


def problems_of(answers: dict, questions: dict) -> list[str]:
    """The plan-check findings as combine() makes them (a yes, or an unsure answer)."""
    out = []
    for k in questions:
        if k.startswith("check."):
            p = float((answers.get(k) or {}).get("noul", 0.0))
            if p >= 0.5 or noul_conf(p) < Gates().check:
                out.append(k)
    return out


def sent_back_before(answers: dict, questions: dict, gates: Gates = Gates()) -> tuple[bool, str]:
    """The rule before the fix, as combine() + the loop's enforce_gate applied it to plan_arrived: the
    route answer decides, a plan-check problem forces an LLM route, and a stay_local whose weakest
    confidence (route, in-plan fix) is under the gate goes to the LLM. -> (sent back, why)."""
    problems = problems_of(answers, questions)
    route = (answers.get("route") or {}).get("choice") or "fast_llm"
    c_route = float((answers.get("route") or {}).get("confidence") or 0.0)
    c_fix = float((answers.get("in_plan_fix") or {}).get("confidence") or 0.0) if "in_plan_fix" in answers else 1.0
    if problems:
        return True, "plan check"
    if route != "stay_local":
        return True, "confident replan" if c_route >= gates.stay_local else "replan"
    if min(c_route, c_fix) < gates.stay_local:
        return True, "unsure stay_local"
    return False, "kept"


def sent_back_after(answers: dict) -> tuple[bool, str]:
    """The brain's rule now: combine() on the recorded answers, then the loop's gate."""
    c = enforce_gate(combine(Event(0, 0, "plan_arrived", "a new plan arrived"), answers, _View(), Gates()))
    return c.route != "stay_local", c.reason


def tally(recs: list[dict], decider: str) -> dict:
    t = {"good": 0, "before": 0, "after": 0, "planted": 0, "caught_before": 0, "caught_after": 0,
         "before_why": Counter(), "after_why": Counter()}
    for r in recs:
        ans = r["answers"].get(decider)
        if not ans:
            continue
        b, why_b = sent_back_before(ans, r["questions"])
        a, why_a = sent_back_after(ans)
        if r["planted"]:
            t["planted"] += 1
            t["caught_before"] += b
            t["caught_after"] += a
        elif "stay_local" in (r["truth"].get("route") or []):
            t["good"] += 1
            t["before"] += b
            t["after"] += a
            if b:
                t["before_why"][why_b] += 1
            if a:
                t["after_why"][why_a] += 1
    return t


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else ""


def report(sources: dict[str, list[dict]]) -> str:
    lines = ["## H4 after the fix: plan_arrived routed by the plan check alone (offline replay of recorded answers)", "",
             "| source | decider | good new plans | sent back before | sent back after | planted bad plans | caught before | caught after |",
             "|---|---|---|---|---|---|---|---|"]
    checks = []
    for src, recs in sources.items():
        for d in ("jev_then", "jev", "decisions"):
            t = tally(recs, d)
            if not t["good"] and not t["planted"]:
                continue
            name = {"jev_then": "Jev, September answers", "jev": "Jev", "decisions": "Decisions API"}[d]
            lines.append(f"| {src} | {name} | {t['good']} | {t['before']} ({pct(t['before'], t['good'])}) | {t['after']} ({pct(t['after'], t['good'])}) | "
                         f"{t['planted'] or ''} | {t['caught_before'] if t['planted'] else ''} | {t['caught_after'] if t['planted'] else ''} |")
            if t["before_why"]:
                lines.append(f"|  |  | before, why: {dict(t['before_why'])} |  |  |  |  |  |")
            pub = PUBLISHED.get((src, d))
            if pub:
                want = (pub[0], pub[1], pub[2] if pub[2] is not None else t["caught_before"], pub[3] if pub[3] is not None else t["planted"])
                got = (t["before"], t["good"], t["caught_before"], t["planted"])
                checks.append((src, d, want, got))
    lines.append("")
    lines.append("Old-rule replica against the numbers docs/live-results.md section 3 printed (sent back/good, caught/planted):")
    for src, d, want, got in checks:
        ok = want == got
        planted = f", {want[2]}/{want[3]}; replica {got[0]}/{got[1]}, {got[2]}/{got[3]}" if want[3] else f"; replica {got[0]}/{got[1]}"
        lines.append(f"- {src} {d}: published {want[0]}/{want[1]}{planted} {'OK' if ok else 'MISMATCH'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=None, help="a replay_*.jsonl(.gz); default: the newest in results/live")
    ap.add_argument("--write", action="store_true", help="also write results/live/replay_plan_route.txt")
    a = ap.parse_args(argv)
    replay = a.replay or sorted(LIVE.glob("replay_*.jsonl*"))[-1]
    sources = {"replay": load_replay(replay), "slice": load_slice(sorted(LIVE.glob("slice_*.jsonl*")))}
    text = f"replay file: {replay.name}; slice files: {len(list(LIVE.glob('slice_*.jsonl*')))}\n\n" + report(sources)
    print(text)
    if a.write:
        out = LIVE / "replay_plan_route.txt"
        out.write_text(text + "\n")
        print(f"\nwrote {out}")
    return 0 if "MISMATCH" not in text else 1


if __name__ == "__main__":
    sys.exit(main())
