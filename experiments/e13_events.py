"""E13 annotated event stream: turn a live E13 log into a timeline of what happened, with latency, tokens
and cost on every model call. Reads the log only; no calls, and nothing here feeds back into scoring.

For each task: the LLM plan call. For each correction, two timelines that start when the user speaks:
  system      Jev answers (route + confidence) -> the gate keeps it, or escalates to the LLM
  always-LLM  the same correction sent straight to the LLM (the baseline)
The run is open loop and sequential, so times are per correction (t=0 at the correction), not wall
time; the LLM escalation is placed after Jev's answer, as the design would run it.

  uv run python e13_events.py results/e13_branches.jsonl                 # every run in the log
  uv run python e13_events.py results/e13_branches.jsonl --run 2026...   # one run
Writes results/e13_events_<run>.jsonl (machine-readable) and .md (annotated), and prints a per-run
latency / token / cost summary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import pct
from e13_branches.score import outcome

JEV_PRICE = 0.042e-6   # $/input token (same constant as e13_branches/cli.py)


def load(path):
    runs = defaultdict(lambda: {"meta": None, "plans": [], "recs": []})
    for line in open(path):
        r = json.loads(line)
        rid = r.get("run")
        if r["kind"] == "run":
            runs[rid]["meta"] = r
        elif r["kind"] == "plan":
            runs[rid]["plans"].append(r)
        elif r["kind"] == "correction":
            runs[rid]["recs"].append(r)
    return runs


def llm_cost(in_tok, out_tok, meta):
    a = meta["args"]
    if a.get("llm_price_in") is None or a.get("llm_price_out") is None:
        return None
    return in_tok * a["llm_price_in"] / 1e6 + out_tok * a["llm_price_out"] / 1e6


def call_stats(attempts, meta):
    in_tok = sum(a["in_tok"] for a in attempts)
    out_tok = sum(a["out_tok"] for a in attempts)
    return {"latency_ms": round(sum(a["latency_ms"] for a in attempts)), "attempts": len(attempts),
            "in_tok": in_tok, "out_tok": out_tok,
            "reasoning_tok": sum(a.get("reasoning_tok", 0) or 0 for a in attempts),
            "cached_tok": sum(a.get("cached_tok", 0) or 0 for a in attempts),
            "cost_usd": llm_cost(in_tok, out_tok, meta),
            "errors": [e for a in attempts for e in (a.get("errors") or [])] + [a["error"] for a in attempts if a.get("error")]}


def events_for(run):
    meta, gate = run["meta"], run["meta"]["args"]["gate"]
    model = meta.get("llm_model")
    plans = {p["tid"]: p for p in run["plans"]}
    for tid, p in plans.items():
        s = call_stats(p["attempts"], meta)
        yield {"type": "plan", "tid": tid, "family": p["family"], "task": p["text"], "model": model, **s,
               "valid_first": p["valid_first"], "valid": p["valid"], "default_ok": p["default_ok"],
               "n_params": p["n_params"], "n_branches": p["n_branches"],
               "coverable_covered": sum(c["covered"] for c in p["coverage"]), "coverable": len(p["coverage"]),
               "note": plan_note(p)}
    for r in run["recs"]:
        o = outcome(r, gate)
        d = o["d"]
        j = r["jev"]
        ans = (j.get("answers") or {}).get("route") or {}
        jev_cost = j["in_tok"] * JEV_PRICE
        ev = {"type": "correction", "cid": r["cid"], "tid": r["tid"], "family": r["family"], "category": r["category"],
              "said": r["text"], "truth_route": r["truth"]["route"], "gate": gate, "timeline": []}
        tl = ev["timeline"]
        tl.append({"t_ms": 0, "who": "user", "what": f'says "{r["text"]}"'})
        tl.append({"t_ms": round(j["latency_ms"]), "who": "jev", "what": f'route={ans.get("choice")} conf={ans.get("confidence")}',
                   "latency_ms": round(j["latency_ms"]), "server_ms": j.get("upstream_ms"), "in_tok": j["in_tok"],
                   "cost_usd": jev_cost, "status": j.get("status")})
        tl.append({"t_ms": round(j["latency_ms"]), "who": "gate",
                   "what": f"{d.action} ({d.reason}; weakest confidence {d.min_conf if d.min_conf is None else round(d.min_conf, 2)} vs gate {gate})"})
        e = r.get("esc")
        es = call_stats(e["attempts"], meta) if e else None
        if o["path"] == "escalated" and e:
            tl.append({"t_ms": round(j["latency_ms"] + e["latency_ms"]), "who": "llm",
                       "what": f"new plan, {'correct' if e['correct'] else 'WRONG'}" + ("" if e["valid"] else " (invalid)"), **es})
        ev["system"] = {"path": o["path"], "correct": o["correct"], "latency_ms": None if o["latency"] is None else round(o["latency"]),
                        "cost_usd": jev_cost + (es["cost_usd"] or 0 if o["path"] == "escalated" and es else 0)}
        if e:
            ev["always_llm"] = {"correct": bool(e["valid"] and e["correct"]), "latency_ms": round(e["latency_ms"]), **es}
        ev["note"] = correction_note(r, o, e)
        yield ev


def plan_note(p):
    if not p["valid"]:
        return "no valid plan after retry"
    bits = []
    if not p["default_ok"]:
        bits.append("default branch WRONG")
    if p["coverage"] and not any(c["covered"] for c in p["coverage"]):
        bits.append(f"no branch for any of the {len(p['coverage'])} coverable corrections")
    if p["n_branches"] <= 1:
        bits.append("single branch: every change must go back to the LLM")
    return "; ".join(bits) or "ok"


def correction_note(r, o, e):
    d = o["d"]
    if o["path"] == "jev":
        return ("Jev handled it" if o["correct"] else "Jev handled it WRONG (mistake not caught by the gate)") + f" [{d.reason}]"
    why = {"route_new_plan": "Jev said no prepared branch fits", "low_confidence": "Jev unsure, gate escalated",
           "missing_branch": "Jev's pick has no prepared branch"}.get(d.reason, d.reason)
    if not e:
        return why + "; no LLM outcome"
    return why + ("; LLM got it right" if e["valid"] and e["correct"] else "; LLM got it WRONG")


def summary(run, evs):
    meta = run["meta"]
    plans = [e for e in evs if e["type"] == "plan"]
    cors = [e for e in evs if e["type"] == "correction"]
    jl = [c["timeline"][1]["latency_ms"] for c in cors]
    pl = [p["latency_ms"] for p in plans]
    el = [c["always_llm"]["latency_ms"] for c in cors if "always_llm" in c]
    sl = [c["system"]["latency_ms"] for c in cors if c["system"]["latency_ms"] is not None]

    def q(xs):
        return f"{pct(xs, 50):.0f} / {pct(xs, 95):.0f} / {max(xs):.0f}" if xs else "-"

    def tok(xs, k):
        return sum(x[k] for x in xs) / len(xs) if xs else 0

    esc = [c["always_llm"] for c in cors if "always_llm" in c]
    cost_llm = sum((p["cost_usd"] or 0) for p in plans) + sum((x["cost_usd"] or 0) for x in esc)
    cost_jev = sum(c["timeline"][1]["cost_usd"] for c in cors)
    sys_ok = sum(bool(c["system"]["correct"]) for c in cors)
    llm_ok = sum(c["always_llm"]["correct"] for c in cors if "always_llm" in c)
    jev_path = [c for c in cors if c["system"]["path"] == "jev"]
    lines = [
        f"run {meta['run']}  model {meta.get('llm_model')}  gate {meta['args']['gate']}  "
        f"tasks {len(plans)}  corrections {len(cors)}",
        f"  latency ms p50/p95/max:  Jev {q(jl)} | LLM plan {q(pl)} | LLM per correction {q(el)} | system per correction {q(sl)}",
        f"  LLM tokens per plan: in {tok(plans, 'in_tok'):.0f} (cached {tok(plans, 'cached_tok'):.0f}), out {tok(plans, 'out_tok'):.0f} "
        f"(reasoning {tok(plans, 'reasoning_tok'):.0f});  per correction: in {tok(esc, 'in_tok'):.0f}, out {tok(esc, 'out_tok'):.0f} "
        f"(reasoning {tok(esc, 'reasoning_tok'):.0f})",
        f"  Jev tokens per request: in {sum(c['timeline'][1]['in_tok'] for c in cors) / max(1, len(cors)):.0f}",
        f"  cost: LLM ${cost_llm:.3f} (plans ${sum((p['cost_usd'] or 0) for p in plans):.3f}, corrections ${sum((x['cost_usd'] or 0) for x in esc):.3f}) + Jev ${cost_jev:.4f}",
        f"  correct: system {sys_ok}/{len(cors)}, always-LLM {llm_ok}/{len(esc)};  Jev handled {len(jev_path)} "
        f"({sum(bool(c['system']['correct']) for c in jev_path)} right) at p50 {pct([c['system']['latency_ms'] for c in jev_path], 50) if jev_path else 0:.0f} ms",
    ]
    return "\n".join(lines)


def to_md(run, evs):
    meta = run["meta"]
    out = [f"# E13 event stream: run {meta['run']} ({meta.get('llm_model')}, gate {meta['args']['gate']})", "",
           "Times are per correction (t=0 when the user speaks). `always-LLM` is the baseline: the same correction sent straight to the LLM.", "",
           "```", summary(run, evs), "```", ""]
    by_task = defaultdict(list)
    for e in evs:
        by_task[e["tid"]].append(e)
    for tid, es in by_task.items():
        p = next((e for e in es if e["type"] == "plan"), None)
        if p:
            c = f"${p['cost_usd']:.4f}" if p["cost_usd"] is not None else "?"
            out += [f"## {tid}: {p['task']}", "",
                    f"- **LLM plan** {p['latency_ms']} ms, {p['in_tok']} in / {p['out_tok']} out ({p['reasoning_tok']} reasoning), {c}, "
                    f"{p['n_params']} params, {p['n_branches']} branches, covers {p['coverable_covered']}/{p['coverable']} coverable corrections. **{p['note']}**", ""]
        for e in es:
            if e["type"] != "correction":
                continue
            ok = e["system"]["correct"]
            mark = "✓" if ok else ("✗" if ok is False else "?")
            out.append(f"- {mark} `{e['category']}` \"{e['said']}\" (want {'/'.join(e['truth_route'])}) → {e['note']}")
            for t in e["timeline"][1:]:
                extra = ""
                if t["who"] == "jev":
                    extra = f" · server {t.get('server_ms')} ms · {t['in_tok']} tok · ${t['cost_usd']:.6f}"
                elif t["who"] == "llm":
                    extra = f" · {t['in_tok']} in / {t['out_tok']} out ({t['reasoning_tok']} reasoning) · " + (f"${t['cost_usd']:.4f}" if t["cost_usd"] is not None else "?")
                out.append(f"    - t={t['t_ms']:>5} ms  {t['who']:<4} {t['what']}{extra}")
            if "always_llm" in e:
                a = e["always_llm"]
                out.append(f"    - always-LLM: {'right' if a['correct'] else 'WRONG'} at {a['latency_ms']} ms")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", default=None, help="one run id (default: every live run in the log)")
    ap.add_argument("--out", default=None, help="directory for the outputs (default: the log's directory)")
    a = ap.parse_args()
    runs = load(a.log)
    out_dir = Path(a.out or Path(a.log).parent)
    for rid, run in runs.items():
        if a.run and rid != a.run or run["meta"] is None or not run["recs"]:
            continue
        evs = list(events_for(run))
        (out_dir / f"e13_events_{rid}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))
        (out_dir / f"e13_events_{rid}.md").write_text(to_md(run, evs))
        print(summary(run, evs), "\n")


if __name__ == "__main__":
    main()
