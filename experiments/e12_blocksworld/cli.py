"""E12: text-only blocks world, closed loop. Tests v2's fast layers (Sequencer, Spotter, Listener) with
no arm, no MuJoCo and no camera.

Run from the repo root (or from experiments/ with plain `uv run`):
  uv run --frozen python experiments/e12_blocksworld.py --dry-run                   # no network: samples + cost
  uv run --frozen python experiments/e12_blocksworld.py --backend oracle            # upper bound, all scenarios
  uv run --frozen python experiments/e12_blocksworld.py --backend mock --mock-error 0.1
  uv run --frozen python experiments/e12_blocksworld.py --backend jev               # needs TYPESAFE_API_KEY
  uv run --frozen python experiments/e12_blocksworld.py --backend jev --sweep       # + raw / no-spotter / no-listener
  uv run --frozen --extra vlm python experiments/e12_blocksworld.py --backend claude --scenario reverse_midway
"""

from __future__ import annotations

import argparse
import json
import time

from common import RESULTS, Log, pct

from . import backends as B
from .harness import PRICE_PER_TOKEN, SWEEP, Metrics, Variant, run_one
from .scenarios import SCENARIOS, make
from .world import NOISE_ON, TICKS_PER_S

AGREE_COLS = ["sequencer.next_skill", "sequencer.task_status", "sequencer.escalate", "spotter.action", "spotter.order",
              "listener.intent", "listener.knob:order", "listener.knob:skip", "listener.knob:destination", "listener.when", "listener.escalate"]


def _utt(m: Metrics) -> str:
    parts = []
    if m.utt_to_change:
        parts.append("/".join(str(t) for t in m.utt_to_change))
    if m.utt_missed:
        parts.append(f"missed {m.utt_missed}")
    if m.utt_spurious:
        parts.append(f"SPURIOUS {m.utt_spurious}")
    return " ".join(parts) or "-"


def print_table(title: str, rows: list[tuple[str, Metrics]]):
    print(f"\n==================== {title} ====================")
    hdr = (f"{'scenario':<16} {'done':<4} {'time s':>6} {'skills':>6} {'dist→rec':>8} {'fails':>5} {'req seq/spot/lis':>16} {'req/tick':>8} "
           f"{'esc':>3} {'gated':>5} {'utt→chg ticks':>14} {'touch':>5} {'hand':>4} {'floor':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        r = m.requests
        req = f"{r['sequencer']}/{r['spotter']}/{r['listener']}"
        fr = f"{m.disturbances}→{m.recovered}"
        t = f"{m.done_tick / TICKS_PER_S:.1f}" if m.done_tick is not None else "-"
        done = "yes" if m.completed else ("CAP" if m.budget_hit else "NO")
        print(f"{name:<16} {done:<4} {t:>6} {m.skills:>6} {fr:>8} {m.failed:>5} "
              f"{req:>16} {m.request_ticks / max(1, m.ticks):>8.1%} "
              f"{m.escalations:>3} {m.gated:>5} {_utt(m):>14} {m.forbidden_touch:>5} {m.hand_contact:>4} {m.floor_stops:>5}")
    tot = sum(sum(m.requests.values()) for _, m in rows)
    print(f"completed {sum(m.completed for _, m in rows)}/{len(rows)}; requests {tot}; "
          f"violations: forbidden touches {sum(m.forbidden_touch for _, m in rows)}, hand contacts {sum(m.hand_contact for _, m in rows)}")
    agree: dict = {}
    for _, m in rows:
        for k, (ok, n) in m.agree.items():
            a = agree.setdefault(k, [0, 0])
            a[0] += ok
            a[1] += n
    cells = [f"{k.split('.', 1)[1]}={100 * agree[k][0] / agree[k][1]:.0f}% (n={agree[k][1]})" for k in AGREE_COLS if k in agree]
    print("agreement with the oracle, per question: " + ", ".join(cells))
    lat = [x for _, m in rows for x in m.latency_ms if x]
    if lat:
        print(f"latency ms: p50={pct(lat, 50):.0f} p95={pct(lat, 95):.0f} max={max(lat):.0f}")
    tok = sum(m.tokens for _, m in rows)
    if tok:
        print(f"input tokens {tok} ≈ ${tok * PRICE_PER_TOKEN:.4f} at Jev's price")


def run_matrix(scenarios, backend_name, variants, args, log=None, capture=None) -> dict:
    out = {}
    for v in variants:
        rows = []
        for s in scenarios:
            backend = B.make(backend_name, args.mock_error, args.mock_conf_noise, args.seed) if backend_name in ("oracle", "mock") else args._backend
            cap = {} if capture is not None else None
            m = run_one(make(s, args.seed), backend, v, latency_ticks=args.latency_ticks, planner_delay_s=args.planner_delay,
                        noise=NOISE_ON if args.noise else None, seed=args.seed, max_s=args.max_s, log=log, capture=cap,
                        escalation=args.escalation, max_requests=args.max_requests)
            if capture is not None:
                capture[(s, v.name)] = cap
            rows.append((s, m))
        out[v.name] = rows
    return out


def dry_run(scenarios, variants, args):
    """No network: run the oracle through every scenario, write one sample request per layer per
    scenario, and count requests and tokens for the live runs."""
    out = RESULTS / "e12_samples"
    out.mkdir(parents=True, exist_ok=True)
    capture: dict = {}
    todo = list({v.name: v for v in [*variants, *SWEEP]}.values())
    res = run_matrix(scenarios, "oracle", todo, args, capture=capture)
    written = 0
    for s in scenarios:
        for layer, rec in capture.get((s, variants[0].name), {}).items():
            (out / f"{s}_{layer}.json").write_text(json.dumps(rec, indent=1))
            written += 1
    for s in scenarios:
        plan = make(s, args.seed).plan
        (out / f"plan_{plan['template']}.json").write_text(json.dumps(plan, indent=1))
    print(f"dry run: wrote {written} sample requests and the compiled Plans to {out}")

    def cost(names):
        rows = [m for n in names for _, m in res[n]]
        req = sum(sum(m.requests.values()) for m in rows)
        tok = sum(m.chars for m in rows) / 3.5
        return req, tok

    sel = [v.name for v in variants]
    for label, names in (("this run (" + ", ".join(sel) + ")", sel), ("--sweep (" + ", ".join(v.name for v in SWEEP) + ")", [v.name for v in SWEEP])):
        req, tok = cost(names)
        print(f"  {label}: {len(scenarios)} scenarios, {req} requests (oracle behaviour), ~{tok / 1e3:.0f}k input tokens "
              f"≈ ${tok * PRICE_PER_TOKEN:.3f} on Jev; x1.5 margin for a noisier live run ≈ ${1.5 * tok * PRICE_PER_TOKEN:.3f}")
    per = {layer: 0 for layer in ("sequencer", "spotter", "listener")}
    for _, m in res[variants[0].name]:
        for k, v in m.requests.items():
            per[k] += v
    print(f"  requests per layer ({variants[0].name}): {per}")


def parse(argv=None):
    ap = argparse.ArgumentParser(description="E12: text-only blocks world, closed loop over v2's fast layers.")
    ap.add_argument("--scenario", default="all", help="one of: " + ", ".join(SCENARIOS) + ", or all (comma list ok)")
    ap.add_argument("--backend", default="oracle", choices=["oracle", "mock", "jev", "claude"])
    ap.add_argument("--facts", default="solved", help="raw | solved | raw,solved")
    ap.add_argument("--no-spotter", action="store_true")
    ap.add_argument("--no-listener", action="store_true", help="every utterance goes straight to the Planner escalation path")
    ap.add_argument("--sweep", action="store_true", help="run the ablation set: solved, raw, solved-nospotter, solved-nolistener")
    ap.add_argument("--mock-error", type=float, default=0.1)
    ap.add_argument("--mock-conf-noise", type=float, default=0.1)
    ap.add_argument("--latency-ticks", default="2", help="ticks from request to applied answer, or 'auto' (measured latency)")
    ap.add_argument("--planner-delay", type=float, default=2.0, help="simulated Planner (escalation) delay, s of sim time")
    ap.add_argument("--escalation", default="oracle", choices=["oracle", "claude"])
    ap.add_argument("--noise", action="store_true", help="perception noise: misread 3%%, drop 2%%, jitter 0.5 cm")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--max-s", type=float, default=300.0, help="sim-time cap per scenario")
    ap.add_argument("--max-requests", type=int, default=600, help="per scenario; a live run that hits it stops, marked incomplete")
    ap.add_argument("--dry-run", action="store_true", help="no network: write sample requests and estimate cost")
    ap.add_argument("--log", action="store_true", help="also log offline (oracle/mock) runs; live runs always log")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    args.latency_ticks = args.latency_ticks if args.latency_ticks == "auto" else int(args.latency_ticks)
    return args


def variants_of(args) -> list[Variant]:
    if args.sweep:
        return list(SWEEP)
    return [Variant(f, spotter=not args.no_spotter, listener=not args.no_listener) for f in args.facts.split(",") if f]


def main(argv=None):
    args = parse(argv)
    scenarios = list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario {s}; choose from {', '.join(SCENARIOS)}")
    variants = variants_of(args)
    if args.dry_run:
        return dry_run(scenarios, variants, args)
    live = args.backend in ("jev", "claude")
    args._backend = B.make(args.backend) if live else None
    log = None
    if not args.no_log and (live or args.log):
        log = Log("e12_blocksworld" if live else f"e12_blocksworld_{args.backend}")
        print(f"logging every tick to {log.path}")
    t0 = time.time()
    try:
        res = run_matrix(scenarios, args.backend, variants, args, log=log)
    finally:
        if args._backend:
            args._backend.close()
    label = args.backend + (f" (error {args.mock_error}, conf noise {args.mock_conf_noise})" if args.backend == "mock" else "")
    for vname, rows in res.items():
        print_table(f"{label} | {vname} | latency {args.latency_ticks} ticks | planner delay {args.planner_delay} s", rows)
    print(f"\nwall time {time.time() - t0:.1f} s")
    return res
