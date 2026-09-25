"""E12 update (e12v2): the v2 design closed loop in a text world, against controls.

Arms (same scenarios, same seeds):
  jev                  the design: one Jev decision per event (three groups), code combines, LLM replans
  rules                no Jev: hand-written reactions (eval/controls.py lists the rules; --show-rules)
  always_llm           no Jev: every non-routine event goes to the LLM, which picks the reaction and replans
  oracle               perfect decisions from the evaluation oracle (the ceiling)
  jev-no-right-now     ablation: right-now is always carry on
  jev-no-in-plan-fix   ablation: no Sequencer group (fix is always none)
  jev-no-router        ablation: route = stay local if there is a fix (or the event is routine), else fast LLM

Run from the repo root (or from experiments/ with `uv run python e12v2.py ...`):
  uv run --frozen python experiments/e12v2.py --dry-run                     # samples + call counts + cost, no network
  uv run --frozen python experiments/e12v2.py                               # offline: mock Jev + mock LLM, every arm x scenario
  uv run --frozen python experiments/e12v2.py --mock-jev-error 0.15 --mock-llm-error 0.1 --noise
  uv run --frozen python experiments/e12v2.py --jev jev                     # live Jev, mock LLM (TYPESAFE_API_KEY)
  cd experiments && OPENAI_MODEL=<model> uv run --extra llm python e12v2.py --jev jev --llm openai --llm-effort low   # live, both
  uv run --frozen python experiments/e12v2.py --replay experiments/results/e12v2_<run>.jsonl   # gate sweep from a log, no calls
(`--extra llm` needs the openai package: from experiments/, `uv run --extra llm python e12v2.py ...`.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from common import RESULTS, payload

from ..core.combine import Ablation, Gates, JevDecider
from ..core.harness import Episode, HarnessConfig
from ..core.planner import PlannerClient
from ..sim.skills import make_skills
from ..sim.world import Noise
from . import metrics as M
from . import scenarios as S
from .controls import RULES, AlwaysLLMDecider, RulesDecider
from .mocks import MockJev, MockLLM

ARMS = ["jev", "rules", "always_llm", "oracle", "jev-no-right-now", "jev-no-in-plan-fix", "jev-no-router"]
NOISE_ON = Noise(misread=0.0, drop=0.02, jitter_cm=0.5)


@dataclass
class RunOpts:
    jev: str = "mock"
    llm: str = "mock"
    mock_jev_error: float = 0.05
    mock_conf_noise: float = 0.1
    mock_llm_error: float = 0.0
    noise: Noise | None = None
    gates: Gates = field(default_factory=Gates)
    max_s: float = 240.0
    max_llm_calls: int = 16
    jev_backend: object = None          # shared live Jev client
    llm_backends: dict | None = None    # shared live LLM backends {"fast_llm": ..., "capable_llm": ...}
    retry: bool = True


def ablation_of(arm: str) -> Ablation:
    return Ablation(right_now="no-right-now" not in arm, in_plan_fix="no-in-plan-fix" not in arm, router="no-router" not in arm)


def run_episode(arm: str, name: str, seed: int, o: RunOpts, log=None, wrap_jev=None, wrap_llm=None):
    sc = S.make(name, seed, o.noise)
    w = sc.world
    rs = seed * 1000 + zlib.crc32(name.encode()) % 1000      # mock randomness: per scenario and seed, same for every arm
    cfg = HarnessConfig(max_s=o.max_s, max_llm_calls=o.max_llm_calls)
    if arm == "rules":
        decider = RulesDecider()
    elif arm == "always_llm":
        decider = AlwaysLLMDecider()
        cfg.decide_plan_arrived = False
    elif arm == "oracle":
        decider = JevDecider(MockJev(sc, w, 0.0, 0.0, rs, name="oracle"), o.gates)
        decider.name = "oracle"
    else:
        be = o.jev_backend if o.jev == "jev" else MockJev(sc, w, o.mock_jev_error, o.mock_conf_noise, rs * 7 + 1)
        if wrap_jev:
            be = wrap_jev(be)
        decider = JevDecider(be, o.gates, ablation_of(arm))
    if o.llm == "mock":
        llm = {"fast_llm": MockLLM(sc, w, rs * 11 + 2, o.mock_llm_error, median_ms=2700),
               "capable_llm": MockLLM(sc, w, rs * 11 + 3, o.mock_llm_error / 2, median_ms=5000)}
    else:
        llm = dict(o.llm_backends)
    if wrap_llm:
        llm = {k: wrap_llm(v) for k, v in llm.items()}
    obs = M.Observer(sc, w)
    lg = (lambda **rec: log(arm=arm, scenario=name, seed=seed, **rec)) if log else None
    ep = Episode(w, sc.person, make_skills(w), decider, PlannerClient(llm, retry=o.retry), cfg, lg, obs)
    res = ep.run(sc.task)
    rec = M.episode_record(arm, sc, seed, res, obs, w)
    if log:
        log(kind="episode", **{k: v for k, v in rec.items()})
    return rec, res, obs, ep


def run_all(arms, names, seeds, o: RunOpts, log=None, progress=True, **wrap) -> list[dict]:
    recs = []
    for arm in arms:
        t0 = time.time()
        for name in names:
            for seed in seeds:
                rec, *_ = run_episode(arm, name, seed, o, log, **wrap)
                recs.append(rec)
        if progress:
            done = sum(r["completed"] for r in recs if r["arm"] == arm)
            print(f"  {arm:20s} {done}/{len(names) * len(seeds)} completed  ({time.time() - t0:.1f} s wall)", flush=True)
    return recs


# ---------------------------------------------------------------- dry run


class CaptureJev:
    """Keeps the first request per event kind, preferring the full `jev` arm over the ablations."""

    def __init__(self, inner, store, arm="jev"):
        self.inner, self.store, self.live, self.arm = inner, store, False, arm

    def answer(self, state, questions, ref=None):
        ev, view = ref
        key = ev.kind
        if ev.kind == "scene_change":
            key += "_hand" if ev.data.get("change", {}).get("what") == "hand" else "_object"
        out = self.inner.answer(state, questions, ref)
        have = self.store.get(key)
        if have is None or (have["arm"] != "jev" and self.arm == "jev"):
            self.store[key] = {"arm": self.arm, "event": ev.text, "request": payload(state, questions), "mock_answers": out["answers"]}
        return out


class ForceMiss:
    """Dry run only: turns the decisions on matching object-moved events into a confident miss, so the
    rarer event kinds happen in the `jev` arm and can be sampled (a missed re-target makes a step fail;
    a missed re-queue leaves the plan finished with a done condition false)."""

    def __init__(self, inner, match: str, overrides: dict):
        self.inner, self.live, self.match, self.overrides = inner, False, match, overrides

    def answer(self, state, questions, ref=None):
        out = self.inner.answer(state, questions, ref)
        ev = ref[0]
        if ev.kind == "scene_change" and self.match in ev.text and out["answers"]:
            for k, v in self.overrides.items():
                if k in out["answers"]:
                    out["answers"][k] = {**out["answers"][k], "choice": v, "confidence": 1.0}
        return out


class CaptureLLM:
    def __init__(self, inner, store):
        self.inner, self.store = inner, store

    def call(self, req, ref=None):
        if req.kind not in self.store:
            from e13_branches.planner import openai_kwargs
            kw = openai_kwargs(req, "<OPENAI_MODEL>")
            kw["prompt_cache_key"] = "e12v2-planner"
            self.store[req.kind] = kw
        return self.inner.call(req, ref)


def dry_run(args, arms, names, seeds, o: RunOpts):
    out = RESULTS / "e12v2_samples"
    out.mkdir(parents=True, exist_ok=True)
    jev_s, llm_s = {}, {}
    print(f"dry run: {len(arms)} arms x {len(names)} scenarios x {len(seeds)} seeds, all offline (mock Jev + mock LLM)")
    recs = []
    for arm in arms:
        recs += run_all([arm], names, seeds, o, progress=False,
                        wrap_jev=lambda b, a=arm: CaptureJev(b, jev_s, a), wrap_llm=lambda b: CaptureLLM(b, llm_s))
    for kind, name, match, over in (("step_failed", "sort_moved_target", "moved by someone else", {"right_now": "carry_on"}),
                                    ("plan_done_unmet", "sort_take_back", "tray slot", {"in_plan_fix": "none", "route": "stay_local"})):
        if jev_s.get(kind, {}).get("arm") != "jev":
            jev_s.pop(kind, None)
            run_episode("jev", name, seeds[0], o, wrap_jev=lambda b, m=match, v=over: CaptureJev(ForceMiss(b, m, v), jev_s,
                                                                                              f"jev ({name}: one decision forced to a miss)"))
    for k, v in jev_s.items():
        (out / f"jev_{k}.json").write_text(json.dumps(v, indent=1))
    for k, v in llm_s.items():
        (out / f"llm_{k}.json").write_text(json.dumps(v, indent=1))
    jev_arms = [a for a in arms if a.startswith("jev")]
    jreq = sum(r["jev_requests"] for r in recs if r["arm"] in jev_arms)
    jtok = sum(r["jev_tok"] for r in recs if r["arm"] in jev_arms)
    lcalls = sum(r["llm_calls"] for r in recs)
    lin = sum(r["llm_in"] for r in recs)
    lout = sum(r["llm_out"] for r in recs)
    print(f"samples: {len(jev_s)} Jev decision requests ({', '.join(sorted(jev_s))}) and {len(llm_s)} LLM requests "
          f"({', '.join(sorted(llm_s))}) in {out}")
    print("\nper arm (from the mock run; a live run makes about the same calls):")
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        print(f"  {arm:20s} Jev requests {sum(r['jev_requests'] for r in rs):5d}   LLM calls {sum(r['llm_calls'] for r in rs):4d}")
    print(f"\nJev (live arms: {', '.join(jev_arms)}): {jreq} requests, ~{jtok / 1e3:.0f}k input tokens "
          f"≈ ${jtok * M.JEV_PRICE:.3f} at $0.042/M")
    print(f"LLM: {lcalls} calls, ~{lin / 1e3:.0f}k input / ~{lout / 1e3:.0f}k visible output tokens "
          f"(reasoning models add hidden output; budget 2-5x)")
    if args.llm_price_in is not None and args.llm_price_out is not None:
        cost = lin * args.llm_price_in / 1e6 + 3 * lout * args.llm_price_out / 1e6
        print(f"LLM cost at ${args.llm_price_in}/M in, ${args.llm_price_out}/M out (x3 output for reasoning) ≈ ${cost:.2f}")
    else:
        print("LLM cost: pass --llm-price-in/--llm-price-out ($ per M tokens); no model price is assumed")


# ---------------------------------------------------------------- report


def report(recs, arms, gate, title, price_in=None, price_out=None):
    core = [r for r in recs if not r["held_out"]]
    held = [r for r in recs if r["held_out"]]
    print(M.table(core, f"{title}: core scenarios", arms))
    if held:
        print(M.table(held, f"{title}: HELD-OUT scenarios (reported separately, never tuned on)", arms))
    ans = [a for r in recs if r["arm"] == "jev" for a in r["answers"]]
    print(M.sweep_table(M.gate_sweep(ans)))
    print(M.pass_criteria(recs, gate, ans))
    jv = [r for r in recs if r["arm"] != "oracle"]          # the oracle arm never calls Jev
    jt = sum(r["jev_tok"] for r in jv)
    li, lo, lc = (sum(r[k] for r in recs) for k in ("llm_in", "llm_out", "llm_cached"))
    cost = f" ≈ ${li * price_in / 1e6 + lo * price_out / 1e6:.3f}" if price_in is not None and price_out is not None else \
        " (pass --llm-price-in/--llm-price-out for $)"
    print(f"\nCost: Jev {sum(r['jev_requests'] for r in jv)} requests, {jt} input tokens ≈ ${jt * M.JEV_PRICE:.4f}; "
          f"LLM {sum(r['llm_calls'] for r in recs)} calls, {li} input ({lc} cached) / {lo} output tokens{cost}")


def load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip() and '"kind": "episode"' in line]


def main(argv=None):
    ap = argparse.ArgumentParser(description="E12 update: v2 design closed loop with controls", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated; default all: " + ",".join(ARMS))
    ap.add_argument("--scenarios", default="all", help="comma-separated names, 'core', 'held_out' or 'all'")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per scenario (block layouts)")
    ap.add_argument("--seed", type=int, default=12, help="first seed")
    ap.add_argument("--jev", choices=["jev", "mock"], default="mock")
    ap.add_argument("--llm", choices=["openai", "mock"], default="mock")
    ap.add_argument("--llm-model", help="fast LLM (OpenAI model id; else env OPENAI_MODEL; no default)")
    ap.add_argument("--llm-capable-model", help="capable LLM (default: the fast model)")
    ap.add_argument("--llm-effort", choices=["minimal", "low", "medium", "high"], default=None)
    ap.add_argument("--llm-timeout", type=float, default=30.0)
    ap.add_argument("--llm-price-in", type=float, default=None, help="$ per M input tokens (cost estimate only)")
    ap.add_argument("--llm-price-out", type=float, default=None, help="$ per M output tokens (cost estimate only)")
    ap.add_argument("--mock-jev-error", type=float, default=0.05)
    ap.add_argument("--mock-conf-noise", type=float, default=0.1)
    ap.add_argument("--mock-llm-error", type=float, default=0.0)
    ap.add_argument("--noise", action="store_true", help="perception noise (jitter 0.5 cm, 2%% drop-outs)")
    ap.add_argument("--gate", type=float, default=0.7, help="the stay_local gate (in-plan fix); pass criterion 4 is read here")
    ap.add_argument("--gate-right-now", type=float, default=0.5)
    ap.add_argument("--gate-fast", type=float, default=0.3)
    ap.add_argument("--no-right-now", action="store_true", help="run the jev arm with right-now off")
    ap.add_argument("--no-in-plan-fix", action="store_true", help="run the jev arm with the in-plan fix off")
    ap.add_argument("--no-router", action="store_true", help="run the jev arm with the router off")
    ap.add_argument("--max-s", type=float, default=240.0, help="sim seconds per episode before timeout")
    ap.add_argument("--max-llm-calls-episode", type=int, default=16)
    ap.add_argument("--max-jev-requests", type=int, default=2500, help="live: stop cleanly after this many Jev requests")
    ap.add_argument("--max-llm-calls", type=int, default=600, help="live: stop cleanly after this many LLM calls")
    ap.add_argument("--log", action="store_true", help="write a JSONL log for mock runs too (live runs always log)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", type=Path, help="gate sweep + tables from a JSONL log, no calls")
    ap.add_argument("--show-rules", action="store_true", help="print the rules control's rule list and exit")
    args = ap.parse_args(argv)

    if args.show_rules:
        print(RULES)
        return
    gates = Gates(right_now=args.gate_right_now, stay_local=args.gate, fast_llm=args.gate_fast)
    if args.replay:
        recs = load_log(args.replay)
        arms = [a for a in ARMS if any(r["arm"] == a for r in recs)] + sorted({r["arm"] for r in recs} - set(ARMS))
        report(recs, arms, args.gate, f"replay {args.replay.name}")
        return
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.no_right_now or args.no_in_plan_fix or args.no_router:
        suffix = "".join(f"-no-{n}" for n, on in (("right-now", args.no_right_now), ("in-plan-fix", args.no_in_plan_fix),
                                                   ("router", args.no_router)) if on)
        arms = [("jev" + suffix) if a == "jev" else a for a in arms]
    for a in arms:
        if a not in ARMS and not a.startswith("jev-no-"):
            raise SystemExit(f"unknown arm {a}")
    names = (list(S.CORE) + list(S.HELD_OUT) if args.scenarios == "all" else list(S.CORE) if args.scenarios == "core"
             else list(S.HELD_OUT) if args.scenarios == "held_out" else [n.strip() for n in args.scenarios.split(",")])
    seeds = list(range(args.seed, args.seed + args.seeds))
    o = RunOpts(mock_jev_error=args.mock_jev_error, mock_conf_noise=args.mock_conf_noise, mock_llm_error=args.mock_llm_error,
                noise=NOISE_ON if args.noise else None, gates=gates, max_s=args.max_s, max_llm_calls=args.max_llm_calls_episode)
    if args.dry_run:
        dry_run(args, arms, names, seeds, o)
        return
    live = args.jev == "jev" or args.llm == "openai"
    if live:   # state the calls up front: the same run with mocks makes about the same calls
        est = run_all(arms, names, seeds, o, progress=False)
        jr = sum(r["jev_requests"] for r in est if r["arm"].startswith("jev")) if args.jev == "jev" else 0
        jt = sum(r["jev_tok"] for r in est if r["arm"].startswith("jev")) if args.jev == "jev" else 0
        lc = sum(r["llm_calls"] for r in est) if args.llm == "openai" else 0
        print(f"expected (from the same run with mocks): ~{jr} Jev requests (~${jt * M.JEV_PRICE:.3f}), ~{lc} OpenAI calls; "
              f"caps: --max-jev-requests {args.max_jev_requests}, --max-llm-calls {args.max_llm_calls}", flush=True)
    o.jev, o.llm = args.jev, args.llm
    from ..backends import BudgetExceeded, JevHTTP, openai_backend
    if args.jev == "jev":
        o.jev_backend = JevHTTP(args.max_jev_requests)
    if args.llm == "openai":
        model = args.llm_model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise SystemExit("--llm openai needs --llm-model or OPENAI_MODEL (no default is assumed; see context/07-experiments-to-run.md for the suggested model)")
        cap = args.llm_capable_model or model
        o.llm_backends = {"fast_llm": openai_backend(model, args.llm_timeout, args.llm_effort, max_calls=args.max_llm_calls),
                          "capable_llm": openai_backend(cap, args.llm_timeout, args.llm_effort, max_calls=args.max_llm_calls)}
    log = None
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if live or args.log:
        path = RESULTS / f"e12v2_{run_id}.jsonl"
        RESULTS.mkdir(exist_ok=True)
        f = path.open("a")

        def log(**rec):
            rec.setdefault("run", run_id)
            f.write(json.dumps(rec, default=lambda x: getattr(x, "__dict__", str(x))) + "\n")
        log(kind="run", args=vars(args) | {"replay": None}, arms=arms, scenarios=names, seeds=seeds, gates=gates.as_dict())
    mock = "+".join(x for x, m in (("Jev", args.jev == "mock"), ("LLM", args.llm == "mock")) if m)
    print(f"E12 update: {len(arms)} arms x {len(names)} scenarios x {len(seeds)} seeds; Jev={args.jev} LLM={args.llm}"
          + (f" (MOCK {mock}: pipeline check, not evidence)" if mock else ""))
    try:
        recs = run_all(arms, names, seeds, o, log)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
        return
    title = "E12 update" + (f" (MOCK {mock})" if mock else "")
    report(recs, arms, args.gate, title, args.llm_price_in, args.llm_price_out)
    if log:
        print(f"\nlog: {path}")
