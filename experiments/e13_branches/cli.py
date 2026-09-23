"""E13: a fast LLM prepares the plan's branches; Jev picks among them on a mid-task correction; low
confidence escalates back to the LLM. Open loop (no arm), scored against a hand-written oracle.

Run from experiments/ (the openai SDK is the optional extra `llm`):
  uv run python e13_branches.py --dry-run                                       # no network: samples + call counts + cost
  uv run python e13_branches.py --llm mock --jev mock                           # offline end to end
  uv run python e13_branches.py --llm mock --jev mock --mock-llm-error 0.15 --mock-jev-error 0.15
  OPENAI_MODEL=<id> uv run --extra llm python e13_branches.py --llm openai --jev jev          # live: ~30 tasks x 5 corrections
  uv run --extra llm python e13_branches.py --llm openai --jev jev --llm-model <id> --max-llm-calls 60
  uv run python e13_branches.py --replay results/e13_branches.jsonl --gate 0.5  # re-score a live log, no calls
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from common import RESULTS, Log, payload

from . import oracle
from . import plan as P
from . import planner as PL
from . import router as R
from . import score as S
from .world import make_tasks, scene_view

JEV_PRICE = 0.042e-6   # $/input token


class BudgetExceeded(Exception):
    pass


class Counting:
    """Wraps an LLM backend: counts calls and enforces --max-llm-calls."""

    def __init__(self, backend, cap: int | None):
        self.backend, self.cap, self.n = backend, cap, 0

    def call(self, req, ref=None):
        if self.cap is not None and self.n >= self.cap:
            raise BudgetExceeded
        self.n += 1
        return self.backend.call(req, ref)


def _counts(n_tasks: int, n_corr: int, baseline: bool) -> str:
    esc = f"{n_corr}" if baseline else f"0-{n_corr} (only when Jev escalates)"
    return (f"planned calls: LLM {n_tasks} plans (+ up to {n_tasks} validation retries) + {esc} escalations "
            f"(+ retries); Jev {n_corr}")


def _jev_tokens(body: dict) -> float:
    return len(json.dumps(body)) / 3.5   # rough chars/token for JSON, as in e11


def dry_run(tasks, args):
    out = RESULTS / "e13_samples"
    out.mkdir(parents=True, exist_ok=True)
    from .mocks import MockLLM
    mock = MockLLM(seed=args.seed)
    n_corr = sum(len(t.corrections) for t in tasks)
    llm_in = llm_out = jev_tok = 0.0
    for i, t in enumerate(tasks):
        sv = scene_view(t.scene)
        req = PL.plan_request(t.text, sv)
        res = mock.call(req, ref=({"task": t}, 0))
        plan = json.loads(res.raw)
        llm_in += (len(req.instructions) + len(req.input) + len(json.dumps(req.schema))) / 4
        llm_out += len(res.raw) / 4
        cur = P.default_settings(plan)
        for j, c in enumerate(t.corrections):
            ereq = PL.escalation_request(t.text, sv, plan, cur, c.text)
            llm_in += (len(ereq.instructions) + len(ereq.input) + len(json.dumps(ereq.schema))) / 4
            llm_out += len(res.raw) / 4 / 2
            body = payload(R.jev_state(t.text, c.text, plan, cur), R.jev_questions(plan))
            jev_tok += _jev_tokens(body)
            if i < 3 and j < 2:
                tr = S.truth(t, c, plan, cur)
                (out / f"jev_{c.cid}.json").write_text(json.dumps({"request": body, "truth": tr, "category": c.category,
                                                                    "note": "plan made by the mock LLM"}, indent=1))
        if i < 3:
            (out / f"llm_plan_{t.tid}.json").write_text(json.dumps({"openai_kwargs": PL.openai_kwargs(req, "<OPENAI_MODEL>"),
                                                                   "oracle_canonical_default": oracle.target(t).canonical}, indent=1))
            c = next(x for x in t.corrections if x.category == "uncoverable")
            ereq = PL.escalation_request(t.text, sv, plan, cur, c.text)
            (out / f"llm_escalation_{c.cid}.json").write_text(json.dumps({"openai_kwargs": PL.openai_kwargs(ereq, "<OPENAI_MODEL>"),
                                                                        "oracle_canonical_after": oracle.target(t, c).canonical}, indent=1))
    n_esc = n_corr if args.baseline else n_corr * 0.4
    per_esc_in = llm_in / (len(tasks) + n_corr)
    print(f"dry run: {len(tasks)} tasks, {n_corr} corrections")
    print(_counts(len(tasks), n_corr, args.baseline))
    print(f"Jev: {n_corr} requests, ~{jev_tok / 1e3:.0f}k input tokens ≈ ${jev_tok * JEV_PRICE:.4f}")
    est_in = llm_in if args.baseline else llm_in - (n_corr - n_esc) * per_esc_in
    print(f"LLM: ~{est_in / 1e3:.0f}k input tokens, ~{llm_out / 1e3:.0f}k output tokens before reasoning tokens "
          f"(reasoning models add hidden output tokens; multiply output by ~2-5)")
    if args.llm_price_in is not None and args.llm_price_out is not None:
        print(f"LLM cost at ${args.llm_price_in}/M in, ${args.llm_price_out}/M out (x3 output for reasoning): "
              f"≈ ${est_in * args.llm_price_in / 1e6 + 3 * llm_out * args.llm_price_out / 1e6:.2f}")
    else:
        print("LLM cost: pass --llm-price-in/--llm-price-out ($ per M tokens) for a $ estimate; no model price is assumed")
    print(f"samples in {out}")


def _llm_backend(args):
    if args.llm == "mock":
        from .mocks import MockLLM
        e = args.mock_llm_error
        return MockLLM(seed=args.seed, wrong_default=e, missing_branch=e, invalid=e, esc_wrong=e / 2)
    model = args.llm_model or os.environ.get("OPENAI_MODEL")
    if not model:
        ids = PL.list_models("terra")
        print("No OpenAI model chosen (--llm-model or OPENAI_MODEL); there is no default.")
        print("Models on this account whose id contains 'terra':" + ("\n  " + "\n  ".join(ids) if ids else " none"))
        raise SystemExit("choose one and rerun with --llm-model <id>")
    return PL.OpenAIBackend(model, timeout_s=args.llm_timeout, max_output_tokens=args.llm_max_output, effort=args.llm_effort)


def run(tasks, args) -> tuple[list[dict], list[dict], dict]:
    llm = Counting(_llm_backend(args), args.max_llm_calls)
    if args.jev == "mock":
        from .mocks import MockJev
        jev = MockJev(seed=args.seed + 1, error=args.mock_jev_error)
    else:
        jev = R.JevBackend()
    live = args.llm == "openai" or args.jev == "jev"
    log = Log("e13_branches") if live else (Log(f"e13_branches_{args.llm}_{args.jev}") if args.log else None)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    meta = {"kind": "run", "run": run_id, "args": {k: v for k, v in vars(args).items()}, "llm_model": getattr(llm.backend, "model", "mock")}
    if log:
        log.write(**meta)
    plans, recs = [], []
    stopped = None
    try:
        for ti, t in enumerate(tasks):
            sv = scene_view(t.scene)
            try:
                po = PL.get_plan(llm, PL.plan_request(t.text, sv), sv, ref={"task": t}, retry=not args.no_retry)
            except BudgetExceeded:
                stopped = f"--max-llm-calls {args.max_llm_calls} reached before task {t.tid}"
                break
            prec = {"kind": "plan", "run": run_id, "tid": t.tid, "family": t.family, "text": t.text, "valid_first": po.valid_first,
                    "valid": po.plan is not None, "latency_ms": po.latency_ms, "in_tok": po.in_tok, "out_tok": po.out_tok,
                    "attempts": po.attempts, "plan": po.plan, "default_ok": False, "n_params": 0, "n_branches": 0, "coverage": []}
            if po.plan:
                plan = po.plan
                cur = P.default_settings(plan)
                prec["default_ok"] = oracle.target(t).check(P.goal_of(P.default_branch(plan)))
                prec["n_params"], prec["n_branches"] = len(plan["parameters"]), len(plan["branches"])
                prec["coverage"] = [{"cid": c.cid, "text": c.text,
                                     "covered": any(oracle.target(t, c).check(g) for g in P.branch_goals(plan).values())}
                                    for c in t.corrections if c.category == "coverable"]
            plans.append(prec)
            if log:
                log.write(**prec)
            if not po.plan:
                continue
            for c in t.corrections:
                tr = S.truth(t, c, plan, cur)
                state, qs = R.jev_state(t.text, c.text, plan, cur), R.jev_questions(plan)
                jr = jev.answer(state, qs, ref={"truth": tr, "plan": plan, "current": cur})
                if not jr["in_tok"]:
                    jr["in_tok"] = round(_jev_tokens(payload(state, qs)))
                d = R.decide(plan, cur, jr["answers"], args.gate)
                esc = None
                if args.baseline or d.action == "escalate":
                    cur_goal = P.goal_of(next(b for b in plan["branches"] if P.combo_key(plan, P.settings_of(b)) == P.combo_key(plan, cur)))
                    try:
                        eo = PL.get_plan(llm, PL.escalation_request(t.text, sv, plan, cur, c.text), sv,
                                         ref={"task": t, "correction": c, "current_goal": cur_goal}, retry=not args.no_retry)
                    except BudgetExceeded:
                        stopped = f"--max-llm-calls {args.max_llm_calls} reached during task {t.tid}"
                        eo = None
                    if eo is not None:
                        esc = {"baseline": args.baseline, "valid": eo.plan is not None, "valid_first": eo.valid_first,
                               "correct": bool(eo.plan) and oracle.target(t, c).check(P.goal_of(P.default_branch(eo.plan))),
                               "latency_ms": eo.latency_ms, "in_tok": eo.in_tok, "out_tok": eo.out_tok,
                               "attempts": eo.attempts, "plan": eo.plan}
                rec = {"kind": "correction", "run": run_id, "tid": t.tid, "family": t.family, "cid": c.cid, "category": c.category,
                       "text": c.text, "plan": P.lite(plan), "current": cur, "truth": tr, "default_ok": prec["default_ok"],
                       "jev": {k: jr[k] for k in ("status", "latency_ms", "upstream_ms", "in_tok", "answers", "body")},
                       "jev_request": payload(state, qs) if live else None,
                       "decision": vars(d), "esc": esc}
                if log:
                    log.write(**rec)
                if esc is None and d.action == "escalate":
                    break          # budget hit mid-task: this correction has no outcome; stop
                recs.append(rec)
            if stopped:
                break
            if (ti + 1) % 5 == 0:
                print(f"  {ti + 1}/{len(tasks)} tasks, LLM calls {llm.n}")
    finally:
        jev.close()
    info = {"llm_calls": llm.n, "stopped": stopped, "log": str(log.path) if log else None,
            "jev_tokens": sum(r["jev"]["in_tok"] for r in recs)}
    return plans, recs, info


def load_log(path: Path, run_id: str | None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    runs = [r["run"] for r in rows if r.get("kind") == "run"]
    if not runs:
        raise SystemExit(f"no runs in {path}")
    rid = run_id or runs[-1]
    plans = [r for r in rows if r.get("kind") == "plan" and (rid == "all" or r["run"] == rid)]
    recs = [r for r in rows if r.get("kind") == "correction" and (rid == "all" or r["run"] == rid)]
    return plans, recs, rid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--llm", choices=["openai", "mock"], default="mock")
    ap.add_argument("--jev", choices=["jev", "mock"], default="mock")
    ap.add_argument("--llm-model", help="OpenAI model id (else env OPENAI_MODEL; no default)")
    ap.add_argument("--llm-timeout", type=float, default=30.0, help="seconds per LLM call (no retries)")
    ap.add_argument("--llm-max-output", type=int, default=None, help="max_output_tokens (unset by default: reasoning tokens count too)")
    ap.add_argument("--llm-effort", choices=["minimal", "low", "medium", "high"], default=None, help="reasoning effort, for reasoning models only")
    ap.add_argument("--llm-price-in", type=float, default=None, help="$ per M input tokens, for the cost estimate only")
    ap.add_argument("--llm-price-out", type=float, default=None, help="$ per M output tokens, for the cost estimate only")
    ap.add_argument("--tasks", type=int, default=28, help="tasks (spread over the 7 task families)")
    ap.add_argument("--corrections", type=int, default=5, help="corrections per task")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--gate", type=float, default=0.7, help="escalate when the weakest relevant Jev confidence is below this")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="escalate only when Jev does (fewer LLM calls; no always-LLM column, partial gate sweep)")
    ap.add_argument("--no-retry", action="store_true", help="no second LLM attempt after a validation failure")
    ap.add_argument("--max-llm-calls", type=int, default=None, help="stop cleanly after this many LLM calls")
    ap.add_argument("--mock-llm-error", type=float, default=0.1)
    ap.add_argument("--mock-jev-error", type=float, default=0.1)
    ap.add_argument("--log", action="store_true", help="log mock runs too (live runs always log)")
    ap.add_argument("--dry-run", action="store_true", help="build every request, write samples, no network")
    ap.add_argument("--replay", type=Path, help="re-score a JSONL log (no calls)")
    ap.add_argument("--run-id", help="with --replay: which run (default: the last; 'all' for every run)")
    args = ap.parse_args(argv)

    if args.replay:
        plans, recs, rid = load_log(args.replay, args.run_id)
        S.print_all(plans, recs, args.gate, f"replay {args.replay.name} run {rid}")
        return

    tasks = make_tasks(args.tasks, args.corrections, args.seed)
    n_corr = sum(len(t.corrections) for t in tasks)
    if args.dry_run:
        dry_run(tasks, args)
        return
    print(f"E13: {len(tasks)} tasks, {n_corr} corrections; LLM={args.llm} Jev={args.jev} gate={args.gate}")
    print(_counts(len(tasks), n_corr, args.baseline) + (f"; LLM calls capped at {args.max_llm_calls}" if args.max_llm_calls else ""))
    plans, recs, info = run(tasks, args)
    mock = " (MOCK " + "+".join(x for x, m in (("LLM", args.llm == "mock"), ("Jev", args.jev == "mock")) if m) + ")" if "mock" in (args.llm, args.jev) else ""
    S.print_all(plans, recs, args.gate, "E13" + mock)
    print(f"\nLLM calls {info['llm_calls']}; Jev input tokens {info['jev_tokens']} ≈ ${info['jev_tokens'] * JEV_PRICE:.4f}"
          + (f"; log {info['log']}" if info["log"] else ""))
    if info["stopped"]:
        print("STOPPED: " + info["stopped"])
