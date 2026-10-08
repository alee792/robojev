"""E11: reorder after a mid-task intent change, without an LLM planner.

A row of numbered blocks; the user asked for one order, then says something new ("actually, reverse
it"). The arm may be holding a block it lifted from a slot (leaving a gap). Every question family is
a separate Jev question in one request (answers never inform each other), scored against a solver in
code. The experimental axis is how much the state precomputes:
  raw    — the row, the held block, the old order, the user's text
  facts  — raw + code-computed facts for the OLD order only
  solved — raw + code-computed facts under BOTH orders (Jev only has to map the intent to a branch)

Ground truth (see `truth()`):
  slot convention  slot 1 is the robot's LEFT end; ascending = smallest in slot 1.
  effective order  "reverse"-type text flips the old order; "biggest first"-type text names one;
                   everything else (continue, pace, stop, not-for-me, ...) keeps the old order.
  new_order        correct = the effective order's name, or "unchanged" when that equals the old order.
  intent           one label, except an explicit order equal to the old one accepts adjust_order or continue.
  target_slot      the held block's rank among ALL blocks (row + held) under the effective order.
  direction        target < lifted-from slot -> carry it left; > -> right; == -> put it back.
  next_pick        (hand empty) the block that belongs in the LEFTMOST slot whose block is wrong under the
                   effective order; "nothing" if the row is already in order.
  escalate         no truth; scored as "does it ask for help more on requests Jev got wrong".
direction / target_slot / next_pick are only asked when the text keeps the blocks task going
(adjust_order / continue / adjust_pace / not_for_me), not after stop / pause / new_task.

Run (from experiments/):  uv run python e11_reorder.py --dry-run        # no network, writes samples
                          uv run python e11_reorder.py                  # 4 sizes x 20 scenes x 3 variants
                          uv run --extra baseline python e11_reorder.py --baseline claude
Cost: default 240 Jev calls x ~1-2k tok ≈ 0.33M tok ≈ $0.015. Baseline: 240 Haiku calls ≈ $0.4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass

from common import RESULTS, Log, ask, make_client, payload, pct, summarize

ASC, DESC = "ascending", "descending"
VARIANTS = ("raw", "facts", "solved")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
QNAMES = ("intent", "new_order", "direction", "target_slot", "next_pick", "escalate")
ACTION_INTENTS = {"adjust_order", "continue", "adjust_pace", "not_for_me"}
NOTHING = "nothing, the row is already in order"
LEFT, RIGHT, BACK = "carry it left", "carry it right", "put it back where it was"
BASELINE_MODEL = "claude-haiku-4-5"

# (text, what it does to the order, intent). Order kinds: flip | ascending | descending | same.
PHRASES = [
    ("actually, reverse it", "flip", "adjust_order"),
    ("wait, do it the other way round", "flip", "adjust_order"),
    ("biggest first", DESC, "adjust_order"),
    ("sort them high to low", DESC, "adjust_order"),
    ("smallest on the left please", ASC, "adjust_order"),
    ("sort them low to high", ASC, "adjust_order"),
    ("keep going the same way", "same", "continue"),
    ("go a bit slower", "same", "adjust_pace"),
    ("stop!", "same", "stop"),
    ("hold on a sec", "same", "pause"),
    ("forget the blocks, put the cup in the bowl", "same", "new_task"),
    ("Sam, can you grab me a coffee?", "same", "not_for_me"),
]
PHRASE_WEIGHTS = [2 if p[2] == "adjust_order" else 1 for p in PHRASES]  # ~2/3 order changes

ORDER_DEFS = {
    "slots": "Slots are numbered left to right as seen from the robot: slot 1 is the robot's left end, the highest slot number its right end.",
    ASC: "smallest number in slot 1 (robot's left), numbers increasing toward the right",
    DESC: "largest number in slot 1 (robot's left), numbers decreasing toward the right",
}


# ---------------------------------------------------------------- scenes + solver

@dataclass
class Scene:
    sid: str
    n: int
    values: list[int]          # every block number, sorted (row + held)
    row: list[int | None]      # block per slot; None = the gap the held block was lifted from
    held: int | None
    lifted_from: int | None    # 1-based slot
    old: str
    text: str
    kind: str
    intent: str
    dense: bool                # numbers are exactly 1..N (else distinct, sparse from 1..3N)


def other(order: str) -> str:
    return DESC if order == ASC else ASC


def target_row(values: list[int], order: str) -> list[int]:
    return sorted(values, reverse=(order == DESC))


def home_slot(block: int, values: list[int], order: str) -> int:
    return target_row(values, order).index(block) + 1


def direction(held: int, lifted_from: int, values: list[int], order: str) -> str:
    t = home_slot(held, values, order)
    return LEFT if t < lifted_from else RIGHT if t > lifted_from else BACK


def next_pick(row: list[int | None], values: list[int], order: str) -> int | None:
    """The block that belongs in the leftmost wrong slot; None if the row is already in order."""
    for have, want in zip(row, target_row(values, order)):
        if have != want:
            return want
    return None


def effective_order(old: str, kind: str) -> str:
    return other(old) if kind == "flip" else kind if kind in (ASC, DESC) else old


def asks_action(sc: Scene) -> bool:
    return sc.intent in ACTION_INTENTS


def truth(sc: Scene) -> dict:
    """Acceptable answers per question (a set; one element except where noted in the docstring)."""
    eff = effective_order(sc.old, sc.kind)
    t = {"intent": {sc.intent}, "new_order": {eff} | ({"unchanged"} if eff == sc.old else set())}
    if sc.kind in (ASC, DESC) and eff == sc.old:
        t["intent"] = {"adjust_order", "continue"}
    if asks_action(sc):
        if sc.held is not None:
            t["direction"] = {direction(sc.held, sc.lifted_from, sc.values, eff)}
            t["target_slot"] = {f"slot {home_slot(sc.held, sc.values, eff)}"}
        else:
            nb = next_pick(sc.row, sc.values, eff)
            t["next_pick"] = {NOTHING if nb is None else f"block {nb}"}
    return t


def make_scene(rng: random.Random, n: int, sid: str) -> Scene:
    dense = rng.random() < 0.5
    values = sorted(range(1, n + 1) if dense else rng.sample(range(1, 3 * n + 1), n))
    old = rng.choice((ASC, DESC))
    # mid-task under the OLD order: the first `progress` slots are already right, the rest shuffled
    progress = rng.randint(0, n - 2)
    want = target_row(values, old)
    rest = want[progress:]
    rng.shuffle(rest)
    row: list[int | None] = want[:progress] + rest
    held = lifted = None
    if rng.random() < 0.5:
        # half the time the arm lifted the block it needed next (old order), else any block
        nb = next_pick(row, values, old)
        held = nb if (nb is not None and rng.random() < 0.5) else rng.choice(values)
        lifted = row.index(held) + 1
        row[lifted - 1] = None
    text, kind, intent = rng.choices(PHRASES, weights=PHRASE_WEIGHTS)[0]
    return Scene(sid, n, values, row, held, lifted, old, text, kind, intent, dense)


def canonical() -> Scene:
    """Row [3,1,2,5,4], sorting ascending, arm holding 5 lifted from slot 4; user: 'actually, reverse it'."""
    return Scene("canonical", 5, [1, 2, 3, 4, 5], [3, 1, 2, None, 4], 5, 4, ASC, "actually, reverse it", "flip", "adjust_order", True)


def scenes(sizes, n_scenes: int, seed: int) -> list[Scene]:
    rng = random.Random(seed)
    out = []
    for n in sizes:
        if n == 5:
            out.append(canonical())
        out += [make_scene(rng, n, f"n{n}_{i:02d}") for i in range(n_scenes - (n == 5))]
    return out


# ---------------------------------------------------------------- state + questions

def _slot_facts(sc: Scene, order: str) -> dict:
    want = target_row(sc.values, order)
    slots = []
    for i, (have, w) in enumerate(zip(sc.row, want), 1):
        slots.append({
            "slot": i,
            "has": "empty" if have is None else f"block {have}",
            "belongs_here": f"block {w}",
            "correct": have == w,
        })
    f = {"order": order, "slots": slots, "wrong_slots": [s["slot"] for s in slots if not s["correct"]]}
    if sc.held is not None:
        t = home_slot(sc.held, sc.values, order)
        f["held_block"] = {
            "block": f"block {sc.held}",
            "belongs_in": f"slot {t}",
            "which_way_from_where_it_was_lifted": {LEFT: "left", RIGHT: "right", BACK: "same slot"}[direction(sc.held, sc.lifted_from, sc.values, order)],
        }
    else:
        nb = next_pick(sc.row, sc.values, order)
        f["next_block_to_pick_up"] = "none, already in order" if nb is None else f"block {nb}"
    return f


def render_state(sc: Scene, variant: str) -> dict:
    state = {
        "task_before": f"Arrange the numbered blocks on the table in {sc.old} order.",
        "order_definitions": ORDER_DEFS,
        "row": [{"slot": i, "block": "empty (gap)" if b is None else f"block {b}"} for i, b in enumerate(sc.row, 1)],
        "arm": ({"holding": f"block {sc.held}", "lifted_from_slot": sc.lifted_from} if sc.held is not None else {"holding": "nothing"}),
        "user_just_said": sc.text,
    }
    if variant == "facts":
        state["facts_for_current_order"] = _slot_facts(sc, sc.old)
    elif variant == "solved":
        state["facts_if_ascending"] = _slot_facts(sc, ASC)
        state["facts_if_descending"] = _slot_facts(sc, DESC)
    return state


WANTED = "If `user_just_said` does not change the order, the wanted order is the one in `task_before`."


def build_questions(sc: Scene) -> dict:
    qs = {
        "intent": {
            "type": "choice",
            "instructions": {"question": "What is the user's latest message (`user_just_said`) asking the robot to do?", "context": "The robot is partway through `task_before`."},
            "criteria": {
                "stop": "Stop right now, halt all motion (e.g. 'stop!', 'freeze')",
                "pause": "Wait a moment, then carry on with the same task (e.g. 'hold on', 'one sec')",
                "adjust_order": "Keep sorting the same blocks but end in a different order (e.g. 'reverse it', 'biggest first')",
                "continue": "Carry on exactly as before; confirms the current plan (e.g. 'keep going', 'looks good')",
                "adjust_pace": "Same task, different speed (e.g. 'slower', 'faster')",
                "new_task": "Drop the blocks task and do something else",
                "not_for_me": "The message is addressed to someone else, or is not an instruction for the robot",
            },
        },
        "new_order": {
            "type": "choice",
            "instructions": {"question": "After the user's latest message (`user_just_said`), in which order should the blocks end up?", "definitions": ORDER_DEFS},
            "criteria": {
                ASC: ORDER_DEFS[ASC],
                DESC: ORDER_DEFS[DESC],
                "unchanged": "The message does not change the order; keep the order from `task_before`",
            },
        },
        "escalate": {
            "type": "choice",
            "instructions": {
                "question": "Can the robot's next move be decided confidently from this state alone, by direct lookup, without careful multi-step reasoning?",
                "next_move_means": "which way to carry the held block, or which block to pick up next, in the order the user now wants",
            },
            "criteria": {
                "decide_now": "The user's wish is clear and the next move can be read directly off facts listed in the state",
                "ask_a_smarter_model": "The message is ambiguous, or the next move needs ranking, counting or several reasoning steps not already done in the state",
            },
        },
    }
    if not asks_action(sc):
        return qs
    notes = [ORDER_DEFS["slots"], WANTED]
    if sc.held is not None:
        qs["direction"] = {
            "type": "choice",
            "instructions": {
                "question": "The arm is holding a block (`arm.holding`) that it lifted from `arm.lifted_from_slot`. Given the order the user now wants, which way should the arm carry it so it ends up in the slot where it belongs?",
                "notes": notes,
            },
            "criteria": {
                LEFT: "Toward lower slot numbers (the robot's left): the held block belongs in a slot left of the one it was lifted from",
                RIGHT: "Toward higher slot numbers (the robot's right): the held block belongs in a slot right of the one it was lifted from",
                BACK: "The held block belongs in the very slot it was lifted from",
            },
        }
        qs["target_slot"] = {
            "type": "choice",
            "instructions": {
                "question": "Given the order the user now wants, which slot does the held block (`arm.holding`) belong in?",
                "notes": notes + ["Count every block, including the held one: in ascending order the k-th smallest goes in slot k; in descending order the k-th largest goes in slot k."],
            },
            "criteria": {f"slot {k}": f"slot {k}, the {_ordinal(k)} slot from the robot's left" for k in range(1, sc.n + 1)},
        }
    else:
        crit = {f"block {b}": f"block {b}, now in slot {i}" for i, b in enumerate(sc.row, 1)}
        crit = dict(sorted(crit.items(), key=lambda kv: int(kv[0].split()[1])))
        crit[NOTHING] = "Every slot already holds the block that belongs there in the wanted order"
        qs["next_pick"] = {
            "type": "choice",
            "instructions": {
                "question": "The arm's hand is empty. Given the order the user now wants, which block should it pick up next?",
                "rule": "Find the leftmost slot (lowest slot number) whose block is not the one that belongs there in the wanted order; the answer is the block that belongs in that slot.",
                "notes": notes,
            },
            "criteria": crit,
        }
    return qs


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


# ---------------------------------------------------------------- answerers

def mock_answers(sc: Scene, qs: dict, rng: random.Random) -> dict:
    """Noisy oracle in Jev's response shape, for checking the pipeline and tables offline."""
    t = truth(sc)
    out = {}
    for k, q in qs.items():
        opts = list(q["criteria"])
        right = sorted(t.get(k, {opts[0]}))[0]
        p_ok = {"target_slot": 0.6, "next_pick": 0.6}.get(k, 0.85)
        pick = right if rng.random() < p_ok else rng.choice(opts)
        pm = rng.uniform(0.7, 1.0) if pick == right else rng.uniform(1 / len(opts), 0.8)
        probs = {o: (pm if o == pick else (1 - pm) / (len(opts) - 1)) for o in opts}
        n = len(opts)
        out[k] = {"choice": pick, "probabilities": probs, "confidence": (pm - 1 / n) / (1 - 1 / n)}
    return out


def claude_client():
    try:
        import anthropic  # optional: `uv run --extra baseline ...`
    except ImportError:
        raise SystemExit("--baseline claude needs the `anthropic` package: uv run --extra baseline python e11_reorder.py ...")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--baseline claude needs ANTHROPIC_API_KEY")
    return anthropic.Anthropic(max_retries=0, timeout=30.0)


def claude_ask(client, state: dict, qs: dict) -> tuple[dict, float, dict]:
    """One forced tool call answering every question at once (an LLM planner sees them jointly,
    unlike Jev). Neutral keys q1..qk so the key names carry no meaning, as with Jev."""
    keys = {f"q{i}": k for i, k in enumerate(qs, 1)}
    props = {qk: {"type": "string", "enum": list(qs[k]["criteria"])} for qk, k in keys.items()}
    tool = {
        "name": "answer",
        "description": "Answer every question with exactly one of its options.",
        "input_schema": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
    }
    prompt = (
        "You decide a robot arm's next action. Answer each question about the state by picking exactly one option.\n\n"
        f"STATE:\n{json.dumps(state, indent=1)}\n\n"
        f"QUESTIONS:\n{json.dumps({qk: {'instructions': qs[k]['instructions'], 'options': qs[k]['criteria']} for qk, k in keys.items()}, indent=1)}"
    )
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=BASELINE_MODEL, max_tokens=512, tools=[tool], tool_choice={"type": "tool", "name": "answer"},
        messages=[{"role": "user", "content": prompt}],
    )
    ms = (time.perf_counter() - t0) * 1000
    inp = next((b.input for b in msg.content if b.type == "tool_use"), {}) or {}
    answers = {k: {"choice": inp.get(qk)} for qk, k in keys.items()}
    return answers, ms, {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}


# ---------------------------------------------------------------- scoring + tables

def score(sc: Scene, variant: str, answers: dict, latency: float) -> dict:
    t = truth(sc)
    rec = {"sid": sc.sid, "n": sc.n, "variant": variant, "dense": sc.dense, "holding": sc.held is not None, "latency": latency, "q": {}}
    for k, a in answers.items():
        if k == "escalate":
            probs = a.get("probabilities") or {}
            rec["ask_smarter"] = a.get("choice") == "ask_a_smarter_model"
            rec["p_ask"] = probs.get("ask_a_smarter_model")
            continue
        rec["q"][k] = {"ok": a.get("choice") in t[k], "conf": a.get("confidence"), "choice": a.get("choice")}
    rec["any_wrong"] = not all(v["ok"] for v in rec["q"].values())
    return rec


def _frac(ok, tot):
    return f"{ok}/{tot}" if tot else "-"


def print_tables(recs: list[dict], sizes, variants, title: str):
    print(f"\n==================== {title} ====================")
    print("accuracy (correct/total) by question, state variant, row size N")
    for q in QNAMES[:-1]:
        print(f"\n{q}")
        for v in variants:
            cells = []
            for n in list(sizes) + ["all"]:
                rs = [r for r in recs if r["variant"] == v and (n == "all" or r["n"] == n) and q in r["q"]]
                cells.append(f"N={n}: {_frac(sum(r['q'][q]['ok'] for r in rs), len(rs)):>6}")
            print(f"  {v:<7} " + "  ".join(cells))

    print("\ndense (numbers 1..N) vs sparse numbers — the rank questions")
    for q in ("target_slot", "next_pick"):
        for v in variants:
            d = [r for r in recs if r["variant"] == v and q in r["q"] and r["dense"]]
            s = [r for r in recs if r["variant"] == v and q in r["q"] and not r["dense"]]
            print(f"  {q:<12} {v:<7} dense={_frac(sum(r['q'][q]['ok'] for r in d), len(d)):>6}  sparse={_frac(sum(r['q'][q]['ok'] for r in s), len(s)):>6}")

    print("\nlatency per request (ms)")
    for v in variants:
        lat = [r["latency"] for r in recs if r["variant"] == v]
        if lat:
            print(f"  {v:<7} {summarize(lat)}")

    print("\nescalate as a router: does it ask for a smarter model more when this request had a wrong answer?")
    for v in variants:
        rs = [r for r in recs if r["variant"] == v and "ask_smarter" in r]
        ok, bad = [r for r in rs if not r["any_wrong"]], [r for r in rs if r["any_wrong"]]

        def rate(g):
            return f"{100 * sum(r['ask_smarter'] for r in g) / len(g):.0f}%" if g else "-"

        def mean_p(g):
            ps = [r["p_ask"] for r in g if r.get("p_ask") is not None]
            return f"{sum(ps) / len(ps):.2f}" if ps else "-"

        print(f"  {v:<7} all-right n={len(ok):<3} asks {rate(ok):>4} mean P(ask)={mean_p(ok)}   any-wrong n={len(bad):<3} asks {rate(bad):>4} mean P(ask)={mean_p(bad)}")

    if not any(a["conf"] is not None for r in recs for a in r["q"].values()):
        return  # baseline: no confidence to calibrate
    print("\nconfidence gate: accuracy of answers at or above a threshold / coverage (share of answers that clear it)")
    for q in QNAMES[:-1]:
        for v in variants:
            xs = [(r["q"][q]["conf"], r["q"][q]["ok"]) for r in recs if r["variant"] == v and q in r["q"] and r["q"][q]["conf"] is not None]
            if not xs:
                continue
            cells = []
            for th in THRESHOLDS:
                kept = [ok for c, ok in xs if c >= th]
                acc = f"{100 * sum(kept) / len(kept):.0f}%" if kept else "-"
                cells.append(f">={th}: acc {acc:>4} cov {100 * len(kept) / len(xs):.0f}%")
            print(f"  {q:<12} {v:<7} " + " | ".join(cells))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-scenes", type=int, default=20, help="scenes per row size")
    ap.add_argument("--sizes", default="3,5,7,9")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dry-run", action="store_true", help="build every request, write samples, no network")
    ap.add_argument("--mock", action="store_true", help="answer with a noisy oracle (no network) to check the tables")
    ap.add_argument("--baseline", choices=["claude"], help=f"also ask {BASELINE_MODEL} (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    variants = [v for v in args.variants.split(",") if v]
    assert all(v in VARIANTS for v in variants), f"variants must be in {VARIANTS}"
    scs = scenes(sizes, args.n_scenes, args.seed)
    jobs = [(sc, v, render_state(sc, v), build_questions(sc)) for sc in scs for v in variants]

    if args.dry_run:
        out = RESULTS / "e11_samples"
        out.mkdir(parents=True, exist_ok=True)
        want = {"canonical"} | {f"n{n}_01" for n in sizes}
        chars = 0
        for sc, v, state, qs in jobs:
            body = payload(state, qs)
            chars += len(json.dumps(body))
            if sc.sid in want:
                (out / f"{sc.sid}_{v}.json").write_text(json.dumps({"request": body, "truth": {k: sorted(x) for k, x in truth(sc).items()}}, indent=1))
        tok = chars / 3.5  # rough chars-per-token for JSON
        print(f"dry run: {len(scs)} scenes x {len(variants)} variants = {len(jobs)} requests, ~{tok / 1e3:.0f}k input tokens ≈ ${tok * 0.042e-6:.3f}")
        print(f"samples in {out}")
        return

    recs, base_recs, tokens = [], [], 0
    log = None if args.mock else Log("e11_reorder")
    rng = random.Random(args.seed + 1)
    client = None if args.mock else make_client()
    try:
        for i, (sc, v, state, qs) in enumerate(jobs):
            if args.mock:
                answers, lat = mock_answers(sc, qs, rng), rng.uniform(100, 250)
            else:
                r = ask(client, state, qs)
                log.write(model="jev", sid=sc.sid, n=sc.n, variant=v, truth={k: sorted(x) for k, x in truth(sc).items()},
                          request=payload(state, qs), status=r.status, latency_ms=r.latency_ms, upstream_ms=r.upstream_ms, body=r.body)
                if r.status != 200:
                    print("ERR", sc.sid, v, r.status, r.body)
                    continue
                answers, lat = r.body["answers"], r.latency_ms
                tokens += r.input_tokens or 0
            recs.append(score(sc, v, answers, lat))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    finally:
        if client:
            client.close()
    print_tables(recs, sizes, variants, "Jev" + (" (MOCK answers)" if args.mock else ""))
    if tokens:
        print(f"\nJev input tokens: {tokens} ≈ ${tokens * 0.042e-6:.4f}")

    if args.baseline == "claude":
        bc = claude_client()
        btok = [0, 0]
        for sc, v, state, qs in jobs:
            try:
                answers, ms, usage = claude_ask(bc, state, qs)
            except Exception as e:  # no retries: a failed call is logged and skipped
                print("ERR baseline", sc.sid, v, repr(e))
                if log:
                    log.write(model=BASELINE_MODEL, sid=sc.sid, variant=v, error=repr(e))
                continue
            btok[0] += usage["input_tokens"]
            btok[1] += usage["output_tokens"]
            if log:
                log.write(model=BASELINE_MODEL, sid=sc.sid, n=sc.n, variant=v, truth={k: sorted(x) for k, x in truth(sc).items()},
                          latency_ms=ms, usage=usage, answers=answers)
            base_recs.append(score(sc, v, answers, ms))
        print_tables(base_recs, sizes, variants, f"baseline {BASELINE_MODEL}")
        print(f"\nbaseline tokens in/out: {btok[0]}/{btok[1]} ≈ ${btok[0] * 1e-6 + btok[1] * 5e-6:.3f}")
        print("\nlatency p50/p95 (ms): " + "  ".join(
            f"{name}={pct([r['latency'] for r in rs], 50):.0f}/{pct([r['latency'] for r in rs], 95):.0f}"
            for name, rs in (("jev", recs), ("claude", base_recs)) if rs))


if __name__ == "__main__":
    main()
