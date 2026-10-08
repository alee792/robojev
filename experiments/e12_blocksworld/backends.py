"""Backends: who answers a layer's Questions.

  oracle  answers every question from ground truth (upper bound; the scripted baseline)
  mock    the oracle with an error rate and confidence noise (exercises gating and recovery offline)
  jev     the real API: raw httpx over HTTP/2 via experiments/common.py, no retries
  claude  claude-haiku-4-5 answering the same questions in one forced tool call (e11's baseline)

`answer()` returns (answers in Jev's response shape or None on a failed call, latency ms, input tokens, meta).
"""

from __future__ import annotations

import random


def _shape(pick: str, opts: list, conf: float) -> dict:
    n = len(opts)
    pm = conf * (1 - 1 / n) + 1 / n if n > 1 else 1.0          # invert (p_max - 1/n) / (1 - 1/n)
    rest = (1 - pm) / (n - 1) if n > 1 else 0.0
    return {"choice": pick, "probabilities": {o: round(pm if o == pick else rest, 2) for o in opts}, "confidence": round(conf, 3)}


def _preferred(truth: list, opts: list) -> str:
    return next((t for t in truth if t in opts), opts[0])


class Oracle:
    name = "oracle"
    live = False

    def close(self):
        pass

    def answer(self, layer, state, questions, truth):
        return {k: _shape(_preferred(truth[k], list(q["criteria"])), list(q["criteria"]), 1.0) for k, q in questions.items()}, 0.0, 0, {}


class Mock(Oracle):
    """Right answers get confidence 1 - |N(0, conf_noise)| clipped to [0.5, 1]; wrong ones (rate `error`)
    get a uniform confidence in [0, 0.7], so a gate at 0.3-0.5 catches some of them and not others."""
    name = "mock"

    def __init__(self, error: float = 0.1, conf_noise: float = 0.1, seed: int = 0):
        self.error, self.conf_noise, self.rng = error, conf_noise, random.Random(seed)

    def answer(self, layer, state, questions, truth):
        out = {}
        for k, q in questions.items():
            opts = list(q["criteria"])
            right = _preferred(truth[k], opts)
            wrong = [o for o in opts if o not in truth[k]]
            if wrong and self.rng.random() < self.error:
                out[k] = _shape(self.rng.choice(wrong), opts, self.rng.uniform(0.0, 0.7))
            else:
                out[k] = _shape(right, opts, min(1.0, max(0.5, 1 - abs(self.rng.gauss(0, self.conf_noise)))))
        return out, 0.0, 0, {}


class Jev:
    name = "jev"
    live = True

    def __init__(self):
        from common import make_client  # needs TYPESAFE_API_KEY
        self.client = make_client()
        self.errors = 0

    def answer(self, layer, state, questions, truth):
        from common import ask
        try:
            r = ask(self.client, state, questions)
        except Exception as e:  # timeout / connection: no retries, the layer re-asks on the next tick
            self.errors += 1
            return None, 0.0, 0, {"error": repr(e)}
        if r.status != 200:
            self.errors += 1
            return None, r.latency_ms, 0, {"status": r.status, "body": r.body}
        return r.body["answers"], r.latency_ms, r.input_tokens or 0, {"upstream_ms": r.upstream_ms, "request_id": r.request_id}

    def close(self):
        self.client.close()


class Claude:
    """Forced-choice baseline, reusing e11's client and single tool call (neutral keys q1..qk). No
    confidence comes back, so its picks are treated as confidence 1 (never gated)."""
    name = "claude"
    live = True

    def __init__(self):
        import os
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise SystemExit("--backend claude needs the `anthropic` package: uv run --frozen --extra vlm python experiments/e12_blocksworld.py ... "
                             "(or from experiments/: uv run --extra baseline python e12_blocksworld.py ...)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--backend claude needs ANTHROPIC_API_KEY")
        from e11_reorder import claude_client
        self.client = claude_client()
        self.errors = 0
        self.out_tokens = 0

    def answer(self, layer, state, questions, truth):
        from e11_reorder import claude_ask
        try:
            answers, ms, usage = claude_ask(self.client, state, questions)
        except Exception as e:
            self.errors += 1
            return None, 0.0, 0, {"error": repr(e)}
        self.out_tokens += usage["output_tokens"]
        for a in answers.values():
            a["confidence"] = 1.0
        return answers, ms, usage["input_tokens"], {}

    def close(self):
        pass


def make(name: str, error: float = 0.1, conf_noise: float = 0.1, seed: int = 0):
    if name == "oracle":
        return Oracle()
    if name == "mock":
        return Mock(error, conf_noise, seed)
    if name == "jev":
        return Jev()
    if name == "claude":
        return Claude()
    raise ValueError(name)
