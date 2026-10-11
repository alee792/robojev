"""OpenAI's Decisions API (POST /v1/decisions, gpt-6-luna) as the brain's DecisionBackend.

It takes the request core/decision.py builds for Jev ({state, questions}) and sends it as ONE
Decisions request: the API takes an array of named questions, so the three groups (right_now,
in_plan_fix, route) and any progress / plan-check Nouls go together, as they do to Jev.

Translation (Jev -> Decisions):
  state                       -> input: one user message, an input_text part with the state as JSON
                                 (the questions refer to its keys, e.g. `plan_position.arm_in_step`),
                                 plus input_image parts (inline base64 data URLs) when images are given
  choice {instructions, criteria{value: {what, not_for}}}
                              -> choice {name, instructions, choices[{value, description}]}; `not_for`
                                 is folded into the description ("Not for: ...")
  noul {instructions, criteria{true, false}}
                              -> predicate {name, instructions}; the true/false criteria are folded in
Answers come back in the shape combine() reads from Jev:
  choice    -> {"type": "choice", "choice", "probabilities": {value: p}, "confidence", "confidence_api",
                "confidence_formula"}
  predicate -> {"type": "noul", "noul": probability}
  refusal   -> the key is left out (combine reads a missing answer as confidence 0: the cautious
               right-now choice and an LLM route); listed in meta["refusals"].

Confidence. Jev's choice confidence is (p_max - 1/n)/(1 - 1/n) over the offered options (robojev.jev
.choice_confidence); the gates in core/combine.py were set on that. The Decisions API returns its own
`confidence` field next to the distribution and does not document how it is computed. Both are kept
(`confidence_api`, `confidence_formula`); `confidence` (what combine reads) is the one picked by
`confidence=` ("formula" by default, so the 0.8 gate means what it meant for Jev; see
docs/live-results.md for whether the two ever differ).
"""
from __future__ import annotations

import base64
import json
import time

import httpx

from . import BudgetExceeded, env_key

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"


def choice_confidence(probs: dict) -> float:
    """(p_max - 1/n)/(1 - 1/n), Jev's formula (robojev.jev.choice_confidence, repeated so the brain
    doesn't import the v1 client)."""
    n = len(probs)
    if n < 2:
        return 0.0
    return max(0.0, (max(probs.values()) - 1 / n) / (1 - 1 / n))


def _instructions(ins) -> str:
    if isinstance(ins, str):
        return ins
    parts = [ins.get("question", "")]
    if ins.get("context"):
        parts.append(f"Context: {ins['context']}")
    if ins.get("ask_about"):
        parts.append(f"Judge {ins['ask_about']}.")
    parts += [f"{k}: {v}" for k, v in ins.items() if k not in ("question", "context", "ask_about")]
    return "\n".join(p for p in parts if p)


def _description(c) -> str:
    if isinstance(c, str):
        return c
    d = c.get("what", "")
    if c.get("not_for"):
        d += f" Not for: {c['not_for']}."
    return d


def to_questions(questions: dict) -> list[dict]:
    """Jev's questions dict -> the Decisions API's questions array (order kept)."""
    out = []
    for name, q in questions.items():
        if q["type"] == "choice":
            out.append({"type": "choice", "name": name, "instructions": _instructions(q.get("instructions", "")),
                        "choices": [{"value": v, "description": _description(c)} for v, c in q["criteria"].items()]})
        elif q["type"] == "noul":
            ins = _instructions(q.get("instructions", ""))
            crit = q.get("criteria") or {}
            if crit.get("true"):
                ins += f"\nTrue means: {crit['true']}"
            if crit.get("false"):
                ins += f"\nFalse means: {crit['false']}"
            out.append({"type": "predicate", "name": name, "instructions": ins})
        else:
            raise ValueError(f"question {name}: type {q['type']!r} has no Decisions API equivalent here")
    return out


def image_part(image) -> dict:
    """bytes (PNG) | (mime, bytes) | a data: URL -> an input_image part (inline base64 only)."""
    if isinstance(image, str):
        if not image.startswith("data:"):
            raise ValueError("the Decisions API takes images as inline base64 data URLs only")
        url = image
    else:
        mime, data = image if isinstance(image, tuple) else ("image/png", image)
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return {"type": "input_image", "image_url": url}


def to_input(state, images=()) -> list[dict]:
    text = state if isinstance(state, str) else "The robot's situation (JSON):\n" + json.dumps(state, ensure_ascii=False)
    return [{"role": "user", "content": [{"type": "input_text", "text": text}, *(image_part(i) for i in images)]}]


def from_answers(answers: list, confidence: str = "formula") -> tuple[dict, list]:
    """The Decisions API's answers array -> (Jev-shaped answers dict, refused question names)."""
    out, refused = {}, []
    for a in answers:
        name, t = a.get("name"), a.get("type")
        if t == "choice":
            probs = {p["value"]: float(p["probability"]) for p in a.get("probabilities") or []}
            f = choice_confidence(probs)
            api = a.get("confidence")
            api = float(api) if api is not None else None
            out[name] = {"type": "choice", "choice": a.get("choice"), "probabilities": probs,
                         "confidence": f if confidence == "formula" or api is None else api,
                         "confidence_api": api, "confidence_formula": f}
        elif t == "predicate":
            out[name] = {"type": "noul", "noul": float(a.get("probability", 0.0))}
        else:                                      # "refusal" (or anything new): not an answer
            refused.append(name)
    return out, refused


class DecisionsAPI:
    live = True

    def __init__(self, model: str = MODEL, timeout_s: float = 5.0, confidence: str = "formula",
                 max_requests: int | None = None, client: httpx.Client | None = None, url: str = URL):
        if confidence not in ("formula", "api"):
            raise ValueError("confidence is 'formula' or 'api'")
        self.model, self.url, self.confidence = model, url, confidence
        self.client = client or httpx.Client(http2=True, timeout=httpx.Timeout(timeout_s, connect=3.0),
                                             headers={"Authorization": f"Bearer {env_key('OPENAI_API_KEY')}"})
        self.max_requests, self.n, self.errors = max_requests, 0, 0

    def warm(self) -> None:
        """Open the TLS/HTTP2 connection before an episode (the first call measured ~1 s)."""
        try:
            self.client.post(self.url, json={"model": self.model, "input": "x",
                                             "questions": [{"type": "predicate", "name": "q", "instructions": "Is this x?"}]})
        except Exception:
            pass

    def answer(self, state, questions: dict, ref=None, images=()) -> dict:
        if self.max_requests is not None and self.n >= self.max_requests:
            raise BudgetExceeded(f"{self.max_requests} Decisions API requests reached")
        self.n += 1
        body = {"model": self.model, "input": to_input(state, images), "questions": to_questions(questions)}
        t0 = time.perf_counter()
        try:
            resp = self.client.post(self.url, json=body)
        except Exception as e:        # timeout / connection: no retries; combine() treats it as low confidence
            self.errors += 1
            return {"answers": None, "latency_ms": (time.perf_counter() - t0) * 1000, "in_tok": 0,
                    "meta": {"backend": "decisions", "error": repr(e)[:200]}}
        ms = (time.perf_counter() - t0) * 1000
        try:
            rb = resp.json()
        except Exception:
            rb = {"raw": resp.text[:500]}
        proc = resp.headers.get("openai-processing-ms")
        meta = {"backend": "decisions", "status": resp.status_code, "request_id": resp.headers.get("x-request-id"),
                "upstream_ms": float(proc) if proc else None, "model": rb.get("model") if isinstance(rb, dict) else None,
                "images": len(images)}
        ok = resp.status_code == 200 and isinstance(rb, dict) and isinstance(rb.get("answers"), list)
        if not ok:
            self.errors += 1
            meta["body"] = rb
            return {"answers": None, "latency_ms": ms, "in_tok": 0, "meta": meta}
        answers, refused = from_answers(rb["answers"], self.confidence)
        if refused:
            meta["refusals"] = refused
        return {"answers": answers, "latency_ms": ms, "in_tok": (rb.get("usage") or {}).get("input_tokens") or 0, "meta": meta}

    def close(self) -> None:
        self.client.close()
