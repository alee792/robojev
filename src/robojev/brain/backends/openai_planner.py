"""The planner's LLMBackend: OpenAI's Responses API with strict JSON-schema output, ported from
experiments/e12v2/backends.py (openai_backend over E13's OpenAIBackend). Same request (instructions +
input + json_schema format, store false, optional reasoning effort), plus a stable `prompt_cache_key`
so the shared instruction prefix is cached across calls. Raw httpx instead of the SDK: no retries,
one timeout, and no new dependency for the brain."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from . import BudgetExceeded, env_key

URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-6-luna"


@dataclass
class LLMResult:
    """e13's LLMResult shape, which brain/core/planner.PlannerClient reads."""
    raw: str | None
    latency_ms: float
    in_tok: int = 0
    out_tok: int = 0
    error: str | None = None
    model: str | None = None
    reasoning_tok: int = 0
    cached_tok: int = 0
    t_start: float = 0.0


def request_body(req, model: str, max_output_tokens: int | None = None, effort: str | None = None,
                 cache_key: str | None = "robojev-planner") -> dict:
    body = {"model": model, "instructions": req.instructions, "input": req.input,
            "text": {"format": {"type": "json_schema", "name": req.name, "schema": req.schema, "strict": True}},
            "store": False}
    if max_output_tokens:
        body["max_output_tokens"] = max_output_tokens
    if effort:
        body["reasoning"] = {"effort": effort}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    return body


def output_text(body: dict) -> tuple[str | None, str | None]:
    """(text, refusal) from a Responses API body (what the SDK's `output_text` joins)."""
    texts, refusal = [], None
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                texts.append(c.get("text") or "")
            elif c.get("type") == "refusal":
                refusal = c.get("refusal") or "refused"
    return ("".join(texts) or None), refusal


class OpenAIPlanner:
    def __init__(self, model: str = MODEL, effort: str | None = "low", timeout_s: float = 30.0,
                 max_output_tokens: int | None = None, max_calls: int | None = None, cache_key: str = "robojev-planner",
                 client: httpx.Client | None = None, url: str = URL):
        self.model, self.effort, self.max_output_tokens, self.cache_key, self.url = model, effort, max_output_tokens, cache_key, url
        self.max_calls, self.calls = max_calls, 0
        self.client = client or httpx.Client(http2=True, timeout=httpx.Timeout(timeout_s, connect=5.0),
                                             headers={"Authorization": f"Bearer {env_key('OPENAI_API_KEY')}"})

    def call(self, req, ref=None) -> LLMResult:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"{self.max_calls} planner calls reached")
        self.calls += 1
        t_wall, t0 = time.time(), time.perf_counter()
        try:
            resp = self.client.post(self.url, json=request_body(req, self.model, self.max_output_tokens, self.effort, self.cache_key))
        except Exception as e:            # no retries: a failed call is recorded as a failure
            return LLMResult(None, (time.perf_counter() - t0) * 1000, error=repr(e)[:300], t_start=t_wall)
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            return LLMResult(None, ms, error=f"HTTP {resp.status_code}: {resp.text[:200]}", t_start=t_wall)
        if resp.status_code != 200:
            return LLMResult(None, ms, error=f"HTTP {resp.status_code}: {str(body.get('error', body))[:250]}", t_start=t_wall)
        u = body.get("usage") or {}
        raw, refusal = output_text(body)
        err = None
        if body.get("status", "completed") != "completed":
            err = f"status {body.get('status')}: {body.get('incomplete_details')}"
        elif raw is None:
            err = f"refusal: {refusal}" if refusal else "no output text"
        return LLMResult(raw, ms, u.get("input_tokens") or 0, u.get("output_tokens") or 0, err, body.get("model"),
                         (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
                         (u.get("input_tokens_details") or {}).get("cached_tokens") or 0, t_wall)

    def close(self) -> None:
        self.client.close()
