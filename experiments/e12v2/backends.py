"""Live backends, reusing the repo's clients: Jev over raw HTTP/2 (experiments/common.py: 5 s timeout,
no retries) and the OpenAI Responses API with strict JSON-schema output (E13's OpenAIBackend, plus a
stable `prompt_cache_key` so the shared instruction prefix is cached across calls)."""

from __future__ import annotations

import time


class BudgetExceeded(Exception):
    pass


class JevHTTP:
    live = True

    def __init__(self, max_requests: int | None = None):
        from common import make_client   # needs TYPESAFE_API_KEY (env or ../.env)
        self.client = make_client()
        self.max_requests, self.n, self.errors = max_requests, 0, 0

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        from common import ask
        if self.max_requests is not None and self.n >= self.max_requests:
            raise BudgetExceeded(f"--max-jev-requests {self.max_requests} reached")
        self.n += 1
        try:
            r = ask(self.client, state, questions)
        except Exception as e:           # timeout / connection: no retries; the combiner treats it as low confidence
            self.errors += 1
            return {"answers": None, "latency_ms": 5000.0, "in_tok": 0, "meta": {"error": repr(e)[:200]}}
        ok = r.status == 200 and isinstance(r.body, dict)
        if not ok:
            self.errors += 1
        return {"answers": r.body.get("answers") if ok else None, "latency_ms": r.latency_ms, "in_tok": r.input_tokens or 0,
                "meta": {"status": r.status, "upstream_ms": r.upstream_ms, "request_id": r.request_id,
                         **({} if ok else {"body": r.body})}}

    def close(self):
        self.client.close()


def openai_backend(model: str, timeout_s: float = 30.0, effort: str | None = None, max_output_tokens: int | None = None,
                   max_calls: int | None = None):
    from e13_branches.planner import LLMResult, OpenAIBackend, openai_kwargs

    class OpenAIPlanner(OpenAIBackend):
        """E13's backend with prompt_cache_key; same no-retry, short-timeout behaviour."""
        calls = 0

        def call(self, req, ref=None):
            if max_calls is not None and OpenAIPlanner.calls >= max_calls:
                raise BudgetExceeded(f"--max-llm-calls {max_calls} reached")
            OpenAIPlanner.calls += 1
            kw = openai_kwargs(req, self.model, self.max_output_tokens, self.effort)
            kw["prompt_cache_key"] = "e12v2-planner"
            t_wall, t0 = time.time(), time.perf_counter()
            try:
                resp = self.client.responses.create(**kw)
            except Exception as e:
                return LLMResult(None, (time.perf_counter() - t0) * 1000, error=repr(e)[:300], t_start=t_wall)
            ms = (time.perf_counter() - t0) * 1000
            u = resp.usage
            err = None
            if getattr(resp, "status", "completed") != "completed":
                err = f"status {resp.status}: {getattr(resp, 'incomplete_details', None)}"
            raw = resp.output_text or None
            if raw is None and err is None:
                err = "no output text (refusal?)"
            rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
            ct = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
            return LLMResult(raw, ms, u.input_tokens if u else 0, u.output_tokens if u else 0, err, getattr(resp, "model", None), rt, ct, t_wall)

    return OpenAIPlanner(model, timeout_s=timeout_s, max_output_tokens=max_output_tokens, effort=effort)
