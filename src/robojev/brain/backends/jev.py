"""Jev as the brain's DecisionBackend: one POST per decision, ported from experiments/e12v2/backends.py
(JevHTTP). Raw HTTP/2, no retries, a short timeout; a timeout or a non-200 comes back as
`answers: None`, which combine() treats as low confidence everywhere (cautious right-now, LLM route)."""
from __future__ import annotations

import os
import time

import httpx

from . import BudgetExceeded, env_key

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")


class JevHTTP:
    live = True

    def __init__(self, model: str = MODEL, timeout_s: float = 5.0, max_requests: int | None = None,
                 client: httpx.Client | None = None, url: str = URL):
        self.model, self.url = model, url
        self.client = client or httpx.Client(http2=True, timeout=httpx.Timeout(timeout_s),
                                             headers={"Authorization": f"Bearer {env_key('TYPESAFE_API_KEY')}"})
        self.max_requests, self.n, self.errors = max_requests, 0, 0

    def warm(self) -> None:
        """Open the TLS/HTTP2 connection before an episode (a new connection costs ~65 ms)."""
        try:
            self.client.post(self.url, json={"state": "x", "model": self.model,
                                             "questions": {"q": {"type": "noul", "instructions": "Is this x?"}}})
        except Exception:
            pass

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        if self.max_requests is not None and self.n >= self.max_requests:
            raise BudgetExceeded(f"{self.max_requests} Jev requests reached")
        self.n += 1
        t0 = time.perf_counter()
        try:
            resp = self.client.post(self.url, json={"state": state, "model": self.model, "questions": questions})
        except Exception as e:
            self.errors += 1
            return {"answers": None, "latency_ms": (time.perf_counter() - t0) * 1000, "in_tok": 0,
                    "meta": {"backend": "jev", "error": repr(e)[:200]}}
        ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        ok = resp.status_code == 200 and isinstance(body, dict) and isinstance(body.get("answers"), dict)
        if not ok:
            self.errors += 1
        up = resp.headers.get("x-envoy-upstream-service-time")
        meta = {"backend": "jev", "status": resp.status_code, "upstream_ms": float(up) if up else None,
                "request_id": resp.headers.get("x-typesafe-request-id"), "model": body.get("model") if isinstance(body, dict) else None}
        if not ok:
            meta["body"] = body
        in_tok = ((body.get("usage") or {}).get("input_tokens") or 0) if isinstance(body, dict) else 0
        return {"answers": body["answers"] if ok else None, "latency_ms": ms, "in_tok": in_tok, "meta": meta}

    def close(self) -> None:
        self.client.close()
