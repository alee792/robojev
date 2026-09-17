"""Raw async Jev client: one persistent HTTP/2 connection, no retries, short timeout.

The SDK's default retry policy can stall a control loop for 30 s (09-sim-and-sdk-notes), so this
speaks HTTP directly like experiments/common.py. Every call is tagged so late, stale and
out-of-order answers can be recognised by the caller.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

URL = "https://api.typesafe.ai/v1/systemone"


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    for env in (Path.cwd() / ".env", Path.home() / "code/robojev/.env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("TYPESAFE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TYPESAFE_API_KEY not set (env or .env)")


@dataclass
class JevResult:
    tag: int
    ok: bool
    status: int | None
    latency_ms: float
    answers: dict = field(default_factory=dict)
    error: str | None = None
    request_id: str | None = None
    upstream_ms: float | None = None
    input_tokens: int | None = None
    model: str | None = None


class JevClient:
    def __init__(self, model: str, timeout_s: float = 0.7, max_connections: int = 8):
        self.model = model
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(timeout_s, connect=2.0),
            headers={"Authorization": f"Bearer {api_key()}"},
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def warm(self) -> None:
        """Open the TLS/HTTP2 connection before the loop starts (a new connection costs ~65 ms)."""
        try:
            await self._client.post(URL, json={"state": "x", "model": self.model,
                                               "questions": {"q": {"type": "noul", "instructions": "Is this x?"}}})
        except Exception:
            pass

    async def ask(self, tag: int, state, questions: dict) -> JevResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(URL, json={"state": state, "model": self.model, "questions": questions})
        except httpx.TimeoutException as e:
            return JevResult(tag, False, None, (time.perf_counter() - t0) * 1000, error=f"timeout: {e.__class__.__name__}")
        except Exception as e:
            return JevResult(tag, False, None, (time.perf_counter() - t0) * 1000, error=f"{e.__class__.__name__}: {e}"[:200])
        latency = (time.perf_counter() - t0) * 1000
        up = resp.headers.get("x-envoy-upstream-service-time")
        rid = resp.headers.get("x-typesafe-request-id")
        if resp.status_code != 200:
            return JevResult(tag, False, resp.status_code, latency, error=resp.text[:200], request_id=rid,
                             upstream_ms=float(up) if up else None)
        body = resp.json()
        return JevResult(tag, True, 200, latency, answers=body.get("answers", {}), request_id=rid,
                         upstream_ms=float(up) if up else None,
                         input_tokens=(body.get("usage") or {}).get("input_tokens"), model=body.get("model"))

    async def close(self) -> None:
        await self._client.aclose()


def choice_confidence(probabilities: dict) -> float:
    """(p_max - 1/n)/(1 - 1/n), the formula confirmed in 08 §E4. Useful offline and for adapters."""
    n = len(probabilities)
    if n < 2:
        return 0.0
    return (max(probabilities.values()) - 1 / n) / (1 - 1 / n)
