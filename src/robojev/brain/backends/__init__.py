"""Live model backends for the brain (the mocks are in brain/mocks.py and brain/eval/).

    jev.py              DecisionBackend: Jev (typesafe /v1/systemone) over raw HTTP/2
    decisions_api.py    DecisionBackend: OpenAI's Decisions API (/v1/decisions), the same questions
    openai_planner.py   LLMBackend: OpenAI Responses API, strict JSON-schema output, prompt_cache_key

All three speak HTTP with httpx directly (no SDK, no hidden retries, short timeouts), like
robojev/jev.py, so the brain's default install needs nothing new. None imports the oracle or a robot.
"""
from __future__ import annotations

import os
from pathlib import Path


class BudgetExceeded(Exception):
    """A backend's request cap was reached (live runs are capped so a loop can't run up a bill)."""


def env_key(name: str) -> str:
    """An API key from the environment, else from a .env in the working directory or the repo root.
    Never logged or put in an exception message."""
    key = os.environ.get(name)
    if key:
        return key
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[4] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not set (env or .env)")
