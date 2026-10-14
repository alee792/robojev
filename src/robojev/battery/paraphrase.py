"""The --paraphrase hook: every line the person says, rewritten by a model before the episode runs,
so a live decider meets wording nobody wrote down. Offline (no `--paraphrase`) nothing is rewritten:
`none()` returns each line as it is. The model call needs OPENAI_API_KEY and runs on the Mac.

A rewrite must keep the WORD (in capitals) in a new_word line and say the same kind of thing; the
generator keeps the original when a rewrite loses the word (generate._paraphrased).
"""
from __future__ import annotations

import json

import httpx

from robojev.brain.backends import env_key
from robojev.brain.backends.openai_planner import MODEL, URL, output_text

from .generate import Paraphrase

INSTRUCTIONS = ("Rewrite the line a person says to a robot that is spelling a word with letter blocks, keeping exactly its "
                "meaning and its kind ({kind}), in different everyday words. If the line contains a word in CAPITALS, keep "
                "that word, in capitals, unchanged, and capitalise nothing else. Answer with the rewritten line only.")


def none(text: str, kind: str) -> str:
    """The offline hook: no rewriting."""
    return text


class OpenAIParaphraser:
    """Rewrites through the Responses API (plain text output). `calls` counts requests."""

    def __init__(self, model: str = MODEL, timeout_s: float = 20.0, client: httpx.Client | None = None, url: str = URL):
        self.model, self.url = model, url
        self.client = client or httpx.Client(http2=True, timeout=httpx.Timeout(timeout_s, connect=5.0),
                                             headers={"Authorization": f"Bearer {env_key('OPENAI_API_KEY')}"})
        self.calls = 0
        self.rewrites: list[tuple[str, str, str]] = []

    def __call__(self, text: str, kind: str) -> str:
        self.calls += 1
        body = {"model": self.model, "instructions": INSTRUCTIONS.format(kind=kind), "input": text, "store": False}
        try:
            resp = self.client.post(self.url, json=body)
            resp.raise_for_status()
            out, refusal = output_text(resp.json())
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            return text
        new = (out or "").strip().strip('"')
        if refusal or not new:
            return text
        self.rewrites.append((kind, text, new))
        return new


def hook(paraphrase: bool) -> Paraphrase:
    return OpenAIParaphraser() if paraphrase else none
