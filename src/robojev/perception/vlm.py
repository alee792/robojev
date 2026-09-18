"""Slow-tier naming and confirmation of tracks with a vision-language model.

Off the control path: a worker thread names each confirmed track once (and again if its
appearance changes a lot) from a colour crop, and says whether it is a real object at all.
The fast loop keeps running on shape labels until a name arrives; phantoms it flags get dropped.
Backends: `ClaudeNamer` (Anthropic SDK, needs ANTHROPIC_API_KEY or an `ant auth login` profile)
and `StubNamer` (rules, for tests and offline runs).
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_object", "name", "kind", "confidence", "reason"],
    "properties": {
        "is_object": {"type": "boolean", "description": "true if the crop shows a real, separate physical object on the table"},
        "name": {"type": "string", "description": "two to four plain words a person would use, e.g. 'white paper cup', 'black phone', 'human hand'"},
        "kind": {"type": "string", "enum": ["cup", "mug", "bottle", "can", "phone", "book", "box", "block", "ball", "hand", "arm", "tool", "cable", "other"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "description": "one short sentence"},
    },
}

PROMPT = ("This is a crop from a depth camera's colour image looking at a tabletop where a small robot arm works. "
          "The crop is centred on something the depth sensor segmented as a separate object about {h:.0f} cm tall and {w:.0f} cm wide "
          "(shape guess: {kind}). Say whether it is a real separate object (not a fragment of the table, a shadow, part of the robot's "
          "gripper or a merged pair of things), and name it the way a person would. Answer with the JSON only.")


@dataclass
class Naming:
    is_object: bool
    name: str
    kind: str
    confidence: float
    reason: str
    t: float


class StubNamer:
    """Deterministic rules for tests/offline: cup-like -> 'cup', flat -> 'phone', hand-sized low box -> 'hand'."""

    def name(self, crop_bgr: np.ndarray, h: float, w: float, kind_guess: str, color: str) -> Naming:
        if kind_guess in ("cup-like object", "upright object"):
            return Naming(True, f"{color} paper cup", "cup", 0.8, "stub", time.time())
        if kind_guess == "flat object":
            return Naming(True, f"{color} phone", "phone", 0.7, "stub", time.time())
        if 0.03 <= h <= 0.06 and w >= 0.08:
            return Naming(True, "human hand", "hand", 0.6, "stub", time.time())
        return Naming(True, f"{color} {kind_guess}", "other", 0.4, "stub", time.time())


class ClaudeNamer:
    def __init__(self, model: str = "claude-opus-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def name(self, crop_bgr: np.ndarray, h: float, w: float, kind_guess: str, color: str) -> Naming:
        ok, jpg = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = base64.standard_b64encode(jpg.tobytes()).decode()
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=512,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT.format(h=h * 100, w=w * 100, kind=kind_guess)},
            ]}],
        )
        if resp.stop_reason == "refusal":
            return Naming(True, f"{color} {kind_guess}", "other", 0.0, "refused", time.time())
        text = next(b.text for b in resp.content if b.type == "text")
        o = json.loads(text)
        return Naming(bool(o["is_object"]), o["name"], o["kind"], float(o["confidence"]), o["reason"], time.time())


class NamingWorker(threading.Thread):
    """Names tracks in the background. `submit(entity_id, crop, h, w, kind, color)`; results land in
    `results[entity_id]` for the perception thread to apply."""

    def __init__(self, namer):
        super().__init__(daemon=True, name="vlm")
        self.namer = namer
        self.q: queue.Queue = queue.Queue()
        self.results: dict[str, Naming] = {}
        self.pending: set[str] = set()
        self.errors = 0
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()

    def submit(self, eid: str, crop: np.ndarray, h: float, w: float, kind: str, color: str) -> None:
        with self.lock:
            if eid in self.pending or eid in self.results:
                return
            self.pending.add(eid)
        self.q.put((eid, crop, h, w, kind, color))

    def run(self):
        while not self.stop_evt.is_set():
            try:
                eid, crop, h, w, kind, color = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                res = self.namer.name(crop, h, w, kind, color)
            except Exception as e:  # never let the slow tier take the loop down
                self.errors += 1
                res = None
            with self.lock:
                self.pending.discard(eid)
                if res is not None:
                    self.results[eid] = res

    def take(self) -> dict[str, Naming]:
        with self.lock:
            out, self.results = self.results, {}
            return out


def crop_around(image: np.ndarray, pixel: tuple[int, int], size: int = 160) -> np.ndarray:
    u, v = pixel
    h, w = image.shape[:2]
    x0, y0 = max(0, u - size // 2), max(0, v - size // 2)
    return image[y0:min(h, y0 + size), x0:min(w, x0 + size)].copy()
