"""Shared helpers: raw HTTP client (no hidden retries), logging, scene generation."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    env = ROOT.parent / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set (env or ../.env)")
    return key


def make_client(**kw) -> httpx.Client:
    return httpx.Client(
        http2=True,
        timeout=httpx.Timeout(5.0),
        headers={"Authorization": f"Bearer {api_key()}"},
        **kw,
    )


def make_async_client(**kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(5.0),
        headers={"Authorization": f"Bearer {api_key()}"},
        limits=httpx.Limits(max_connections=20),
        **kw,
    )


@dataclass
class Call:
    status: int
    latency_ms: float
    body: dict
    request_id: str | None
    upstream_ms: float | None
    input_tokens: int | None


def _call_from(resp: httpx.Response, t0: float) -> Call:
    latency = (time.perf_counter() - t0) * 1000
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}
    up = resp.headers.get("x-envoy-upstream-service-time")
    return Call(
        status=resp.status_code,
        latency_ms=latency,
        body=body,
        request_id=resp.headers.get("x-typesafe-request-id"),
        upstream_ms=float(up) if up else None,
        input_tokens=(body.get("usage") or {}).get("input_tokens") if isinstance(body, dict) else None,
    )


def payload(state, questions) -> dict:
    return {"state": state, "model": MODEL, "questions": questions}


def ask(client: httpx.Client, state, questions) -> Call:
    t0 = time.perf_counter()
    resp = client.post(URL, json=payload(state, questions))
    return _call_from(resp, t0)


async def aask(client: httpx.AsyncClient, state, questions) -> Call:
    t0 = time.perf_counter()
    resp = await client.post(URL, json=payload(state, questions))
    return _call_from(resp, t0)


class Log:
    """Append-only JSONL log per experiment; every request/response is kept for replay."""

    def __init__(self, name: str):
        RESULTS.mkdir(exist_ok=True)
        self.path = RESULTS / f"{name}.jsonl"
        self.f = self.path.open("a")

    def write(self, **rec):
        rec.setdefault("ts", time.time())
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(xs) -> str:
    return f"n={len(xs)} p50={pct(xs,50):.0f} p90={pct(xs,90):.0f} p95={pct(xs,95):.0f} p99={pct(xs,99):.0f} max={max(xs):.0f}ms"


# ---------------------------------------------------------------- tabletop scenes
# Frame convention (placeholder until checked against the WidowX sim): robot base at origin,
# x forward (m), y left (m), z up (m). Bearing measured from +x, positive = left.

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "gray"]
KINDS = ["block", "cup", "bowl", "ball", "can", "sponge", "marker", "box", "banana", "bottle"]


def letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def dist_band(m: float) -> str:
    if m < 0.03:
        return "touching"
    if m < 0.10:
        return "very close"
    if m < 0.25:
        return "near"
    if m < 0.50:
        return "mid-range"
    return "far"


def bearing_words(deg: float) -> str:
    a = abs(deg)
    if a <= 10:
        return "dead ahead"
    side = "left" if deg > 0 else "right"
    if a <= 60:
        return f"ahead {side}"
    if a <= 120:
        return side
    return f"behind {side}"


def random_scene(n_objects: int, rng: random.Random, gripper=None):
    gx, gy, gz = gripper or (rng.uniform(0.2, 0.4), rng.uniform(-0.15, 0.15), rng.uniform(0.08, 0.25))
    objs = []
    used = set()
    for i in range(n_objects):
        while True:
            color, kind = rng.choice(COLORS), rng.choice(KINDS)
            if (color, kind) not in used or len(used) >= len(COLORS) * len(KINDS):
                used.add((color, kind))
                break
        x, y = rng.uniform(0.15, 0.6), rng.uniform(-0.35, 0.35)
        objs.append({"id": f"{color} {kind} {letter(i)}", "color": color, "kind": kind, "xyz": (x, y, 0.02)})
    return {"gripper": (gx, gy, gz), "objects": objs}


def rel(gripper, xyz):
    dx, dy, dz = (xyz[0] - gripper[0], xyz[1] - gripper[1], xyz[2] - gripper[2])
    horiz = math.hypot(dx, dy)
    return {
        "dist": math.sqrt(dx * dx + dy * dy + dz * dz),
        "horiz": horiz,
        "bearing": math.degrees(math.atan2(dy, dx)),
        "height": -dz,
    }


def render(scene, fmt: str) -> dict:
    """fmt: 'numbers' | 'bands' | 'both' — the Doom-style '57 (contact)' is 'both'."""
    g = scene["gripper"]
    items = []
    for o in scene["objects"]:
        r = rel(g, o["xyz"])
        d_num = f"{r['dist']*100:.0f} cm"
        b_num = f"{r['bearing']:+.0f}°"
        if fmt == "numbers":
            d, b = d_num, b_num
        elif fmt == "bands":
            d, b = dist_band(r["dist"]), bearing_words(r["bearing"])
        else:
            d, b = f"{d_num} ({dist_band(r['dist'])})", f"{b_num} ({bearing_words(r['bearing'])})"
        items.append({"label": o["id"], "distance_from_gripper": d, "bearing_from_gripper": b})
    state = {
        "measurement_context": {
            "distance": "straight-line distance from the gripper fingertips to the object's center",
            "bearing": "horizontal direction from the gripper; 0 is straight ahead (away from the robot base), positive is left, negative is right",
        },
        "objects": items,
    }
    if fmt != "numbers":
        state["measurement_context"]["distance_bands"] = {
            "touching": "under 3 cm",
            "very close": "3 to under 10 cm",
            "near": "10 to under 25 cm",
            "mid-range": "25 to under 50 cm",
            "far": "50 cm or more",
        }
    return state


def filler_state(target_tokens: int, rng: random.Random) -> dict:
    """A scene padded with more objects to hit an approximate token budget (~55 tok/object)."""
    n = max(2, target_tokens // 55)
    scene = random_scene(n, rng)
    return render(scene, "both")
