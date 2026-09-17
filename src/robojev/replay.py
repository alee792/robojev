"""Offline replay: re-ask recorded states with a (possibly new) question set and diff the picks
against what was recorded. `--questions vN` rebuilds questions from the recorded world facts is not
possible without the World object, so replay rebuilds them from the recorded *state* by loading the
tick's questions when the set is unchanged, or the new module's build() from a World reconstructed
from the tick's obs. Minimal v0: same-set replay (determinism / drift) and new-set replay via obs."""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from robojev.config import DEFAULT
from robojev.jev import JevClient
from robojev.log import read_jsonl


async def replay(run_dir: str, questions: str | None, limit: int | None, concurrency: int):
    run = Path(run_dir)
    cfg = DEFAULT
    ticks = list(read_jsonl(run / "ticks.jsonl"))
    if limit:
        ticks = ticks[:limit]
    recorded = {}
    for a in read_jsonl(run / "answers.jsonl"):
        if a.get("outcome") == "applied" or a.get("answers"):
            recorded[a["tick"]] = a.get("answers", {})
    qmod = None
    if questions:
        from robojev.questions import load
        qmod = load(questions)
    client = JevClient(cfg.loop.model, timeout_s=5.0)
    sem = asyncio.Semaphore(concurrency)
    results = {}

    async def one(t):
        qs = t["questions"] if qmod is None else _rebuild(qmod, cfg, t)
        async with sem:
            r = await client.ask(t["tick"], t["state"], qs)
        results[t["tick"]] = r

    await asyncio.gather(*(one(t) for t in ticks))
    await client.close()
    agree = defaultdict(Counter)
    for tick, r in sorted(results.items()):
        if not r.ok:
            agree["_errors"]["n"] += 1
            continue
        rec = recorded.get(tick)
        for k, a in r.answers.items():
            new = _pick(a)
            old = _pick(rec[k]) if rec and k in rec else None
            agree[k]["same" if old == new else ("no_record" if old is None else "different")] += 1
    print(json.dumps({k: dict(v) for k, v in agree.items()}, indent=1))
    return results


def _pick(a):
    if a.get("type") == "choice":
        return a["choice"]
    if a.get("type") == "noul":
        return "yes" if a["noul"] >= 0.5 else "no"
    return round(a.get("score", 0))


def _rebuild(qmod, cfg, t):
    from robojev.arm import ArmSnapshot
    from robojev.perception.memory import Entity
    from robojev.world import build_world
    import numpy as np
    o = t["obs"]
    ents = [Entity(e["id"], np.array(e["xyz"]), e["h"], e["w"], e["color"], e["last_seen"], e["last_seen"], seen_count=5)
            for e in o["entities"]]
    snap = ArmSnapshot(t["t"], tuple(o["ee"]), o["gripper"], frozen=o.get("frozen", False), status=o.get("status", "live"),
                       setpoint=tuple(o["ee"]))
    st = t["state"]
    orders = [] if st["standing_orders"] == ["(none)"] else st["standing_orders"]
    task = "" if st["user_request"]["text"] == "(none yet)" else st["user_request"]["text"]
    w = build_world(cfg, snap, ents, lambda e, now: True, o["table_z"], task, orders, t.get("brain", {}), t["t"])
    return qmod.build(cfg, w)
