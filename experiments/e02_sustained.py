"""E2: sustained open-loop request rate. Fires on a fixed schedule regardless of latency
(like a control loop), records latency drift, late responses, 429/529s, retry-after.

usage: e02_sustained.py [hz=10] [seconds=120] [state_tokens=3000]
Cost at defaults: 1200 calls x ~3k tok ≈ 3.6M tok ≈ $0.15.
"""

import asyncio
import random
import sys
from collections import Counter

from common import Log, aask, filler_state, make_async_client, pct, summarize

HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 10
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 120
TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
log = Log(f"e02_sustained_{HZ:g}hz")

QS = {
    "gripper": {"type": "choice", "instructions": "Should the gripper be open or closed right now?", "criteria": {"open": None, "closed": None}},
    "target": {"type": "choice", "instructions": "Which side of the table has more objects?", "criteria": {"left": None, "right": None}},
    "move": {"type": "choice", "instructions": "How should the arm move right now?", "criteria": {"approach": None, "hold": None, "back_off": None}},
    "contact": {"type": "noul", "instructions": "Is any object touching the gripper?"},
}


async def main():
    rng = random.Random(2)
    states = [filler_state(TOK, rng) for _ in range(20)]
    lat, statuses, retry_after = [], Counter(), []
    period = 1 / HZ
    n = int(SECS * HZ)
    async with make_async_client() as c:
        await aask(c, "warmup", {"w": {"type": "noul", "instructions": "warmup?"}})
        loop = asyncio.get_running_loop()
        t_start = loop.time()
        tasks = []

        async def one(i):
            try:
                r = await aask(c, states[i % len(states)], QS)
            except Exception as e:  # timeouts etc.
                statuses[type(e).__name__] += 1
                log.write(i=i, error=repr(e))
                return
            statuses[r.status] += 1
            log.write(i=i, status=r.status, latency_ms=r.latency_ms, upstream_ms=r.upstream_ms, input_tokens=r.input_tokens)
            if r.status == 200:
                lat.append((i, r.latency_ms))
            else:
                retry_after.append(r.body)

        for i in range(n):
            target = t_start + i * period
            await asyncio.sleep(max(0, target - loop.time()))
            tasks.append(asyncio.create_task(one(i)))
        await asyncio.gather(*tasks)

    ls = [l for _, l in lat]
    print(f"{HZ:g} Hz for {SECS:g}s, ~{TOK} tok state, {len(QS)} questions")
    print("statuses:", dict(statuses))
    if ls:
        print("latency:", summarize(ls))
        late = sum(l > period * 1000 for l in ls)
        print(f"responses slower than one period ({period*1000:.0f} ms): {late}/{len(ls)} = {100*late/len(ls):.1f}%")
        q = len(ls) // 4 or 1
        first, last = [l for i, l in lat[:q]], [l for i, l in lat[-q:]]
        print(f"drift: first-quarter p50={pct(first,50):.0f}ms last-quarter p50={pct(last,50):.0f}ms")
    if retry_after:
        print("non-200 bodies (first 3):", retry_after[:3])


asyncio.run(main())
