"""E1 + E3: latency vs question count and state size; warm vs cold connections.

Cost: ~ (4 q-counts x 3 sizes x N) calls; at N=15 and avg ~8k tok ≈ 1.5M tok ≈ $0.06.
"""

import random
import sys

from common import Log, ask, filler_state, make_client, summarize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rng = random.Random(1)
log = Log("e01_latency")


def questions(n):
    qs = {}
    for i in range(n):
        qs[f"q{i}"] = {
            "type": "choice",
            "instructions": f"Which object in `objects` is the best candidate #{i} for the gripper to approach next?",
            "criteria": {"nearest": "The closest object", "leftmost": "The object furthest to the left", "rightmost": "The object furthest to the right"},
        }
    return qs


with make_client() as c:
    ask(c, "warmup", {"w": {"type": "noul", "instructions": "Is this a warmup?"}})
    for size in (1000, 6000, 20000):
        state = filler_state(size, rng)
        for nq in (1, 6, 12, 40):
            lat, toks, ups = [], None, []
            for _ in range(N):
                r = ask(c, state, questions(nq))
                log.write(exp="grid", size=size, nq=nq, status=r.status, latency_ms=r.latency_ms, upstream_ms=r.upstream_ms, input_tokens=r.input_tokens, request_id=r.request_id, err=None if r.status == 200 else r.body)
                if r.status == 200:
                    lat.append(r.latency_ms)
                    toks = r.input_tokens
                    if r.upstream_ms:
                        ups.append(r.upstream_ms)
                else:
                    print("ERR", r.status, r.body)
            if lat:
                print(f"state~{size:>5} tok(actual {toks}) q={nq:>2}: {summarize(lat)}  upstream p50={sorted(ups)[len(ups)//2] if ups else 'n/a'}")

# cold: new connection each call
state = filler_state(6000, rng)
lat = []
for _ in range(N):
    with make_client() as c:
        r = ask(c, state, questions(6))
    log.write(exp="cold", status=r.status, latency_ms=r.latency_ms, input_tokens=r.input_tokens)
    if r.status == 200:
        lat.append(r.latency_ms)
print(f"COLD (new TLS conn per call) ~6k tok q=6: {summarize(lat)}")
