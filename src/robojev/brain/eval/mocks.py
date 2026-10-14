"""Offline backends. Both read the evaluation oracle, so they are stand-ins, not the system.

MockJev  answers every question in Jev's response shape from oracle.truth(), with an error rate and
         confidence noise (as E12/E13's mocks): a right answer gets confidence 1 - |N(0, conf_noise)|,
         a wrong one a uniform confidence in [0, 0.7]. error=0, conf_noise=0 is the `oracle` arm.
         Latency is sampled (lognormal, median 150 ms, p95 ~260 ms: E12's measured Jev latency).
MockLLM  returns oracle-derived plans for the task version the request's user messages imply, from
         the true world: full plans for "plan", diffs (core.plan.make_diff) for "replan"/"react".
         Optional error injection: leave an object out, swap two destinations, or return an invalid
         plan (the validator catches it; the retry fixes it with probability `retry_fixes`).
"""

from __future__ import annotations

import json
import math
import random

from ..core import plan as P
from . import oracle as O


def _shape_choice(pick: str, opts: list, conf: float) -> dict:
    n = len(opts)
    pm = conf * (1 - 1 / n) + 1 / n if n > 1 else 1.0
    rest = (1 - pm) / (n - 1) if n > 1 else 0.0
    return {"choice": pick, "probabilities": {o: round(pm if o == pick else rest, 3) for o in opts}, "confidence": round(conf, 3)}


class MockJev:
    live = False

    def __init__(self, scenario, world, error: float = 0.05, conf_noise: float = 0.1, seed: int = 0, name: str = "mock"):
        self.sc, self.world = scenario, world
        self.error, self.conf_noise = error, conf_noise
        self.rng = random.Random(seed)
        self.name = name

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        ev, view = ref
        T = O.truth(ev, view, self.sc, self.world)
        out = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                opts = list(q["criteria"])
                right = next((t for t in T.get(k, []) if t in opts), opts[0])
                wrong = [o for o in opts if o not in T.get(k, [])]
                if wrong and self.error and self.rng.random() < self.error:
                    out[k] = _shape_choice(self.rng.choice(wrong), opts, self.rng.uniform(0.0, 0.7))
                else:
                    c = 1.0 if not self.conf_noise else min(1.0, max(0.5, 1 - abs(self.rng.gauss(0, self.conf_noise))))
                    out[k] = _shape_choice(right, opts, c)
            else:
                yes = bool(T.get(k, False))
                p = 0.97 if yes else 0.03
                if self.conf_noise:
                    p = self.rng.uniform(0.9, 0.99) if yes else self.rng.uniform(0.01, 0.1)
                if self.error and self.rng.random() < self.error:
                    p = self.rng.uniform(0.3, 0.7)
                out[k] = {"noul": round(p, 3)}
        lat = 0.0 if self.name == "oracle" else math.exp(self.rng.gauss(math.log(150), 0.33))
        tok = int(len(json.dumps({"state": state, "questions": questions})) / 3.5)
        return {"answers": out, "latency_ms": lat if self.name != "oracle" else 150.0, "in_tok": tok, "meta": {"mock": self.name}}

    def close(self):
        pass


class _Res:
    def __init__(self, raw, latency_ms, in_tok, out_tok, error=None, model="mock"):
        self.raw, self.latency_ms, self.in_tok, self.out_tok, self.error, self.model = raw, latency_ms, in_tok, out_tok, error, model
        self.cached_tok = self.reasoning_tok = 0


class MockLLM:
    """`median_ms`: 2700 for the fast model, 5000 for the capable one (E13's measured medians)."""

    def __init__(self, scenario, world, seed: int = 0, error: float = 0.0, retry_fixes: float = 0.8, median_ms: float = 2700.0):
        self.sc, self.world = scenario, world
        self.rng = random.Random(seed)
        self.error, self.retry_fixes, self.median_ms = error, retry_fixes, median_ms

    def _goal(self, req):
        body = json.loads(req.input.split("\n\nYour previous answer was:")[0])
        return self.sc.goal_for(body.get("user_messages", [])), body

    def call(self, req, ref=None):
        ctx, attempt = ref if ref else (None, 0)
        ep = ctx["episode"] if ctx else None
        goal, body = self._goal(req)
        ts = self.world.truth()
        holding = ts.arm.holding
        said = body.get("user_messages", [])
        reading = body.get("task", "") + (" As corrected: " + "; ".join(said) if said else "")
        plan = O.oracle_plan(goal, ts, holding, reading=reading)
        err = self.rng.random() < self.error if attempt == 0 else False
        kind = self.rng.choice(["omit", "swap", "invalid"]) if err else None
        if attempt > 0 and self.rng.random() > self.retry_fixes:
            kind = "invalid"
        plan = self._inject(plan, kind)
        if req.kind == "plan":
            out = plan
        else:
            diff = P.make_diff(ep.plan, ep.queue, plan, reading=plan["reading"])
            if req.kind == "react":
                ev = ctx["event"]
                T = O.truth(ev, ep.view(ev), self.sc, self.world)
                same = [(s["skill"], s["object"], s["target"], s["direction"]) for s in plan["steps"]] == \
                       [(s["skill"], s["object"], s["target"], s["direction"]) for s in (P.steps_by_id(ep.plan)[i] for i in ep.queue)]
                same = same and not diff["drop_done_when"] and not diff["new_done_when"] and not diff["drop_constraints"] and not diff["new_constraints"]
                diff = {"right_now": T["right_now"][0], "change_plan": not same, **diff}
            out = diff
        raw = json.dumps(out)
        lat = math.exp(self.rng.gauss(math.log(self.median_ms), 0.35))
        n_in = (len(req.instructions) + len(req.input) + len(json.dumps(req.schema))) // 4
        return _Res(raw, lat, n_in, len(raw) // 4)

    def _inject(self, plan: dict, kind: str | None) -> dict:
        if kind is None:
            return plan
        moves = [s for s in plan["steps"] if P.step_goal(s)]
        if kind == "omit" and len(moves) >= 2:
            s = self.rng.choice(moves)
            plan["steps"] = [x for x in plan["steps"] if x is not s]
            plan["done_when"] = [c for c in plan["done_when"] if c["object"] != s["object"]]
        elif kind == "swap" and len(moves) >= 2:
            a, b = self.rng.sample(moves, 2)
            if a["skill"] == b["skill"] == "move_object":
                a["target"], b["target"] = b["target"], a["target"]
                for c in plan["done_when"]:
                    if c["object"] == a["object"]:
                        c["target"], c["text"] = a["target"], f"{a['object']} is in {a['target']}"
                    elif c["object"] == b["object"]:
                        c["target"], c["text"] = b["target"], f"{b['object']} is in {b['target']}"
        elif kind == "invalid" and moves:
            plan["done_when"] = []
        return plan
