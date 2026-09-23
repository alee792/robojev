"""E13 offline backends. Both read the evaluation oracle, so they are stand-ins, not the system.

MockLLM  builds plans from `oracle.natural_parameter` (one parameter, both branches), with optional
         error injection: wrong default branch, missing alternative branch, invalid plan. Escalations
         return the oracle's corrected arrangement, with optional errors (unchanged plan, invalid).
MockJev  answers in Jev's response shape from the scoring truth, with noise as in E11's mock.
"""

from __future__ import annotations

import json
import math
import random

from . import oracle
from . import plan as P
from .planner import LLMRequest, LLMResult


def _goal_list(goal: dict) -> list[dict]:
    return [{"block": b, "destination": d} for b, d in goal.items()]


def _break(goal: dict, rng: random.Random) -> dict:
    """A plausible wrong goal: swap two blocks' destinations (or send one to the table)."""
    g = dict(goal)
    ks = [k for k in g]
    a, b = rng.sample(ks, 2)
    if g[a] != g[b]:
        g[a], g[b] = g[b], g[a]
    else:
        g[a] = "stays_on_table" if g[a] != "stays_on_table" else "left_bin"
    return g


class MockLLM:
    def __init__(self, seed: int = 0, wrong_default: float = 0.0, missing_branch: float = 0.0, invalid: float = 0.0,
                 esc_wrong: float = 0.0, retry_fixes: float = 0.8):
        self.rng = random.Random(seed)
        self.wrong_default, self.missing_branch, self.invalid = wrong_default, missing_branch, invalid
        self.esc_wrong, self.retry_fixes = esc_wrong, retry_fixes

    def _plan_for(self, task) -> dict:
        np_ = oracle.natural_parameter(task)
        base = oracle.task_rule(task)
        (dv, dm), (ov, om) = np_["values"]
        default = oracle.target_for_rule(base, task.scene).canonical
        other = oracle.target_for_rule(oracle.apply(base, np_["op"]), task.scene).canonical
        if self.rng.random() < self.wrong_default:
            default = _break(default, self.rng)
        branches = [{"settings": [{"parameter": np_["name"], "value": dv}], "goal": _goal_list(default)},
                    {"settings": [{"parameter": np_["name"], "value": ov}], "goal": _goal_list(other)}]
        if self.rng.random() < self.missing_branch:
            branches = branches[:1]
        return {"task_reading": task.text, "parameters": [{"name": np_["name"], "about": np_["about"], "default": dv,
                                                           "values": [{"value": dv, "meaning": dm}, {"value": ov, "meaning": om}]}],
                "branches": branches}

    def _escalation_for(self, task, correction, current_goal: dict) -> dict:
        goal = oracle.target(task, correction).canonical
        if self.rng.random() < self.esc_wrong:
            goal = current_goal if goal != current_goal else _break(goal, self.rng)
        return {"task_reading": f"{task.text} Then: {correction.text}", "parameters": [], "branches": [{"settings": [], "goal": _goal_list(goal)}]}

    def call(self, req: LLMRequest, ref=None) -> LLMResult:
        (ctx, attempt) = ref
        if req.kind == "plan":
            plan = self._plan_for(ctx["task"])
        else:
            plan = self._escalation_for(ctx["task"], ctx["correction"], ctx["current_goal"])
        p_invalid = self.invalid if attempt == 0 else self.invalid * (1 - self.retry_fixes)
        if self.rng.random() < p_invalid:
            g = plan["branches"][0]["goal"]
            g[1]["destination"] = g[0]["destination"] = "tray_slot_1"   # two blocks in one slot
            g.pop()                                                     # and one block missing
        raw = json.dumps(plan)
        lat = math.exp(self.rng.gauss(math.log(1500 if req.kind == "plan" else 1800), 0.35))
        return LLMResult(raw, lat, (len(req.instructions) + len(req.input) + len(json.dumps(req.schema))) // 4, len(raw) // 4, None, "mock")


class MockJev:
    """Noisy oracle answers. `ref` = {"truth": score.truth(...), "plan": plan, "current": settings}."""

    def __init__(self, seed: int = 0, error: float = 0.1):
        self.rng = random.Random(seed)
        self.error = error

    def _choice(self, opts: list[str], right: str) -> dict:
        n = len(opts)
        pick = right if self.rng.random() >= self.error else self.rng.choice(opts)
        lo = 1 - min(0.25, 2.5 * self.error)      # E11's mock spread at error 0.1; certain at error 0
        pm = self.rng.uniform(lo, 1.0) if pick == right else self.rng.uniform(1 / n, 0.8)
        probs = {o: (pm if o == pick else (1 - pm) / (n - 1)) for o in opts}
        return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": (pm - 1 / n) / (1 - 1 / n)}

    def _noul(self, yes: bool) -> dict:
        p = self.rng.uniform(0.9, 0.99) if yes else self.rng.uniform(0.01, 0.1)
        if self.rng.random() < self.error:
            p = self.rng.uniform(0.3, 0.7)
        return {"type": "noul", "noul": p}

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        t, plan, cur = ref["truth"], ref["plan"], ref["current"]
        routes = list(questions["route"]["criteria"])
        right_route = "continue" if "continue" in t["route"] else t["route"][0]
        target = cur
        if t.get("target_key"):
            target = next(P.settings_of(b) for b in plan["branches"] if P.combo_key(plan, P.settings_of(b)) == t["target_key"])
        ans = {"route": self._choice(routes, right_route)}
        for i, p in enumerate(plan["parameters"]):
            ans[f"p{i}.value"] = self._choice([v["value"] for v in p["values"]], target[p["name"]])
            ans[f"p{i}.stated"] = self._noul(target[p["name"]] != cur[p["name"]])
        return {"status": 200, "latency_ms": self.rng.uniform(100, 250), "upstream_ms": None, "in_tok": 0, "answers": ans, "body": None}

    def close(self):
        pass
