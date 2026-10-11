"""Code combines the three answers (docs/v2.md "The decision", "Confidence decides escalation").

1. The right-now answer applies at once. Below the right-now gate it falls back to the cautious
   choice (table CAUTIOUS). While an LLM replans, it decides whether the step carries on or holds.
2. The in-plan fix applies only if the Router says stay_local and the weakest confidence among the
   route and the fix clears the stay_local gate.
3. Otherwise the fast or capable LLM replans. Each route has its own threshold, set by what a wrong
   answer costs: a low-confidence route moves one rung up the ladder
   (stay_local -> fast_llm -> capable_llm; ask_user -> capable_llm).
Code rules on top: a plan-check "yes" (or an unsure one) sends the plan back to the LLM; a failed
step or an unmet plan with no in-plan fix cannot stay local; a Jev error is treated as low confidence
everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .decision import build_questions, build_state
from .data import Combined, Event

# Low-confidence right-now answers fall back to the cautious choice. `resume` falls back to carry_on,
# i.e. stay paused/holding (the paused-idle event re-asks later); `back_off` falls back to pause
# (no motion around a hand on a guess).
CAUTIOUS = {"carry_on": "hold", "re_target": "hold", "resume": "carry_on", "hold": "hold", "pause": "pause", "back_off": "pause"}


@dataclass
class Gates:
    right_now: float = 0.5
    stay_local: float = 0.8      # the in-plan fix / keeping it local (e12v2: 0.8 caught 98% of wrong answers, 0.7 89%)
    fast_llm: float = 0.3        # below: capable LLM
    capable_llm: float = 0.0
    ask_user: float = 0.5        # below: capable LLM instead of bothering the user
    check: float = 0.5           # plan check / progress Nouls, on |2p - 1|

    def as_dict(self):
        return asdict(self)


@dataclass
class Ablation:
    right_now: bool = True
    in_plan_fix: bool = True
    router: bool = True

    @property
    def name(self) -> str:
        off = [n for n, on in (("right-now", self.right_now), ("in-plan-fix", self.in_plan_fix), ("router", self.router)) if not on]
        return "jev" + "".join(f"-no-{n}" for n in off)


def noul_conf(p: float) -> float:
    return abs(2 * p - 1)


def _ch(answers: dict, key: str) -> tuple[str | None, float]:
    a = (answers or {}).get(key) or {}
    return a.get("choice"), float(a.get("confidence") or 0.0)


def combine(event: Event, answers: dict | None, view, gates: Gates, abl: Ablation = Ablation()) -> Combined:
    hand = view.hand is not None
    if answers is None:
        return Combined("pause" if hand else "hold", "none", "fast_llm", "jev_error", conf={})
    conf = {}
    rn, c_rn = _ch(answers, "right_now") if abl.right_now else ("carry_on", 1.0)
    rn = rn or "carry_on"
    rn_raw = rn
    conf["right_now"] = c_rn
    if c_rn < gates.right_now:
        rn = CAUTIOUS.get(rn, "hold")
        if rn == "hold" and hand:
            rn = "pause"
    # not asked (ablated, or "none" was the only option code offered) = none, certain
    fix, c_fix = _ch(answers, "in_plan_fix") if abl.in_plan_fix and "in_plan_fix" in answers else ("none", 1.0)
    fix = fix or "none"
    conf["in_plan_fix"] = c_fix
    problems, reason = [], ""
    for k, a in answers.items():
        if k.startswith("check."):
            p = float(a.get("noul", 0.0))
            conf[k] = noul_conf(p)
            if p >= 0.5:
                problems.append(f"{view.name(k[6:])} is something the task asks about but the plan leaves out")
            elif noul_conf(p) < gates.check:
                problems.append(f"{view.name(k[6:])} may be something the task asks about that the plan leaves out (unsure)")
    done = {}
    for k, a in answers.items():
        if k.startswith("done."):
            p = float(a.get("noul", 0.0))
            conf[k] = noul_conf(p)
            done[k[5:]] = p >= 0.5 and noul_conf(p) >= gates.check
    if abl.router:
        route, c_route = _ch(answers, "route")
        route = route or "fast_llm"
    else:
        routine = event.kind == "step_done" or (event.kind == "plan_arrived" and not problems)
        route, c_route = ("stay_local" if fix != "none" or routine else "fast_llm"), 1.0
    conf["route"] = c_route
    route_raw = route
    if problems and route not in ("fast_llm", "capable_llm"):
        route, reason = "fast_llm", "plan check"
    if route == "stay_local":
        w = min(c_route, c_fix)
        if w < gates.stay_local:
            route, reason = "fast_llm", "low confidence"
        elif fix == "none" and event.kind in ("step_failed", "plan_done_unmet"):
            route, reason = "fast_llm", "nothing local fixes it"
    elif route == "fast_llm" and c_route < gates.fast_llm:
        route, reason = "capable_llm", "low confidence"
    elif route == "ask_user" and c_route < gates.ask_user:
        route, reason = "capable_llm", "low confidence"
    applied_fix = fix if route == "stay_local" else "none"
    return Combined(rn, applied_fix, route, reason or route, right_now_raw=rn_raw, route_raw=route_raw, fix_raw=fix,
                    conf=conf, problems=problems, done=done)


class JevDecider:
    """Decision = one request (three groups + progress/plan-check Nouls) + combine()."""

    def __init__(self, backend, gates: Gates | None = None, ablation: Ablation | None = None):
        self.backend, self.gates, self.abl = backend, gates or Gates(), ablation or Ablation()
        self.name = self.abl.name

    def decide(self, event: Event, view) -> dict:
        groups = [g for g, on in (("right_now", self.abl.right_now), ("in_plan_fix", self.abl.in_plan_fix), ("route", self.abl.router)) if on]
        state = build_state(event, view)
        qs = build_questions(event, view, groups)
        r = self.backend.answer(state, qs, ref=(event, view))
        answers = r.get("answers")
        comb = combine(event, answers, view, self.gates, self.abl)
        return {"combined": comb, "latency_ms": r.get("latency_ms", 0.0), "request": {"state": state, "questions": qs},
                "answers": answers, "in_tok": r.get("in_tok", 0), "meta": r.get("meta", {}), "jev": True}
