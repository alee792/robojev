"""E13 system under test, step 2: the Jev Router picks among the branches the LLM prepared.

State: the task, the correction, and the plan's parameters with each value described by its meaning
and the current value (never the goal maps: Jev does not compute positions). Questions, all in one
request and answered independently:
  route          Choice: continue / adjust / new_plan_needed / pause, with `not_for` boundaries
  p<i>.value     Choice over parameter i's values (function-calling style; no "unchanged" option)
  p<i>.stated    Noul: does the message say anything about parameter i?
Code then takes the new value of each parameter whose `stated` is yes, looks the combination up
among the prepared branches, and gates on the weakest confidence among the questions that action
depends on. Anything else escalates to the LLM. Must not import the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import plan as P

ROUTES = ("continue", "adjust", "new_plan_needed", "pause")
PAUSE_GATE = 0.3        # pausing is the cautious action: its own, lower threshold (v2 "Routing")
STATED = 0.5            # P(yes) above which a parameter counts as mentioned

ROUTE_CRITERIA = {
    "continue": {
        "what": "Carry on with the plan exactly as it is: the message is praise, chatter, a remark to someone else, or confirms what the robot is already doing.",
        "not_for": "any request to change where blocks end up, however small",
        "examples": ["great", "perfect, thanks"],
    },
    "adjust": {
        "what": "The user wants a different end result that setting one or more of `plan.parameters` to one of its listed values fully expresses.",
        "not_for": "a change the listed values cannot express, even partly (use new_plan_needed)",
        "examples": ["asking for another listed value of a setting"],
    },
    "new_plan_needed": {
        "what": "The user wants a different end result that no combination of the listed parameter values expresses: e.g. leaving some blocks out, a different container for some blocks, a different rule, or a new task.",
        "not_for": "a change fully covered by the listed parameter values (use adjust)",
        "examples": ["a request about blocks or places that no parameter mentions"],
    },
    "pause": {
        "what": "The user asks the robot to wait or hold on for a moment before carrying on.",
        "not_for": "a request to change the task",
        "examples": ["one moment"],
    },
}


def jev_state(task_text: str, correction: str, plan: dict, current: dict) -> dict:
    return {
        "task": task_text,
        "user_just_said": correction,
        "plan": {
            "robot_is_doing": plan.get("task_reading", ""),
            "parameters": [
                {"name": p["name"], "controls": p["about"], "current_value": current[p["name"]],
                 "values": {v["value"]: v["meaning"] for v in p["values"]}}
                for p in plan["parameters"]
            ],
        },
    }


def jev_questions(plan: dict) -> dict:
    qs = {
        "route": {
            "type": "choice",
            "instructions": {
                "question": "What does the user's latest message (`user_just_said`) need from the robot, which is partway through `task` following `plan`?",
                "note": "`plan.parameters` lists every setting the robot can change on its own, with the values it has prepared.",
            },
            "criteria": ROUTE_CRITERIA,
        }
    }
    for i, p in enumerate(plan["parameters"]):
        path = f"`plan.parameters[{i}]`"
        qs[f"p{i}.value"] = {
            "type": "choice",
            "instructions": {
                "question": f"After the user's latest message (`user_just_said`), which value should the setting {path} ({p['about']}) have?",
                "if_not_mentioned": f"keep its current value, `plan.parameters[{i}].current_value`",
            },
            "criteria": {v["value"]: v["meaning"] for v in p["values"]},
        }
        qs[f"p{i}.stated"] = {
            "type": "noul",
            "instructions": f"Does the user's latest message (`user_just_said`) ask for, or state, a particular value of {path}: {p['about']}?",
            "criteria": {
                "true": "The message names or clearly implies a value for this setting, including asking for the opposite, the other one, or a switch.",
                "false": "The message says nothing about this setting.",
            },
        }
    return qs


def noul_conf(p: float) -> float:
    """Noul returns P(yes) only. Confidence of the side taken, on Choice's scale with n=2: |2p - 1|."""
    return abs(2 * p - 1)


@dataclass
class Decision:
    route: str | None
    action: str            # "keep" (current branch) | "branch" (switch) | "escalate"
    key: str | None        # combination key of the branch kept or switched to
    reason: str
    min_conf: float | None


def decide(plan: dict, current: dict, answers: dict | None, gate: float, pause_gate: float = PAUSE_GATE) -> Decision:
    """Pure function of the plan (goals not needed), the current settings and Jev's answers."""
    cur_key = P.combo_key(plan, current)
    if not answers or "route" not in answers:
        return Decision(None, "escalate", None, "jev_error", None)
    route = answers["route"].get("choice")
    rconf = answers["route"].get("confidence") or 0.0
    if route == "new_plan_needed":
        return Decision(route, "escalate", None, "route_new_plan", rconf)
    if route == "pause":
        ok = rconf >= min(gate, pause_gate)
        return Decision(route, "keep" if ok else "escalate", cur_key if ok else None, "pause" if ok else "low_confidence", rconf)
    confs = [rconf]
    new = dict(current)
    stated = 0
    for i, p in enumerate(plan["parameters"]):
        s = answers.get(f"p{i}.stated", {}).get("noul")
        v = answers.get(f"p{i}.value", {})
        if s is None or not v:
            return Decision(route, "escalate", None, "jev_error", None)
        confs.append(noul_conf(s))
        if s >= STATED:
            stated += 1
            new[p["name"]] = v.get("choice")
            confs.append(v.get("confidence") or 0.0)
    mc = min(confs)
    key = P.combo_key(plan, new)
    if mc < gate:
        return Decision(route, "escalate", None, "low_confidence", mc)
    if route == "continue":
        if key != cur_key:
            return Decision(route, "escalate", None, "continue_but_value_changed", mc)
        return Decision(route, "keep", cur_key, "continue", mc)
    # adjust
    if key == cur_key:
        if stated:
            return Decision(route, "keep", cur_key, "adjust_to_current", mc)
        return Decision(route, "escalate", None, "adjust_without_change", mc)
    if key not in P.branch_keys(plan):
        return Decision(route, "escalate", None, "missing_branch", mc)
    return Decision(route, "branch", key, "adjust", mc)


class JevBackend:
    """Raw HTTP/2 to Jev via experiments/common.py (5 s timeout, no retries)."""

    def __init__(self):
        from common import make_client
        self.client = make_client()

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        from common import ask
        r = ask(self.client, state, questions)
        return {"status": r.status, "latency_ms": r.latency_ms, "upstream_ms": r.upstream_ms, "in_tok": r.input_tokens or 0,
                "answers": r.body.get("answers") if r.status == 200 and isinstance(r.body, dict) else None,
                "body": r.body if r.status != 200 else None}

    def close(self):
        self.client.close()
