"""The three Jev layers as request builders. Each renders its own filtered State and its own Questions
from the Plan's templates, and returns the ground-truth answers (acceptable options, preferred first)
computed from the TRUE world, which the oracle backend answers with and every backend is scored
against. Every question is a Choice, so every Pick carries a confidence.

  Sequencer  next skill (code-offered options), is the task complete, escalate
  Listener   intent, one choice per knob (values + unchanged), when, escalate   [only on an Utterance]
  Spotter    action (continue / slow / hold / back_off / evade_up), whether each conditional Order is on
"""

from __future__ import annotations

import math

from . import plan as P
from .skills import offered
from .world import dxy, seg_dist

from common import dist_band  # experiments/common.py (the entry script puts experiments/ on sys.path)


def dist_text(cm: float) -> str:
    return f"{cm:.0f} cm ({dist_band(cm / 100)})"


INTENTS = {
    "adjust": "Change a setting of the current task: one of the knobs in `knobs` (e.g. the order, the speed, the bin, or setting aside the block the arm is going for)",
    "stop": "Stop the task now and stay stopped",
    "pause": "Wait for a moment, then carry on with the same task (e.g. 'hold on', 'one sec')",
    "resume": "Carry on after a pause (e.g. 'ok, go on')",
    "new_task": "Drop this task and do something different that the knobs cannot express",
    "continue": "Carry on exactly as before; confirms or praises what the arm is doing (e.g. 'keep going', 'looks good')",
    "not_for_me": "Not addressed to the robot, or not an instruction at all",
}
WHEN = {
    "now": "right away, even in the middle of the current move",
    "at_next_safe_point": "after the current step finishes",
}
ACTIONS = {
    "continue": "carry on with the current skill at the current pace",
    "slow": "carry on, at half speed",
    "hold": "stop moving and stay put until this is re-assessed",
    "back_off": "move the gripper about 10 cm away from the hand and up, then stay put",
    "evade_up": "lift the gripper straight up about 13 cm, fast, then stay put",
}
ESCALATE = {
    "decide_now": "yes: the answer can be read directly off the state and the options offered",
    "ask_a_smarter_model": "no: it is unclear, contradictory, refers to something not in the state, or needs several reasoning steps not already done in the state",
}


# ---------------------------------------------------------------- shared rendering

def orders_view(rt) -> list[dict]:
    out = []
    for o in rt.plan["orders"]:
        if o["kind"] == "always":
            st = "always on"
        else:
            st = "on now" if o["id"] in rt.limiter.active else f"off (applies only while {o['condition']})"
        out.append({"order": o["text"], "status": st})
    return out


def status_view(rt) -> dict:
    arm = rt.world.arm
    return {
        "holding": rt.name(arm.holding) if arm.holding else "nothing",
        "gripper": f"at ({arm.x:.0f}, {arm.y:.0f}) cm, " + ("up at travel height" if arm.high else "down low, among the blocks"),
        "current_skill": rt.skill_text(arm.skill) if arm.skill else "none (waiting for the next skill)",
        "last_result": arm.last_result,
    }


def knobs_view(rt) -> dict:
    out = {}
    for k, spec in rt.plan["knobs"].items():
        out[k] = {"controls": spec["describe"], "now": rt.knobs.get(k, "n/a (one-off)"), "values": spec["values"]}
    return out


def scene_view(rt, scene) -> dict:
    blocks = [{"name": e["name"], "colour": e["colour"], "where": rt.where_text(e["where"], e)} for e in scene["blocks"].values()]
    v = {"blocks": blocks}
    if scene["slots"]:
        v["tray"] = {"layout": "slot 1 is the robot's left end; slot numbers increase to the right",
                     "slots": [{"slot": k, "has": rt.name(s["has"]) if s["has"] else "empty"} for k, s in scene["slots"].items()]}
    if scene["bins"]:
        v["bins"] = [f"{b} bin" for b in scene["bins"]]
    return v


def _above_what(rt, scene) -> str:
    arm = rt.world.arm
    if not arm.high:
        return "nothing: the gripper is low"
    for e in scene["blocks"].values():
        if e["where"] != "gripper" and P.above(arm, (e["x"], e["y"])):
            return e["name"]
    if arm.dest and P.above(arm, arm.dest[1:]):
        return rt.dest_name(arm.dest[0])
    return "nothing in particular"


def branch_facts(rt, scene, knobs: dict) -> dict:
    """Code's answer under one set of knob values."""
    arm = rt.world.arm
    g = P.goals(rt.plan, knobs, scene)
    left = P.todo(rt.plan, knobs, scene)
    f = {
        "goal_of_each_block": {scene["blocks"][b]["name"]: rt.dest_name(d) for b, d in g.items()},
        "blocks_not_at_their_goal": [scene["blocks"][b]["name"] for b in left] or "none",
    }
    if arm.holding and arm.holding in g:
        free = P.goal_free(scene, g[arm.holding], arm.holding)
        f["held_block_goes_to"] = rt.dest_name(g[arm.holding]) + ("" if free else " (taken: set it aside on the table first)")
    f["next_step"] = rt.skill_text(P.next_skill(rt.plan, knobs, scene, arm, rt.deferred))
    f["task_complete"] = P.complete(rt.plan, knobs, scene, arm)
    return f


def facts(rt, scene, variant: str) -> dict:
    f = {"gripper_is_above": _above_what(rt, scene)}
    if scene["slots"]:
        f["empty_slots"] = [k for k, s in scene["slots"].items() if s["has"] is None] or "none"
    if rt.deferred:
        f["set_aside_until_last"] = [rt.name(b) for b in sorted(rt.deferred)]
    if variant != "solved":
        return f
    gk = next((k for k in P.GOAL_KNOBS if k in rt.knobs), None)
    if gk is None:
        f["with_the_current_knobs"] = branch_facts(rt, scene, rt.knobs)
        return f
    for v in rt.plan["knobs"][gk]["values"]:
        tag = " (current)" if rt.knobs[gk] == v else ""
        f[f"if {gk} is {v}{tag}"] = branch_facts(rt, scene, {**rt.knobs, gk: v})
    return f


def _q(instructions, criteria) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


# ---------------------------------------------------------------- Sequencer

def sequencer(rt, variant: str) -> tuple[dict, dict, dict]:
    T = rt.plan["question_templates"]["sequencer"]
    scene = rt.scene
    state = {
        "orders": orders_view(rt),
        "task": P.task_text(rt.plan, rt.knobs),
        "status": status_view(rt),
        "knobs": {k: v for k, v in rt.knobs.items()},
        "scene": scene_view(rt, scene),
        "facts": facts(rt, scene, variant),
    }
    opts = offered(rt)
    qs = {
        "next_skill": _q(T["next_skill"], opts),
        "task_status": _q(T["task_status"], {
            "complete": "every block in `task` is at its goal for the current knob values, the gripper holds nothing and is up",
            "not_complete": "some block is not at its goal yet, or the arm still holds a block or is low",
        }),
        "escalate": _q(T["escalate"], ESCALATE),
    }
    ts = rt.truth_scene
    ns = P.next_skill(rt.plan, rt.knobs, ts, rt.world.arm, rt.deferred)
    truth = {
        "next_skill": [ns if ns in opts else "hold"],
        "task_status": ["complete" if P.complete(rt.plan, rt.knobs, ts, rt.world.arm) else "not_complete"],
        "escalate": ["decide_now"],
    }
    return state, qs, truth


# ---------------------------------------------------------------- Listener

def listener(rt, variant: str, utt) -> tuple[dict, dict, dict]:
    T = rt.plan["question_templates"]["listener"]
    scene = rt.scene
    state = {
        "task": P.task_text(rt.plan, rt.knobs),
        "status": {k: v for k, v in status_view(rt).items() if k in ("holding", "current_skill")},
        "knobs": knobs_view(rt),
    }
    if variant == "solved":
        gk = next((k for k in P.GOAL_KNOBS if k in rt.knobs), None)
        if gk:
            state["facts_under_each_knob_value"] = {
                f"{gk} = {v}": {k: x for k, x in branch_facts(rt, scene, {**rt.knobs, gk: v}).items() if k != "goal_of_each_block"}
                for v in rt.plan["knobs"][gk]["values"]}
        if rt.world.arm.target and rt.world.arm.target not in rt.deferred:
            state["facts_if_skip_current_target"] = {"would_set_aside": rt.name(rt.world.arm.target)}
    state["utterance"] = {"text": utt.text, "said_while": state["status"]["current_skill"]}
    qs = {"intent": _q(T["intent"], INTENTS)}
    truth = {"intent": list(utt.intent)}
    for k, spec in rt.plan["knobs"].items():
        now = rt.knobs.get(k, "none")
        ins = {"question": T["knob"]["question"].format(knob=k, describe=spec["describe"], now=now)}
        crit = {v: d for v, d in spec["values"].items()}
        crit[P.UNCHANGED] = "the message does not ask to change this"
        qs[f"knob:{k}"] = _q(ins, crit)
        truth[f"knob:{k}"] = list(utt.knobs.get(k, [P.UNCHANGED]))
    qs["when"] = _q(T["when"], WHEN)
    qs["escalate"] = _q(T["escalate"], ESCALATE)
    truth["when"] = list(utt.when)
    truth["escalate"] = list(utt.escalate)
    return state, qs, truth


# ---------------------------------------------------------------- Spotter

def heading(rt, scene):
    arm = rt.world.arm
    if not arm.skill:
        return None
    if arm.skill.startswith("carry_to") and arm.dest:
        return arm.dest[1:]
    if arm.skill.startswith("move_above:"):
        e = scene["blocks"].get(arm.skill.split(":", 1)[1])
        return (e["x"], e["y"]) if e else None
    return None


def hand_facts(rt, scene) -> dict | None:
    h = scene.get("hand")
    if not h:
        return None
    arm = rt.world.arm
    d = math.dist(arm.xyz, (h["x"], h["y"], h["z"]))
    sp = math.hypot(h["vx"], h["vy"])
    if sp < 0.5:
        motion = "still"
    else:
        gx, gy = arm.x - h["x"], arm.y - h["y"]
        c = (h["vx"] * gx + h["vy"] * gy) / (sp * (math.hypot(gx, gy) or 1))
        motion = "coming toward the gripper" if c > 0.3 else "moving away from the gripper" if c < -0.3 else "moving across"
    hd = heading(rt, scene)
    in_path = hd is not None and seg_dist((h["x"], h["y"]), (arm.x, arm.y), hd) < 10
    return {"d": d, "motion": motion, "in_path": in_path}


def spotter(rt, variant: str) -> tuple[dict, dict, dict]:
    T = rt.plan["question_templates"]["spotter"]
    scene, arm = rt.scene, rt.world.arm
    hd = heading(rt, scene)
    state = {
        "orders": orders_view(rt),
        "arm": {
            **{k: v for k, v in status_view(rt).items() if k != "last_result"},
            "heading_to": f"({hd[0]:.0f}, {hd[1]:.0f}) cm" if hd else "not travelling",
            "pace": rt.pace(),
        },
    }
    hf = hand_facts(rt, scene)
    state["hands"] = [] if hf is None else [{
        "what": "a person's hand",
        "distance_to_gripper": dist_text(hf["d"]),
        "motion": hf["motion"],
        "in_the_arm_path": "yes: within 10 cm of the line from the gripper to where it is heading" if hf["in_path"] else "no",
    }]
    near = []
    for e in scene["blocks"].values():
        if e["where"] == "gripper":
            continue
        d = dxy((arm.x, arm.y), (e["x"], e["y"]))
        if d < 25:
            near.append({"name": e["name"], "distance_to_gripper": dist_text(d), "moved_by_a_person_recently": e["id"] in rt.recently_moved})
    state["nearby"] = near
    qs = {"action": _q(T["action"], ACTIONS)}
    for oid, o in rt.limiter.conditional.items():
        qs[f"order:{oid}"] = _q({"question": T["order"]["question"].format(condition=o["condition"]), "order": o["text"]},
                                {"on": "yes, the condition holds now", "off": "no, it does not hold now"})
    return state, qs, spotter_truth(rt)


def spotter_truth(rt) -> dict:
    ts, arm = rt.truth_scene, rt.world.arm
    hf = hand_facts(rt, ts)
    if hf is None:
        act = "continue"
    elif hf["d"] < 10 and hf["motion"].startswith("coming"):
        act = "evade_up"
    elif hf["d"] < 15 and not hf["motion"].startswith("moving away"):
        act = "back_off"
    elif hf["motion"].startswith("coming") or hf["in_path"] or hf["d"] < 15:
        act = "hold"
    else:
        act = "slow"
    t = {"action": [act]}
    for oid, o in rt.limiter.conditional.items():
        enf = o["enforce"]
        if "hand_clearance_cm" in enf:
            on = hf is not None and hf["d"] < 40
        elif "near" in enf:
            e = ts["blocks"].get(enf["near"])
            on = e is not None and e["where"] != "gripper" and dxy((arm.x, arm.y), (e["x"], e["y"])) < enf["within_cm"]
        else:
            on = False
        t[f"order:{oid}"] = ["on" if on else "off"]
    return t
