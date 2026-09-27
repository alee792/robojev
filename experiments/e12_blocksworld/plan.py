"""Planner stub: hand-written compiled Plans, one per task template, plus what a Plan means in code.

A Plan is plain JSON: bindings (entity ids), knobs (declared adjustable settings with enumerated
values and a default), the skill graph, the goal rule, Orders, and question templates per layer.
`goals()` and `next_skill()` are code reading the Plan: the goal of every bound block under given knob
values, and the scripted policy (the oracle, and the `solved` Facts, are both this function run on a
Scene: the oracle on the true one, Facts on the perceived one).
"""

from __future__ import annotations

from .world import ABOVE_CM, FINGER_CM, dxy

ASC, DESC = "ascending", "descending"
UNCHANGED = "unchanged"

# ---------------------------------------------------------------- knobs

PACE = {
    "describe": "how fast the arm moves",
    "values": {"normal": "normal speed", "slow": "half speed; for when the user wants the arm slower or more careful"},
    "default": "normal",
}
SKIP = {
    "describe": "a one-off: set aside the block the arm is going for right now and do it last",
    "values": {"current_target": "set aside the block the arm is currently going for (do it last) and start on another block"},
    "default": None,
    "event": True,   # no standing value: applying it defers one block, then it is spent
}
ORDER = {
    "describe": "which way the numbers run along the tray",
    "values": {
        ASC: "smallest number in slot 1 (the robot's left end of the tray), numbers increasing to the right",
        DESC: "largest number in slot 1 (the robot's left end of the tray), numbers decreasing to the right",
    },
    "default": ASC,
}


def destination_knob(bins: list[str], default: str) -> dict:
    return {"describe": "which bin the blocks go into", "values": {b: f"the {b} bin" for b in bins}, "default": default}


GOAL_KNOBS = ("order", "destination")   # knobs that change where blocks end up

# ---------------------------------------------------------------- orders

CLEAR_OF_HANDS = {
    "id": "clear_of_hands", "kind": "conditional",
    "text": "While a person's hand is within 40 cm of the gripper, keep at least 15 cm from it and never move toward it.",
    "condition": "a person's hand is within 40 cm of the gripper",
    "enforce": {"hand_clearance_cm": 15},
}


def no_touch(bid: str, name: str) -> dict:
    return {"id": f"no_touch_{bid}", "kind": "always", "text": f"Don't touch the {name}.", "enforce": {"forbid": bid}}


def slow_near(bid: str, name: str) -> dict:
    return {"id": f"slow_near_{bid}", "kind": "conditional",
            "text": f"Move slowly while the gripper is within 10 cm of the {name}.",
            "condition": f"the gripper is within 10 cm of the {name}", "enforce": {"pace": "slow", "near": bid, "within_cm": 10}}


# ---------------------------------------------------------------- question templates (per layer)

QUESTION_TEMPLATES = {
    "sequencer": {
        "next_skill": {
            "question": "Which skill should the arm run next to make progress on `task`, within `orders`?",
            "rules": [
                "Only skills whose preconditions hold right now are listed.",
                "Each block goes: move above it, pick it, carry it to its goal, place, release, retreat.",
                "Where `facts` already answers something (where a block belongs, which step is next), use it rather than re-deriving it.",
                "If `status.last_result` is a failure, choose the skill that recovers (look again, move above the block where it is now, or retreat).",
                "If a block's goal is taken by another block, carry the held block to a free spot on the table and come back to it later.",
                "If everything is already where it belongs, choose to stay still.",
            ],
        },
        "task_status": {"question": "Is `task` complete?"},
        "escalate": {"question": "Can the arm's next step be decided confidently from this state by direct lookup, without careful multi-step reasoning?"},
    },
    "listener": {
        "intent": {"question": "What is the user's new message (`utterance.text`) asking the robot to do, given what it is doing now (`task`, `status`)?"},
        "knob": {"question": "After the user's new message (`utterance.text`), what should the knob `{knob}` ({describe}) be? It is `{now}` now."},
        "when": {"question": "If the user's new message (`utterance.text`) changes anything, when should the change take effect?"},
        "escalate": {"question": "Can the user's new message (`utterance.text`) be handled with the knobs listed in `knobs` and the intents offered, without a smarter model re-planning the task?"},
    },
    "spotter": {
        "action": {
            "question": "What should the arm do right now about what is happening around it (`hands`, `nearby`), within `orders`?",
            "rules": [
                "This overrides the arm's current skill.",
                "A hand close to the gripper (under 15 cm) calls for backing off, or lifting straight up if it is still coming closer and very close.",
                "A hand that is coming toward the gripper, or sits in the arm's path, calls for holding still.",
                "A hand that is present but neither coming closer nor in the path calls for slowing down.",
                "Blocks being moved by a person are not a reason to stop unless a hand is involved.",
            ],
        },
        "order": {"question": "Does this condition hold right now: {condition}?"},
    },
}

SKILL_GRAPH = {
    "start": "survey",
    "per_block": ["move_above(block)", "pick(block)", "carry_to(goal)", "place", "release", "retreat"],
    "on_failure": "survey, or move_above(block) where the block is now, or retreat if the gripper is low",
    "goal_taken": "carry_to(park), a free table spot; come back to that block later",
    "crowded_by_forbidden": "nudge_clear(block) before pick",
    "done_when": "every bound block is at its goal, the hand is empty and the gripper is up",
}


def _plan(template: str, task: str, bindings: dict, knobs: dict, goal: dict, orders: list) -> dict:
    return {
        "template": template, "task": task, "bindings": bindings, "knobs": knobs, "goal": goal,
        "skill_graph": SKILL_GRAPH, "orders": [CLEAR_OF_HANDS, *orders], "question_templates": QUESTION_TEMPLATES,
    }


def plan_sort_number(block_ids: list[str], n_slots: int, order: str = ASC, orders=()) -> dict:
    knobs = {"order": {**ORDER, "default": order}, "pace": PACE, "skip": SKIP}
    return _plan("sort_number", "Put the numbered blocks into the tray, one per slot, sorted by number.",
                 {"blocks": list(block_ids), "container": "tray", "slots": n_slots, "slot_1": "the robot's left end of the tray"},
                 knobs, {"rule": "slot_by_rank", "knob": "order"}, list(orders))


def plan_sort_colour(block_ids: list[str], bins: list[str], orders=()) -> dict:
    return _plan("sort_colour", "Put each block into the bin of its own colour.",
                 {"blocks": list(block_ids), "bins": list(bins)}, {"pace": PACE, "skip": SKIP},
                 {"rule": "bin_by_colour"}, list(orders))


def plan_move_except(block_ids: list[str], except_name: str, bins: list[str], default: str, orders=()) -> dict:
    return _plan("move_except", f"Move every block except the {except_name} into one bin.",
                 {"blocks": list(block_ids), "except": except_name, "bins": list(bins)},
                 {"destination": destination_knob(bins, default), "pace": PACE, "skip": SKIP},
                 {"rule": "bin_by_knob", "knob": "destination"}, list(orders))


def default_knobs(plan: dict) -> dict:
    return {k: v["default"] for k, v in plan["knobs"].items() if not v.get("event")}


def task_text(plan: dict, knobs: dict) -> str:
    t = plan["task"]
    if "order" in knobs:
        t += f" Order: {knobs['order']}, i.e. {plan['knobs']['order']['values'][knobs['order']]}."
    if "destination" in knobs:
        t += f" Destination: the {knobs['destination']} bin."
    return t


def forbidden(plan: dict) -> set[str]:
    return {o["enforce"]["forbid"] for o in plan["orders"] if o["kind"] == "always" and "forbid" in o["enforce"]}


# ---------------------------------------------------------------- what a Plan means, in code

def goals(plan: dict, knobs: dict, scene: dict) -> dict[str, str]:
    """Goal of every bound block that is in the Scene, under these knob values."""
    blocks = scene["blocks"]
    ids = [b for b in plan["bindings"]["blocks"] if b in blocks]
    rule = plan["goal"]["rule"]
    if rule == "slot_by_rank":
        ranked = sorted(ids, key=lambda i: (blocks[i]["number"] or 0, i), reverse=knobs["order"] == DESC)
        return {bid: f"slot:{k}" for k, bid in enumerate(ranked, 1)}
    if rule == "bin_by_colour":
        return {bid: f"bin:{blocks[bid]['colour']}" for bid in ids}
    if rule == "bin_by_knob":
        return {bid: f"bin:{knobs['destination']}" for bid in ids}
    raise ValueError(rule)


def goal_free(scene: dict, goal: str, bid: str) -> bool:
    if goal.startswith("slot:"):
        has = scene["slots"][int(goal.split(":")[1])]["has"]
        return has is None or has == bid
    return True


def dest_xy(scene: dict, key: str, arm=None) -> tuple | None:
    kind, _, arg = key.partition(":")
    if kind == "slot":
        s = scene["slots"][int(arg)]
        return (s["x"], s["y"])
    if kind == "bin":
        b = scene["bins"][arg]
        return (b["x"], b["y"])
    if kind == "park" and arm is not None and arm.dest and arm.dest[0] == "park":
        return arm.dest[1:]
    return None


def above(arm, xy) -> bool:
    # Same distance test as pick's grasp check (skills.py): a per-axis box let the corners pass here and fail there.
    return xy is not None and arm.high and dxy((arm.x, arm.y), xy) <= ABOVE_CM


def crowded(scene: dict, bid: str, forb) -> str | None:
    """A forbidden block close enough that open fingers grasping `bid` would touch it."""
    b = scene["blocks"].get(bid)
    for f in forb:
        e = scene["blocks"].get(f)
        if b and e and e["where"] != "gripper" and ((b["x"] - e["x"]) ** 2 + (b["y"] - e["y"]) ** 2) ** 0.5 < FINGER_CM + 1.0:
            return f
    return None


def todo(plan: dict, knobs: dict, scene: dict) -> list[str]:
    g = goals(plan, knobs, scene)
    forb = forbidden(plan)
    return [b for b, goal in g.items() if b not in forb and scene["blocks"][b]["where"] not in (goal, "gripper")]


def complete(plan: dict, knobs: dict, scene: dict, arm) -> bool:
    return not todo(plan, knobs, scene) and arm.holding is None and arm.high


def next_skill(plan: dict, knobs: dict, scene: dict, arm, deferred=()) -> str:
    """The scripted policy: the skill a careful operator would run next. Returns a skill key."""
    g = goals(plan, knobs, scene)
    if arm.holding:
        if not arm.high:
            at_dest = arm.dest is not None and dxy((arm.x, arm.y), arm.dest[1:]) <= ABOVE_CM
            return "release" if at_dest else "retreat"
        goal = g.get(arm.holding)
        target = goal if goal and goal_free(scene, goal, arm.holding) else "park"
        if arm.dest and arm.dest[0] == target and above(arm, arm.dest[1:]):
            return "place"
        return f"carry_to:{target}"
    if not arm.high:
        return "retreat"
    left = todo(plan, knobs, scene)
    if not left:
        return "hold"

    def rank(b):
        goal = g[b]
        where = scene["blocks"][b]["where"]
        blocked = 0 if goal_free(scene, goal, b) else (1 if where.startswith("slot:") else 2)
        idx = int(goal.split(":")[1]) if goal.startswith("slot:") else 0
        return (b in deferred, blocked, idx, b)

    b = min(left, key=rank)
    e = scene["blocks"][b]
    if above(arm, (e["x"], e["y"])):
        return f"nudge_clear:{b}" if crowded(scene, b, forbidden(plan)) else f"pick:{b}"
    return f"move_above:{b}"
