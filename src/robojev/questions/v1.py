"""Question battery v1: v0's judgments plus the sequencer.

`next` offers only primitives whose preconditions hold (code-filtered), so Jev never picks the
impossible; `place` turns "put it beside the phone" into a relation code can resolve; `task_done`
lets the loop go idle. Still one request per tick, still text only.
"""
from __future__ import annotations

from robojev.config import Config
from robojev.skills import PLACE_WORDS, offered
from robojev.world import World

VERSION = "v1"
NONE = "none_of_these"


def build(cfg: Config, w: World, brain=None) -> dict:
    labels = [e.label for e in w.entities]
    tgt = w.committed_target if w.committed_target in labels else None
    tgt_text = f"`{tgt}`" if tgt else "not chosen yet"
    q = {}
    q["target"] = {
        "type": "choice",
        "instructions": {
            "question": "Which object in `objects` is the one the user's request (`user_request.text`) wants the arm to act on (pick up, hover over, move)?",
            "rules": "Objects are seen by a depth camera and described only by colour, size and shape, so e.g. a paper cup appears as a white or tan cup-like object. Colour names come from a camera and can be off by a shade (black vs dark gray, white vs light gray, orange vs tan). Pick the object that most plausibly matches the request. If the arm is already holding an object, that object is the target. Choose none_of_these only when there is no request or nothing plausibly matches.",
        },
        "criteria": {**{l: None for l in labels}, NONE: "the request matches none of the listed objects, or there is no request"},
    }
    place_opts = {"unspecified": "the request does not say where to put the object, or it is not a moving task",
                  "where_it_was": "back where the object was picked up"}
    for l in labels:
        if l == tgt:
            continue
        for rel, words in PLACE_WORDS.items():
            place_opts[f"{rel}:{l}"] = f"{words} {l}"
    q["place"] = {
        "type": "choice",
        "instructions": {
            "question": f"If the user wants {tgt_text} moved somewhere, where should it be put down?",
            "rules": "Left and right are the robot's left and right as it faces the table. 'In front of' means closer to the robot than the reference object. If the request only asks to pick up or hover, choose unspecified.",
        },
        "criteria": place_opts,
    }
    prims = offered(cfg, w, brain) if brain is not None else []
    next_criteria = {p.key: p.text for p in prims}
    # `next` is the guard that moves the arm, so it is asked three ways. Same options, same state,
    # different wording; the brain acts only on a 2-of-3 majority. Extra questions in one request are
    # evaluated in parallel and cost essentially no latency (01-jev-facts, patterns/fan-out).
    q["next"] = {
        "type": "choice",
        "instructions": {
            "question": f"The user's request is `user_request.text`. Given `arm` (what the gripper holds, what it is above, the current primitive and its last result) and the standing orders, which primitive should run next?",
            "rules": "Only feasible primitives are listed. A pick-and-place goes: move above the object, descend to grasp, close gripper, lift, move to place, lower to place, open gripper, retreat. If the request is complete, or nothing useful can be done, choose hold. If a primitive just failed, choose what recovers (e.g. open the gripper and try again).",
        },
        "criteria": dict(next_criteria),
    }
    q["next_b"] = {
        "type": "choice",
        "instructions": {
            "question": "What is the single best next step towards finishing `user_request.text`, starting from the situation in `arm` and `objects`?",
            "rules": "Every listed step is possible right now. Think of the usual order for moving an object: get above it, come down around it, close on it, raise it, carry it to where it goes, set it down, let go, back away. Pick the one step that follows from what has already happened (`arm.primitive.last_result`). If the request is already satisfied, or no listed step helps, pick staying put.",
        },
        "criteria": dict(next_criteria),
    }
    q["next_c"] = {
        "type": "choice",
        "instructions": {
            "question": "Which of the listed actions would a careful operator run now, given `user_request.text`, `standing_orders`, `arm` and `objects`?",
            "rules": "A careful operator does not skip a stage, does not move on until the last action reported success, and stops rather than guessing. Weigh the standing orders first. If the last action failed, choose the action that recovers from it. If nothing listed is clearly the right move, or the request is finished, choose to stay put.",
        },
        "criteria": dict(next_criteria),
    }
    q["speed"] = {
        "type": "score",
        "instructions": "How fast should the arm move right now, given the standing orders, how close it is to objects, and whether it is carrying something?",
        "criteria": ["very slow, creeping", "slow", "normal", "fast"],
    }
    q["avoid"] = {
        "type": "choice",
        "instructions": {
            "question": "Given the standing orders and where everything is, should the gripper move away from any object right now, even if that means leaving its target?",
            "rules": "Choose an object only when a standing order requires distance from it and the gripper is too close, or is about to be. Otherwise choose no_avoidance_needed.",
        },
        "criteria": {"no_avoidance_needed": "nothing needs avoiding right now",
                     **{f"move_away_from:{l}": f"the gripper should back away from {l}" for l in labels}},
    }
    q["orders_violated"] = {
        "type": "noul",
        "instructions": "Is the arm currently doing, or about to do, something the `standing_orders` forbid?",
        "criteria": {"true": "the current motion, position, target or speed conflicts with a standing order",
                     "false": "no standing order is being broken, or there are no standing orders"},
    }
    q["task_done"] = {
        "type": "noul",
        "instructions": "Has the user's request (`user_request.text`) been completed, judging by `arm` and `objects`?",
        "criteria": {"true": "everything the request asked for has happened (e.g. the object has been placed and released where asked)",
                     "false": "something the request asked for has not happened yet, or there is no request"},
    }
    return q
