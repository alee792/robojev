"""Question battery v0: six narrow typed judgments, one request per tick.

Motion questions are templated with the previously committed target (option 2 of the ordering
puzzle in 04-doom-questions.md). Geometry is never asked; it is stated. Keys are not sent to the
model, so every instruction carries its own meaning.
"""
from __future__ import annotations

from robojev.config import Config
from robojev.world import World

VERSION = "v0"

NONE = "none_of_these"


def build(cfg: Config, w: World) -> dict:
    labels = [e.label for e in w.entities]
    tgt = w.committed_target if w.committed_target in labels else None
    tgt_text = f"`{tgt}`" if tgt else "not chosen yet"
    q = {}
    q["target"] = {
        "type": "choice",
        "instructions": {
            "question": "Which object in `objects` does the user's request (`user_request.text`) refer to?",
            "rules": "Judge by the description in `looks_like` and the words of the request. If the request names nothing present, or no object matches, choose none_of_these.",
        },
        "criteria": {**{l: None for l in labels}, NONE: "the request matches none of the listed objects, or there is no request"},
    }
    q["motion"] = {
        "type": "choice",
        "instructions": {
            "question": f"The arm's current target is {tgt_text}. Considering `arm`, `objects`, `relations` and the standing orders, what should the arm do right now?",
            "context": "The arm hovers above its target; it does not grasp. `relations.gripper_offset_from_target` says how far it is from being above the target.",
        },
        "criteria": {
            "approach": "move to hover above the target (or keep closing in on it)",
            "hold": "stay where it is",
            "back_off": "move horizontally away from the target",
            "rise_away": "go up, away from the table and everything on it",
        },
    }
    q["hover_position"] = {
        "type": "choice",
        "instructions": f"When hovering over {tgt_text}, where should the gripper sit relative to the object, given the standing orders?",
        "criteria": {
            "directly_above": "centred over the object",
            "offset_toward_robot": "over the near side, between the object and the robot base",
            "offset_left": "to the left of the object (the robot's left)",
            "offset_right": "to the right of the object (the robot's right)",
        },
    }
    q["hover_height"] = {
        "type": "choice",
        "instructions": "How high above the table should the gripper hover, given the standing orders and the objects present?",
        "criteria": {"low": "about 12 cm above the table, close to the objects", "high": "about 22 cm above the table, well clear of everything"},
    }
    q["speed"] = {
        "type": "score",
        "instructions": "How fast should the arm move right now, given the standing orders, how close it is to objects, and what the user asked?",
        "criteria": ["very slow, creeping", "slow", "normal", "fast"],
    }
    q["orders_violated"] = {
        "type": "noul",
        "instructions": "Is the arm currently doing, or about to do, something the `standing_orders` forbid?",
        "criteria": {"true": "the current motion, position, target or speed conflicts with a standing order",
                     "false": "no standing order is being broken, or there are no standing orders"},
    }
    return q
