"""Scenarios: a starting world, a compiled Plan, and a scripted human.

The human acts on triggers (conditions on the true world), not fixed times, so every backend meets the
same disturbance at the same point in the task. An Utterance carries its own ground truth: acceptable
answers per Listener question (preferred first) and whether behaviour should change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from . import plan as P
from .world import TRAY_Y, Block, Hand, World, dxy

SPOTS = [(x, y) for y in (14.0, 28.0) for x in (15.0, 27.0, 39.0, 51.0, 63.0, 75.0, 87.0)]
COLOURS = ["red", "blue", "yellow", "orange", "purple", "white"]


@dataclass
class Utterance:
    text: str
    intent: list
    knobs: dict = field(default_factory=dict)        # knob -> acceptable values; missing = ["unchanged"]
    when: list = field(default_factory=lambda: ["now", "at_next_safe_point"])
    escalate: list = field(default_factory=lambda: ["decide_now"])
    expect_change: bool = True
    tick: int | None = None
    changed_at: int | None = None

    def resolution(self) -> dict:
        """What the Planner (oracle escalation) makes of it: the preferred truth."""
        return {"intent": self.intent[0], "knobs": {k: v[0] for k, v in self.knobs.items()}, "when": self.when[0]}


@dataclass
class Event:
    name: str
    when: Callable
    do: Callable            # (world) -> Utterance | None
    disturbs: bool = True   # counts as a disturbance (moves the world) rather than talk
    fired: bool = False
    utt: Utterance | None = None


@dataclass
class Scenario:
    name: str
    about: str
    world: World
    plan: dict
    events: list
    final_knobs: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers for triggers

def placed(w: World) -> int:
    return sum(1 for b in w.blocks.values() if b.where.startswith(("slot:", "bin:")))


def running(w: World, prefix: str) -> bool:
    return bool(w.arm.skill and w.arm.skill.startswith(prefix))


def say(text: str, **kw) -> Callable:
    return lambda w: Utterance(text, **kw)


def numbered_world(rng: random.Random, n=5, extra=()) -> World:
    spots = rng.sample(SPOTS, n + len(extra))
    nums = list(range(1, n + 1))
    rng.shuffle(nums)
    blocks = [Block(f"b{k}", rng.choice(COLOURS), *spots[i], number=k) for i, k in enumerate(nums)]
    blocks += [Block(e["id"], e["colour"], *spots[n + j], label=e.get("label", "")) for j, e in enumerate(extra)]
    return World(blocks, n_slots=n)


def sort_number(rng, orders_for=None, extra=()) -> tuple[World, dict]:
    w = numbered_world(rng, 5, extra)
    ids = [b for b in w.blocks if b.startswith("b")]
    orders = orders_for(w) if orders_for else ()
    return w, P.plan_sort_number(sorted(ids), len(ids), P.ASC, orders)


# ---------------------------------------------------------------- the human's moves

def move_block_away(bid_of: Callable) -> Callable:
    def do(w: World):
        b = w.blocks[bid_of(w)]
        x, y = w.free_spot(near=(b.x + 14, b.y))
        b.x, b.y, b.where = x, y, "table"
    return do


def take_from_slot(k: int) -> Callable:
    def do(w: World):
        bid = w.slot_has(k)
        if bid:
            x, y = w.free_spot(near=(45.0, 30.0))
            w.blocks[bid].x, w.blocks[bid].y, w.blocks[bid].where = x, y, "table"
    return do


def reach_into_path(w: World):
    """A hand reaches in from the side toward a point 12 cm ahead of the gripper on its carry path,
    stays 3 s, and leaves. At 4 cm/tick it arrives about when the arm would."""
    arm = w.arm
    gx, gy, (tx, ty) = arm.x, arm.y, arm.dest[1:]
    L = dxy((gx, gy), (tx, ty)) or 1.0
    ux, uy = (tx - gx) / L, (ty - gy) / L
    px, py = gx + 12 * ux, gy + 12 * uy
    for sgn in (1, -1):
        sx, sy = px - sgn * 25 * uy, py + sgn * 25 * ux
        if 2 <= sx <= 98 and 2 <= sy <= 70:
            break
    w.hand = Hand(start=(sx, sy), target=(px, py), stay_ticks=30, speed=4.0)


def slide_forbidden_next_to_target(fid: str) -> Callable:
    def do(w: World):
        t = w.blocks[w.arm.skill.split(":", 1)[1]]
        side = 3.2 if t.x < 80 else -3.2
        f = w.blocks[fid]
        f.x, f.y, f.where = t.x + side, t.y, "table"
    return do


# ---------------------------------------------------------------- scenarios

def crawl_sort(seed):
    w, p = sort_number(random.Random(seed))
    return Scenario("crawl_sort", "static: sort 5 numbered blocks ascending into the tray", w, p, [])


def sort_colours(seed):
    rng = random.Random(seed)
    cols = ["red", "red", "blue", "blue", "green", "green"]
    spots = rng.sample(SPOTS, len(cols))
    blocks = [Block(f"c{i + 1}", c, *spots[i], label="AB"[cols[:i].count(c)]) for i, c in enumerate(cols)]
    w = World(blocks, bins={"red": (25.0, TRAY_Y), "blue": (50.0, TRAY_Y), "green": (75.0, TRAY_Y)})
    return Scenario("sort_colours", "sort 6 mixed blocks into red / blue / green bins", w,
                    P.plan_sort_colour([b.id for b in blocks], list(w.bins)), [])


def reverse_midway(seed):
    w, p = sort_number(random.Random(seed))
    ev = Event("user: actually, reverse it", lambda w: running(w, "carry_to") and placed(w) >= 2,
               say("actually, reverse it", intent=["adjust"], knobs={"order": [P.DESC]}), disturbs=False)
    return Scenario("reverse_midway", "'actually, reverse it' while carrying the third block", w, p, [ev], {"order": P.DESC})


def moved_target(seed):
    w, p = sort_number(random.Random(seed))

    def approaching(w):
        if not running(w, "move_above:") or placed(w) < 1:
            return False
        b = w.blocks[w.arm.skill.split(":", 1)[1]]
        return dxy((w.arm.x, w.arm.y), (b.x, b.y)) < 8

    evs = [
        Event("human moves the block the arm is approaching", approaching, move_block_away(lambda w: w.arm.skill.split(":", 1)[1])),
        Event("human moves the block during the grasp", lambda w: running(w, "pick:") and placed(w) >= 3 and w.arm.z < 10,
              move_block_away(lambda w: w.arm.target)),
    ]
    return Scenario("moved_target", "the human moves the target during the approach, and later during a grasp", w, p, evs)


def undo(seed):
    w, p = sort_number(random.Random(seed))
    ev = Event("human takes block out of slot 1", lambda w: running(w, "carry_to") and placed(w) >= 3, take_from_slot(1))
    return Scenario("undo", "the human takes a sorted block out of the tray and drops it on the table", w, p, [ev])


def hand_in_path(seed):
    w, p = sort_number(random.Random(seed))

    def mid_carry(w):
        a = w.arm
        return running(w, "carry_to") and a.holding and placed(w) >= 1 and a.dest and dxy((a.x, a.y), a.dest[1:]) >= 20

    ev = Event("hand reaches into the carry path", mid_carry, reach_into_path)
    return Scenario("hand_in_path", "a hand enters the path mid-carry and leaves after 3 s", w, p, [ev])


def forbidden_moves(seed):
    green = {"id": "g1", "colour": "green"}
    w, p = sort_number(random.Random(seed), lambda w: [P.no_touch("g1", "green block"), P.slow_near("g1", "green block")], extra=[green])
    ev = Event("human slides the green block next to the target", lambda w: running(w, "move_above:") and placed(w) >= 1,
               slide_forbidden_next_to_target("g1"))
    return Scenario("forbidden_moves", "order 'don't touch the green block'; the human slides it next to the target", w, p, [ev])


def chatter(seed):
    w, p = sort_number(random.Random(seed))
    ev = Event("user: nice, looking good", lambda w: running(w, "carry_to") and placed(w) >= 1,
               say("nice, looking good", intent=["continue", "not_for_me"], expect_change=False), disturbs=False)
    return Scenario("chatter", "an irrelevant utterance mid-task must not change behaviour", w, p, [ev])


def ambiguous(seed):
    w, p = sort_number(random.Random(seed))
    ev = Event("user: not that one", lambda w: running(w, "move_above:") and placed(w) >= 1,
               say("not that one", intent=["adjust"], knobs={"skip": ["current_target", P.UNCHANGED]},
                   escalate=["decide_now", "ask_a_smarter_model"]), disturbs=False)
    return Scenario("ambiguous", "'not that one' while approaching a block: set it aside, or escalate", w, p, [ev])


def move_except(seed):
    rng = random.Random(seed)
    cols = ["red", "blue", "yellow", "orange", "purple"]
    spots = rng.sample(SPOTS, len(cols))
    blocks = [Block(f"m{i + 1}", c, *spots[i]) for i, c in enumerate(cols)]
    w = World(blocks, bins={"left": (25.0, TRAY_Y), "right": (75.0, TRAY_Y)})
    p = P.plan_move_except([b.id for b in blocks if b.colour != "red"], "red block", list(w.bins), "left")
    ev = Event("user: use the right bin instead", lambda w: running(w, "carry_to") and placed(w) >= 2,
               say("use the right bin instead", intent=["adjust"], knobs={"destination": ["right"]}), disturbs=False)
    return Scenario("move_except", "move all blocks except the red one into a bin; 'use the right bin instead' midway",
                    w, p, [ev], {"destination": "right"})


SCENARIOS = {f.__name__: f for f in (crawl_sort, sort_colours, reverse_midway, moved_target, undo, hand_in_path,
                                     forbidden_moves, chatter, ambiguous, move_except)}


def make(name: str, seed: int = 12) -> Scenario:
    return SCENARIOS[name](seed)
