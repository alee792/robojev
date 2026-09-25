"""Scenarios: a starting world, a task, a scripted person, and the ground truth the oracle scores with.

Core (sorting-centred; prompts may be tuned on these):
  sort_static        sort numbered blocks into the tray, lowest on the left
  sort_moved_target  ... the person slides the block the arm is about to pick
  sort_take_back     ... the person takes a placed block back out
  sort_hand_in_path  ... a hand reaches into the carry path, stays 3 s, withdraws
  sort_correction    ... "actually, highest on the left" with a block in the gripper
  sort_chatter       ... "nice, that's looking good" mid-task
  sort_wait_go       ... "wait", then "ok go on"
  group_colour       put each block in the bin of its own colour (the person nudges a waiting block)
  sort_constraint    ... "don't touch the green block"; the person slides it against the next target
Held out (reported separately, never tuned on; utterances phrased differently from the core ones):
  tower              stack three blocks in a given order; the red block is moved during the approach;
                     the order changes mid-task
  handover           hand a named block to the person; "hang on a sec" / "right, go ahead"
  standing_rule      a standing rule ("keep the yellow block out of the tray") while the person keeps
                     putting it back in the tray

Each person line carries its truth (Utt): what kind of line it is, and for a correction the task
version it switches to. Goals are per task version (eval/oracle.py Goal).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from e12_blocksworld.scenarios import SPOTS

from ..sim.person import (Beat, Person, after, approaching, carrying, hold_out_hand, intent, put_into, reach_into_path,
                          say, slide_block, slide_next_to, take_out)
from ..sim.world import Block, Noise, SimWorld
from .oracle import Goal

NUMBERS = [1, 3, 5, 7, 11, 12]
PLAIN = ["red", "blue", "orange", "purple", "white"]


@dataclass
class Utt:
    kind: str                     # correction | chatter | wait | go_on
    version: int | None = None    # corrections: the task version it switches to
    clarify: str | None = None    # what the person answers if asked what they meant

    @property
    def expect_change(self) -> bool:
        return self.kind != "chatter"


@dataclass
class Scenario:
    name: str
    about: str
    task: str
    world: SimWorld
    person: Person
    versions: list
    utts: dict = field(default_factory=dict)
    held_out: bool = False
    interference: bool = False     # counts for pass criterion 2 (interference / correction)

    def version_for(self, messages: list) -> int:
        v = 0
        for m in messages:
            u = self.utts.get(m)
            if u is not None and u.version is not None:
                v = u.version
            elif u is None:
                for u2 in self.utts.values():   # an answer to "what did you mean?" carries its correction's version
                    if u2.clarify == m and u2.version is not None:
                        v = u2.version
        return v

    def goal_for(self, messages: list) -> Goal:
        return self.versions[self.version_for(messages)]

    def said(self) -> list:
        return [b.text for b in sorted(self.person.beats, key=lambda b: b.fired_t or 0) if b.text and b.fired_t is not None]

    def goal_now(self) -> Goal:
        return self.goal_for(self.said())

    def answer(self, question: str) -> str:
        said = self.said()
        for m in reversed(said):
            u = self.utts.get(m)
            if u and u.clarify:
                return u.clarify
        return "yes, just carry on as you were"


# ---------------------------------------------------------------- worlds


def numbered(rng: random.Random, numbers=NUMBERS, extra=(), colours=PLAIN):
    spots = rng.sample(SPOTS, len(numbers) + len(extra))
    blocks = [Block(f"block_{n}", rng.choice(colours), *spots[i], number=n, size=rng.choice(["small", "medium"]))
              for i, n in enumerate(numbers)]
    blocks += [Block(e["id"], e["colour"], *spots[len(numbers) + j], number=e.get("number"), size=e.get("size", "small"))
               for j, e in enumerate(extra)]
    return blocks


def sort_goal(ids_by_number: dict, desc: bool = False, constraints=()) -> Goal:
    ranked = sorted(ids_by_number, key=lambda b: ids_by_number[b], reverse=desc)
    return Goal({b: f"tray_slot_{k}" for k, b in enumerate(ranked, 1)}, list(constraints))


def _nums(blocks) -> dict:
    return {b.id: b.number for b in blocks if b.number is not None}


SORT_TASK = "Line the numbered blocks up in the tray, lowest number on the left."


def _sort(seed, noise, beats_fn=None, utts=None, task=SORT_TASK, extra=(), constraints=(), n_slots=6, name="", about="",
          interference=False, versions_fn=None):
    rng = random.Random(seed)
    blocks = numbered(rng, NUMBERS[: n_slots] if not extra else NUMBERS[:5], extra)
    w = SimWorld(blocks, n_slots=n_slots, noise=noise, seed=seed)
    nums = _nums(blocks)
    versions = versions_fn(nums) if versions_fn else [sort_goal(nums, constraints=constraints)]
    beats = beats_fn(w) if beats_fn else []
    sc = Scenario(name, about, task, w, None, versions, utts or {}, interference=interference)
    sc.person = Person(beats, answer=sc.answer)
    w.person = sc.person
    return sc


# ---------------------------------------------------------------- core


def sort_static(seed, noise=None):
    return _sort(seed, noise, name="sort_static", about="sort 6 numbered blocks into the tray, lowest on the left (static)")


def sort_moved_target(seed, noise=None):
    beats = lambda w: [Beat("person slides the block the arm is about to pick", lambda w, t: approaching(w, 1, 10.0),  # noqa: E731
                            slide_block(lambda w: intent(w)["object"]))]
    return _sort(seed, noise, beats, name="sort_moved_target", interference=True,
                 about="the person slides the block the arm is about to pick")


def sort_take_back(seed, noise=None):
    beats = lambda w: [Beat("person takes the block in slot 1 back out", lambda w, t: carrying(w, 3), take_out("tray_slot_1"))]  # noqa: E731
    return _sort(seed, noise, beats, name="sort_take_back", interference=True,
                 about="the person takes a placed block back out of the tray while the arm carries another")


def sort_hand_in_path(seed, noise=None):
    beats = lambda w: [Beat("hand reaches into the carry path", lambda w, t: carrying(w, 1, 20.0), reach_into_path)]  # noqa: E731
    return _sort(seed, noise, beats, name="sort_hand_in_path", interference=True,
                 about="a hand reaches into the arm's path mid-carry, stays 3 s, withdraws")


CORRECTION = "actually, highest on the left"


def sort_correction(seed, noise=None):
    beats = lambda w: [Beat("user: actually, highest on the left", lambda w, t: carrying(w, 2), say(CORRECTION), disturbs=False)]  # noqa: E731
    return _sort(seed, noise, beats, name="sort_correction", interference=True,
                 utts={CORRECTION: Utt("correction", 1, clarify="I mean the biggest number goes in the leftmost slot")},
                 versions_fn=lambda nums: [sort_goal(nums), sort_goal(nums, desc=True)],
                 about="'actually, highest on the left' while carrying the third block")


CHATTER = "nice, that's looking good"


def sort_chatter(seed, noise=None):
    beats = lambda w: [Beat("user: nice, that's looking good", lambda w, t: carrying(w, 1), say(CHATTER), disturbs=False)]  # noqa: E731
    return _sort(seed, noise, beats, name="sort_chatter", utts={CHATTER: Utt("chatter")},
                 about="chatter mid-task must not change behaviour")


def sort_wait_go(seed, noise=None):
    def beats(w):
        b1 = Beat("user: wait", lambda w, t: carrying(w, 2), say("wait"), disturbs=False)
        b2 = Beat("user: ok go on", after(b1, 4.0), say("ok go on"), disturbs=False)
        return [b1, b2]
    return _sort(seed, noise, beats, name="sort_wait_go", interference=True,
                 utts={"wait": Utt("wait"), "ok go on": Utt("go_on")}, about="'wait' mid-carry, then 'ok go on' 4 s later")


def group_colour(seed, noise=None):
    rng = random.Random(seed)
    cols = ["red", "red", "blue", "blue", "green", "green"]
    rng.shuffle(cols)
    nums = [2, 4, 6, 8, 9, 10]
    spots = rng.sample(SPOTS, 6)
    blocks = [Block(f"block_{n}", c, *spots[i], number=n) for i, (n, c) in enumerate(zip(nums, cols))]
    w = SimWorld(blocks, bins=("red", "blue", "green"), noise=noise, seed=seed)
    goal = Goal({b.id: f"{b.colour}_bin" for b in blocks}, [])

    def not_current(w):
        cur = intent(w).get("object")
        return next(b.id for b in w.blocks.values() if b.where == "table" and b.id != cur)

    beats = [Beat("person nudges a block that is still waiting", lambda w, t: carrying(w, 2), slide_block(not_current, 10.0))]
    sc = Scenario("group_colour", "put each block in the bin of its colour; the person nudges a waiting block",
                  "Put each block in the bin of its own colour.", w, None, [goal])
    sc.person = Person(beats, answer=sc.answer)
    w.person = sc.person
    return sc


def sort_constraint(seed, noise=None):
    green = {"id": "green_block", "colour": "green"}
    con = {"text": "don't touch the green block", "kind": "dont_touch", "object": "green_block", "place": "none"}
    beats = lambda w: [Beat("person slides the green block against the next target", lambda w, t: approaching(w, 1, 30.0, "move_object"),  # noqa: E731
                            slide_next_to("green_block"))]
    return _sort(seed, noise, beats, task="Line the numbered blocks up in the tray, lowest number on the left. Don't touch the green block.",
                 extra=[green], constraints=[con], name="sort_constraint", interference=True,
                 about="'don't touch the green block'; the person slides it right against the next target")


# ---------------------------------------------------------------- held out

TOWER_FIX = "Hmm, yellow in the middle please, red goes up top."


def tower(seed, noise=None):
    rng = random.Random(seed)
    extra = [{"id": "blue_block", "colour": "blue"}, {"id": "red_block", "colour": "red"}, {"id": "yellow_block", "colour": "yellow"}]
    blocks = numbered(rng, [2, 6], extra, colours=["orange", "purple", "white"])
    w = SimWorld(blocks, n_slots=0, noise=noise, seed=seed)
    v0 = Goal({"red_block": "on:blue_block", "yellow_block": "on:red_block"}, [], stays=["blue_block"])
    v1 = Goal({"yellow_block": "on:blue_block", "red_block": "on:yellow_block"}, [], stays=["blue_block"])
    beats = [Beat("person moves the red block during the approach", lambda w, t: approaching(w, 0, 12.0) and intent(w).get("object") == "red_block",
                  slide_block(lambda w: "red_block", -12.0)),
             Beat("user: yellow in the middle, red on top", lambda w, t: w.blocks["red_block"].where == "on:blue_block" and carrying(w),
                  say(TOWER_FIX), disturbs=False)]
    sc = Scenario("tower", "stack red on blue, yellow on top; red is moved during the approach; then the order changes",
                  "Build a tower on the blue block: the red block on it, then the yellow block on top.", w, None, [v0, v1],
                  {TOWER_FIX: Utt("correction", 1, clarify="The yellow block should sit on the blue one, and the red one on top of the yellow.")},
                  held_out=True, interference=True)
    sc.person = Person(beats, answer=sc.answer)
    w.person = sc.person
    return sc


def handover(seed, noise=None):
    rng = random.Random(seed)
    blocks = numbered(rng, [2, 4, 6, 8])
    w = SimWorld(blocks, n_slots=0, noise=noise, seed=seed)
    goal = Goal({"block_4": "person"}, [])
    b1 = Beat("user: hang on a sec", lambda w, t: t >= 45, say("hang on a sec"), disturbs=False)
    b2 = Beat("user: right, go ahead", after(b1, 3.0), say("right, go ahead"), disturbs=False)
    b3 = Beat("person holds out a hand", lambda w, t: w.arm.holding == "block_4", hold_out_hand(), disturbs=False)
    sc = Scenario("handover", "hand block 4 to the person; 'hang on a sec' / 'right, go ahead'; the person holds out a hand",
                  "Please pass me block 4.", w, None, [goal],
                  {"hang on a sec": Utt("wait"), "right, go ahead": Utt("go_on")}, held_out=True, interference=True)
    sc.person = Person([b1, b2, b3], answer=sc.answer)
    w.person = sc.person
    return sc


def standing_rule(seed, noise=None):
    rng = random.Random(seed)
    blocks = numbered(rng, [2, 5, 8, 9], [{"id": "yellow_block", "colour": "yellow"}])
    w = SimWorld(blocks, n_slots=6, noise=noise, seed=seed)
    con = {"text": "keep the yellow block out of the tray", "kind": "keep_out_of", "object": "yellow_block", "place": "tray"}
    goal = sort_goal(_nums(blocks), constraints=[con])
    b1 = Beat("person puts the yellow block into tray slot 6", lambda w, t: carrying(w, 1), put_into("yellow_block", "tray_slot_6"))
    b2 = Beat("person puts the yellow block into tray slot 5", lambda w, t: carrying(w, 3) and w.blocks["yellow_block"].where == "table",
              put_into("yellow_block", "tray_slot_5"))
    sc = Scenario("standing_rule", "keep the yellow block out of the tray while the person keeps putting it back",
                  "Put the numbered blocks in the tray in order, smallest number first on the left, and keep the yellow block out of the tray.",
                  w, None, [goal], held_out=True, interference=True)
    sc.person = Person([b1, b2], answer=sc.answer)
    w.person = sc.person
    return sc


CORE = {f.__name__: f for f in (sort_static, sort_moved_target, sort_take_back, sort_hand_in_path, sort_correction, sort_chatter,
                                sort_wait_go, group_colour, sort_constraint)}
HELD_OUT = {f.__name__: f for f in (tower, handover, standing_rule)}
ALL = {**CORE, **HELD_OUT}


def make(name: str, seed: int = 12, noise: Noise | None = None) -> Scenario:
    return ALL[name](seed, noise)
