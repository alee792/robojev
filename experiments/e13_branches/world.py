"""E13 inputs: seeded scenes, task sentences and mid-task corrections.

Each task and correction carries a small label (`family`, `params`, `op`) saying what it means. Only
the evaluation side reads those labels (`oracle.py`, the mock backends, scoring). The system under
test (`plan.py`, `planner.py`, `router.py`) gets the sentence and `scene_view(scene)`, nothing else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PALETTE = ["red", "blue", "green", "yellow"]
BINS = ("left_bin", "right_bin")
TABLE = "stays_on_table"
SIZES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
FAMILIES = ("sort_number", "sort_size", "group_colour", "colour_to_bin", "all_except", "alternate", "parity_bins")
CATEGORIES = ("coverable", "uncoverable", "restate", "chatter", "pause")
CHATTER = ["nice", "looking good", "Sam, can you grab me a coffee?", "hmm, interesting"]
PAUSE = ["hold on a sec", "wait a moment"]


def slot(i: int) -> str:
    return f"tray_slot_{i}"


@dataclass
class Block:
    id: str
    number: int
    colour: str
    size_cm: float


@dataclass
class Scene:
    blocks: list[Block]
    stickers: dict[str, str] | None = None   # bin -> sticker colour (group_colour only)

    @property
    def n_slots(self) -> int:
        return len(self.blocks)

    def ids(self) -> list[str]:
        return [b.id for b in self.blocks]

    def colours(self) -> list[str]:
        return sorted({b.colour for b in self.blocks}, key=PALETTE.index)

    def count(self, colour: str) -> int:
        return sum(b.colour == colour for b in self.blocks)


@dataclass
class Correction:
    cid: str
    text: str
    category: str      # one of CATEGORIES; "coverable" = a natural parameter of the task expresses it
    op: dict           # meaning, for the oracle only


@dataclass
class Task:
    tid: str
    family: str
    text: str
    params: dict       # meaning, for the oracle only
    scene: Scene
    corrections: list[Correction] = field(default_factory=list)


def scene_view(scene: Scene) -> dict:
    """What the system under test sees: blocks, tray, bins, and which destinations hold one block."""
    slots = [slot(i) for i in range(1, scene.n_slots + 1)]
    bins = []
    for b in BINS:
        d = {"id": b, "position": "left of the tray" if b == "left_bin" else "right of the tray"}
        if scene.stickers:
            d["sticker"] = scene.stickers[b]
        bins.append(d)
    return {
        "blocks": [{"id": b.id, "number": b.number, "colour": b.colour, "size_cm": b.size_cm} for b in scene.blocks],
        "tray": {"slots": slots, "layout": f"{slots[0]} is the robot's far left; slots run left to right up to {slots[-1]}"},
        "bins": bins,
        "table": f"{TABLE} means the block is left where it is on the table",
        "destinations": {"one_block_each": slots, "any_number_of_blocks": list(BINS) + [TABLE]},
    }


# ---------------------------------------------------------------- scenes

def _numbers(rng: random.Random, n: int, need_both_parities=False) -> list[int]:
    while True:
        nums = list(range(1, n + 1)) if rng.random() < 0.5 else rng.sample(range(1, 13), n)
        rng.shuffle(nums)
        if not need_both_parities or len({x % 2 for x in nums}) == 2:
            return nums


def _colours(rng: random.Random, n: int, k: int) -> list[str]:
    """n colours drawn from k palette colours, each present at least once."""
    pal = rng.sample(PALETTE, k)
    cols = pal + [rng.choice(pal) for _ in range(n - k)]
    rng.shuffle(cols)
    return cols


def _scene(rng: random.Random, colours: list[str], need_both_parities=False) -> Scene:
    n = len(colours)
    nums = _numbers(rng, n, need_both_parities)
    sizes = rng.sample(SIZES, n)
    return Scene([Block(f"{c}_{x}", x, c, s) for c, x, s in zip(colours, nums, sizes)])


# ---------------------------------------------------------------- tasks + corrections

def _other_bin(b: str) -> str:
    return BINS[1] if b == BINS[0] else BINS[0]


def _bin_words(b: str) -> str:
    return b.replace("_bin", "")


def _excludable(sc: Scene, avoid=()) -> list[str]:
    return [c for c in sc.colours() if 0 < sc.count(c) < len(sc.blocks) and c not in avoid]


def _sort_task(rng, family) -> tuple[str, dict, Scene, list]:
    sc = _scene(rng, _colours(rng, rng.randint(4, 8), rng.randint(2, 3)))
    desc = rng.random() < 0.5
    if family == "sort_number":
        key, lo, hi = "number", "lowest number", "highest number"
        text = f"Line the blocks up in the tray by number, {hi if desc else lo} on the left."
    else:
        key, lo, hi = "size", "smallest", "biggest"
        text = (f"Put all the blocks in the tray biggest first, starting from the left." if desc
                else "Put all the blocks in the tray from smallest on the left to biggest on the right.")
    now, other = (hi, lo) if desc else (lo, hi)
    c = rng.choice(_excludable(sc))
    other_key = "size" if key == "number" else "number"
    other_text = ("actually, sort them by size instead, biggest on the left" if other_key == "size"
                  else "actually, sort them by number instead, lowest on the left")
    pool = [
        ("coverable", "actually, reverse it", {"op": "flip"}),
        ("coverable", "other way round please", {"op": "flip"}),
        ("coverable", f"{other} on the left instead", {"op": "set_order", "desc": not desc}),
        ("uncoverable", f"leave the {c} ones out", {"op": "assign_first", "select": {"colour": c}, "dest": TABLE}),
        ("uncoverable", f"put the {c} ones in the left bin instead", {"op": "assign_first", "select": {"colour": c}, "dest": "left_bin"}),
        ("uncoverable", other_text, {"op": "replace", "rule": {"assign": [], "tray": {"select": {"all": True}, "order": {"by": other_key, "desc": other_key == "size"}}}}),
        ("restate", f"{now} on the left, like you're doing", {"op": "none"}),
    ]
    return text, {"key": key, "desc": desc}, sc, pool


def _group_colour(rng):
    sc = _scene(rng, _colours(rng, rng.randint(4, 8), 2))
    c1, c2 = rng.sample(sc.colours(), 2)
    sc.stickers = {"left_bin": c1, "right_bin": c2}
    text = "Put each block in the bin whose sticker matches its colour."
    pool = [
        ("coverable", "swap the bins", {"op": "swap_bins"}),
        ("coverable", "other way round: each colour in the other bin", {"op": "swap_bins"}),
        ("uncoverable", f"leave the {c1} ones out", {"op": "assign_first", "select": {"colour": c1}, "dest": TABLE}),
        ("uncoverable", "put them all in the tray by number instead, lowest on the left",
         {"op": "replace", "rule": {"assign": [], "tray": {"select": {"all": True}, "order": {"by": "number", "desc": False}}}}),
        ("uncoverable", "just put everything in the left bin", {"op": "replace", "rule": {"assign": [[{"all": True}, "left_bin"]], "tray": None}}),
    ]
    return text, {"stickers": dict(sc.stickers)}, sc, pool


def _colour_to_bin(rng):
    sc = _scene(rng, _colours(rng, rng.randint(4, 8), rng.randint(2, 3)))
    c = rng.choice(sc.colours())
    c2 = rng.choice([x for x in sc.colours() if x != c])
    b = rng.choice(BINS)
    text = f"Put all the {c} blocks in the {_bin_words(b)} bin and leave the rest where they are."
    pool = [
        ("coverable", f"use the {_bin_words(_other_bin(b))} bin instead", {"op": "set_bin", "bin": _other_bin(b)}),
        ("coverable", "actually, the other bin", {"op": "set_bin", "bin": _other_bin(b)}),
        ("uncoverable", f"do the {c2} ones instead of the {c} ones", {"op": "set_colour", "from": c, "to": c2}),
        ("uncoverable", f"put the {c2} ones in the {_bin_words(_other_bin(b))} bin too",
         {"op": "assign_first", "select": {"colour": c2}, "dest": _other_bin(b)}),
    ]
    return text, {"colour": c, "bin": b}, sc, pool


def _all_except(rng):
    n = rng.randint(4, 8)
    g = rng.choice(PALETTE)
    rest = [rng.choice([p for p in PALETTE if p != g]) for _ in range(n - 1)]
    rest[0], rest[1] = rng.sample([p for p in PALETTE if p != g], 2)   # at least two other colours
    cols = rest + [g]
    rng.shuffle(cols)
    sc = _scene(rng, cols)
    c = rng.choice(_excludable(sc, avoid=(g,)))
    text = f"Put everything except the {g} one into the tray, in any order."
    pool = [
        ("coverable", f"the {g} one can go in too", {"op": "include_all"}),
        ("coverable", f"actually, include the {g} one as well", {"op": "include_all"}),
        ("uncoverable", "and line them up by number, lowest on the left", {"op": "tray_order", "order": {"by": "number", "desc": False}}),
        ("uncoverable", f"leave the {c} ones out too", {"op": "assign_first", "select": {"colour": c}, "dest": TABLE}),
        ("uncoverable", f"put the {g} one in the left bin", {"op": "assign_first", "select": {"colour": g}, "dest": "left_bin"}),
    ]
    return text, {"except": g}, sc, pool


def _alternate(rng):
    n = rng.choice([4, 6, 8])
    c1, c2 = rng.sample(PALETTE, 2)
    cols = [c1] * (n // 2) + [c2] * (n // 2)
    rng.shuffle(cols)
    sc = _scene(rng, cols)
    text = f"Line them up in the tray alternating {c1} and {c2}, starting with {c1} on the far left."
    pool = [
        ("coverable", f"start with {c2} instead", {"op": "flip"}),
        ("coverable", "the other colour first", {"op": "flip"}),
        ("uncoverable", f"use the bins instead: {c1} in the left bin, {c2} in the right",
         {"op": "replace", "rule": {"assign": [[{"colour": c1}, "left_bin"], [{"colour": c2}, "right_bin"]], "tray": None}}),
        ("uncoverable", "actually, just line them up by number, lowest on the left",
         {"op": "replace", "rule": {"assign": [], "tray": {"select": {"all": True}, "order": {"by": "number", "desc": False}}}}),
    ]
    return text, {"first": c1, "second": c2}, sc, pool


def _parity_bins(rng):
    sc = _scene(rng, _colours(rng, rng.randint(4, 8), rng.randint(2, 3)), need_both_parities=True)
    even = rng.choice(BINS)
    c = rng.choice(_excludable(sc))
    text = f"Put the even-numbered blocks in the {_bin_words(even)} bin and the odd-numbered ones in the {_bin_words(_other_bin(even))} bin."
    pool = [
        ("coverable", "swap the bins", {"op": "swap_bins"}),
        ("coverable", f"odd ones on the {_bin_words(even)} instead", {"op": "swap_bins"}),
        ("uncoverable", f"leave the {c} ones out", {"op": "assign_first", "select": {"colour": c}, "dest": TABLE}),
        ("uncoverable", "leave the odd ones on the table", {"op": "assign_first", "select": {"parity": "odd"}, "dest": TABLE}),
    ]
    return text, {"even_bin": even}, sc, pool


def make_task(rng: random.Random, family: str, tid: str, n_corrections: int = 5) -> Task:
    if family in ("sort_number", "sort_size"):
        text, params, sc, pool = _sort_task(rng, family)
    else:
        text, params, sc, pool = {"group_colour": _group_colour, "colour_to_bin": _colour_to_bin, "all_except": _all_except,
                                  "alternate": _alternate, "parity_bins": _parity_bins}[family](rng)
    by = {k: [p for p in pool if p[0] == k] for k in CATEGORIES}
    by["chatter"] = [("chatter", t, {"op": "none"}) for t in CHATTER]
    by["pause"] = [("pause", t, {"op": "none"}) for t in PAUSE]
    # ~2 coverable, ~2 uncoverable, 1 other (restate where the family has one, else chatter or pause)
    n_cov = min(len(by["coverable"]), max(1, round(n_corrections * 0.4)))
    n_unc = min(len(by["uncoverable"]), max(1, round(n_corrections * 0.4)))
    picks = rng.sample(by["coverable"], n_cov) + rng.sample(by["uncoverable"], n_unc)
    others = by["restate"] + by["chatter"] + by["pause"]
    weights = [2 if o[0] == "restate" else 1 for o in others]
    while len(picks) < n_corrections:
        o = rng.choices(others, weights=weights)[0]
        if o not in picks:
            picks.append(o)
    corrs = [Correction(f"{tid}_c{i}", t, cat, op) for i, (cat, t, op) in enumerate(picks[:n_corrections])]
    return Task(tid, family, text, params, sc, corrs)


def make_tasks(n_tasks: int, n_corrections: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [make_task(rng, FAMILIES[i % len(FAMILIES)], f"t{i:02d}_{FAMILIES[i % len(FAMILIES)]}", n_corrections) for i in range(n_tasks)]
