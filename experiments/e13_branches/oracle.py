"""E13 EVALUATION ORACLE — used only to score results and to drive the mock backends.

This module is hand-written task knowledge: what each task sentence and each correction means, and
therefore which final arrangements are correct. It is deliberately kept apart from the system under
test. `plan.py`, `planner.py` and `router.py` must never import it (a test checks this); in the real
system this knowledge exists only in the LLM's plan.

A task's meaning is a `rule`:
  {"assign": [[selector, destination], ...],        first match wins; destination is a bin or the table
   "tray":   {"select": selector, "order": order} | None}
Blocks not assigned and not selected for the tray stay on the table. Orders:
  {"by": "number"|"size", "desc": bool}   left to right; gaps between tray blocks are allowed
  {"alternate": [c1, c2]}                 colours alternate from the leftmost tray block, c1 first
  {"any": True}                           any order
Selectors: {"all": True}, {"colour": c}, {"not_colour": c}, {"parity": "even"|"odd"}.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import zip_longest
from typing import Callable

from .world import BINS, TABLE, Block, Correction, Scene, Task, slot


def _sel(s: dict, b: Block) -> bool:
    if s.get("all"):
        return True
    if "colour" in s:
        return b.colour == s["colour"]
    if "not_colour" in s:
        return b.colour != s["not_colour"]
    if "parity" in s:
        return (b.number % 2 == 0) == (s["parity"] == "even")
    raise ValueError(s)


def task_rule(task: Task) -> dict:
    p, f = task.params, task.family
    if f in ("sort_number", "sort_size"):
        return {"assign": [], "tray": {"select": {"all": True}, "order": {"by": p["key"], "desc": p["desc"]}}}
    if f == "group_colour":
        return {"assign": [[{"colour": c}, b] for b, c in p["stickers"].items()], "tray": None}
    if f == "colour_to_bin":
        return {"assign": [[{"colour": p["colour"]}, p["bin"]]], "tray": None}
    if f == "all_except":
        return {"assign": [], "tray": {"select": {"not_colour": p["except"]}, "order": {"any": True}}}
    if f == "alternate":
        return {"assign": [], "tray": {"select": {"all": True}, "order": {"alternate": [p["first"], p["second"]]}}}
    if f == "parity_bins":
        other = BINS[1] if p["even_bin"] == BINS[0] else BINS[0]
        return {"assign": [[{"parity": "even"}, p["even_bin"]], [{"parity": "odd"}, other]], "tray": None}
    raise ValueError(f)


def _swap(d: str) -> str:
    return {BINS[0]: BINS[1], BINS[1]: BINS[0]}.get(d, d)


def apply(rule: dict, op: dict) -> dict:
    r = copy.deepcopy(rule)
    o = op["op"]
    if o == "none":
        pass
    elif o == "flip":
        order = r["tray"]["order"]
        if "by" in order:
            order["desc"] = not order["desc"]
        elif "alternate" in order:
            order["alternate"] = order["alternate"][::-1]
        else:
            raise ValueError("flip on an unordered tray")
    elif o == "set_order":
        r["tray"]["order"]["desc"] = op["desc"]
    elif o == "swap_bins":
        r["assign"] = [[s, _swap(d)] for s, d in r["assign"]]
    elif o == "set_bin":
        r["assign"] = [[s, op["bin"] if d in BINS else d] for s, d in r["assign"]]
    elif o == "set_colour":
        r["assign"] = [[{"colour": op["to"]} if s.get("colour") == op["from"] else s, d] for s, d in r["assign"]]
    elif o == "assign_first":
        r["assign"].insert(0, [op["select"], op["dest"]])
    elif o == "include_all":
        r["tray"]["select"] = {"all": True}
    elif o == "tray_order":
        r["tray"]["order"] = op["order"]
    elif o == "replace":
        r = copy.deepcopy(op["rule"])
    else:
        raise ValueError(o)
    return r


def category(rule: dict, b: Block) -> str:
    """'tray', a bin name, or TABLE."""
    for s, d in rule["assign"]:
        if _sel(s, b):
            return d
    if rule["tray"] and _sel(rule["tray"]["select"], b):
        return "tray"
    return TABLE


def _dest_category(d: str) -> str:
    return "tray" if d.startswith("tray_slot_") else d


def _slot_index(d: str) -> int:
    return int(d.rsplit("_", 1)[1])


def _order_ok(order: dict, blocks: list[Block]) -> bool:
    if order.get("any"):
        return True
    if "by" in order:
        vals = [getattr(b, "number" if order["by"] == "number" else "size_cm") for b in blocks]
        pairs = list(zip(vals, vals[1:]))
        return all(a > b for a, b in pairs) if order["desc"] else all(a < b for a, b in pairs)
    c1, c2 = order["alternate"]
    return all(b.colour == (c1 if i % 2 == 0 else c2) for i, b in enumerate(blocks))


def _canonical_tray(order: dict, blocks: list[Block]) -> list[Block]:
    if order.get("any"):
        return sorted(blocks, key=lambda b: b.number)
    if "by" in order:
        return sorted(blocks, key=lambda b: b.number if order["by"] == "number" else b.size_cm, reverse=order["desc"])
    c1, c2 = order["alternate"]
    a = sorted((b for b in blocks if b.colour == c1), key=lambda b: b.number)
    z = sorted((b for b in blocks if b.colour == c2), key=lambda b: b.number)
    out = [b for pair in zip_longest(a, z) for b in pair if b is not None]
    return out + sorted((b for b in blocks if b.colour not in (c1, c2)), key=lambda b: b.number)


@dataclass
class Target:
    """The correct final arrangement(s): `check(goal)` says whether a goal {block: destination} is one."""
    canonical: dict[str, str]
    check: Callable[[dict[str, str]], bool]


def target_for_rule(rule: dict, scene: Scene) -> Target:
    cats = {b.id: category(rule, b) for b in scene.blocks}
    tray_blocks = [b for b in scene.blocks if cats[b.id] == "tray"]
    order = rule["tray"]["order"] if rule["tray"] else {"any": True}
    canonical = {b.id: cats[b.id] for b in scene.blocks}
    for i, b in enumerate(_canonical_tray(order, tray_blocks), 1):
        canonical[b.id] = slot(i)
    by_id = {b.id: b for b in scene.blocks}

    def check(goal: dict[str, str]) -> bool:
        if set(goal) != set(cats):
            return False
        if any(_dest_category(goal[k]) != cats[k] for k in cats):
            return False
        slots = [goal[k] for k in cats if cats[k] == "tray"]
        if len(set(slots)) != len(slots):
            return False
        row = [by_id[k] for k in sorted((k for k in cats if cats[k] == "tray"), key=lambda k: _slot_index(goal[k]))]
        return _order_ok(order, row)

    assert check(canonical), (rule, canonical)
    return Target(canonical, check)


def target(task: Task, correction: Correction | None = None) -> Target:
    """Correct arrangement for the task as given, or after `correction` (applied to the task as given)."""
    rule = task_rule(task)
    if correction is not None:
        rule = apply(rule, correction.op)
    return target_for_rule(rule, task.scene)


# ---------------------------------------------------------------- the mock planner's parameter (one per family)



def natural_parameter(task: Task) -> dict:
    """The one parameter a sensible planner would expose: name, about, (default value, meaning), (other value, meaning), op.
    Used by the mock LLM only."""
    p, f = task.params, task.family
    if f in ("sort_number", "sort_size"):
        lo, hi = ("lowest_number", "highest_number") if f == "sort_number" else ("smallest", "biggest")
        vals = [(f"{hi}_on_left", f"{hi.replace('_', ' ')} in the leftmost slot"), (f"{lo}_on_left", f"{lo.replace('_', ' ')} in the leftmost slot")]
        if not p["desc"]:
            vals.reverse()
        return {"name": "line_direction", "about": "which end of the tray the line starts from", "values": vals, "op": {"op": "flip"}}
    if f == "group_colour":
        return {"name": "bin_matching", "about": "which bin each colour goes in",
                "values": [("matching_sticker", "each block in the bin whose sticker is its colour"), ("other_bin", "each block in the bin whose sticker is NOT its colour")],
                "op": {"op": "swap_bins"}}
    if f == "colour_to_bin":
        other = BINS[1] if p["bin"] == BINS[0] else BINS[0]
        return {"name": "target_bin", "about": f"which bin the {p['colour']} blocks go in",
                "values": [(p["bin"], f"the {p['bin'].replace('_', ' ')}"), (other, f"the {other.replace('_', ' ')}")],
                "op": {"op": "set_bin", "bin": other}}
    if f == "all_except":
        g = p["except"]
        return {"name": f"{g}_block", "about": f"whether the {g} block goes in the tray",
                "values": [("left_out", f"the {g} block stays on the table"), ("included", f"the {g} block goes in the tray too")],
                "op": {"op": "include_all"}}
    if f == "alternate":
        c1, c2 = p["first"], p["second"]
        return {"name": "first_colour", "about": "which colour is in the leftmost slot",
                "values": [(f"{c1}_first", f"{c1} in the leftmost slot, then alternating"), (f"{c2}_first", f"{c2} in the leftmost slot, then alternating")],
                "op": {"op": "flip"}}
    if f == "parity_bins":
        other = BINS[1] if p["even_bin"] == BINS[0] else BINS[0]
        return {"name": "even_numbers_bin", "about": "which bin the even-numbered blocks go in (odd ones go in the other)",
                "values": [(p["even_bin"], f"even numbers in the {p['even_bin'].replace('_', ' ')}"), (other, f"even numbers in the {other.replace('_', ' ')}")],
                "op": {"op": "swap_bins"}}
    raise ValueError(f)
