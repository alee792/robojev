"""The scripted person: moves blocks, takes a placed block back out, reaches into the arm's path and
withdraws, holds out a hand for a handover, and types corrections and chatter. Implements core's
UserChannel (typed text, questions back, the STOP button).

A Beat fires once, when its trigger holds: a sim time, or a condition on the true world (the arm's
posted intent, what is in the tray), so every arm meets the same disturbance at the same point of the
task. `do` changes the world and may return text the person types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core.data import dxy
from .world import TRAVEL_Z, PersonHand, SimWorld


@dataclass
class Beat:
    name: str
    when: Callable[[SimWorld, int], bool]
    do: Callable[[SimWorld, int], str | None]
    disturbs: bool = True           # moves the world (a disturbance) rather than only talking
    fired_t: int | None = None
    moved: list = field(default_factory=list)
    text: str | None = None


class Person:
    def __init__(self, beats: list[Beat], answer: Callable[[str], str] | None = None, stop_at: int | None = None,
                 answer_delay: int = 20):
        self.beats = beats
        self.answer = answer
        self.stop_at = stop_at
        self.answer_delay = answer_delay
        self.outbox: list[tuple[int, str]] = []
        self.asked: list[tuple[int, str]] = []
        self.fired_now: list[Beat] = []

    # ---------------------------------------------------------------- acting
    def act(self, w: SimWorld, t: int):
        self.fired_now = []
        for b in self.beats:
            if b.fired_t is not None or not b.when(w, t):
                continue
            before = {k: (v.x, v.y, v.where) for k, v in w.blocks.items()}
            b.fired_t = t
            out = b.do(w, t)
            w._restack()
            b.moved = [k for k, v in w.blocks.items() if (v.x, v.y, v.where) != before[k]]
            if isinstance(out, str):
                b.text = out
                self.outbox.append((t, out))
            self.fired_now.append(b)

    # ---------------------------------------------------------------- UserChannel
    def poll(self, t: int) -> list[str]:
        due = [txt for (tt, txt) in self.outbox if tt <= t]
        self.outbox = [(tt, txt) for (tt, txt) in self.outbox if tt > t]
        return due

    def ask(self, t: int, question: str) -> None:
        self.asked.append((t, question))
        ans = self.answer(question) if self.answer else "just carry on"
        self.outbox.append((t + self.answer_delay, ans))

    def stop_pressed(self, t: int) -> bool:
        return self.stop_at is not None and t >= self.stop_at


# ---------------------------------------------------------------- triggers


def intent(w: SimWorld) -> dict:
    return w.arm.intent or {}


def placed(w: SimWorld) -> int:
    return sum(1 for b in w.blocks.values() if b.where.startswith(("tray_slot", "on:")) or b.where.endswith("_bin"))


def carrying(w: SimWorld, min_placed: int = 0, min_left_cm: float = 0.0) -> bool:
    i = intent(w)
    if not (w.arm.holding and i.get("phase") == "carry" and placed(w) >= min_placed and i.get("dest")):
        return False
    return dxy((w.arm.x, w.arm.y), i["dest"]) >= min_left_cm


def approaching(w: SimWorld, min_placed: int = 0, within_cm: float = 10.0, skill: str | None = None) -> bool:
    i = intent(w)
    if w.arm.holding or i.get("phase") != "approach" or placed(w) < min_placed or (skill and i.get("skill") != skill):
        return False
    b = w.blocks.get(i.get("object"))
    return b is not None and dxy((w.arm.x, w.arm.y), (b.x, b.y)) < within_cm


def at_time(s: float) -> Callable:
    return lambda w, t: t >= int(s * 10)


def after(beat: Beat, s: float) -> Callable:
    return lambda w, t: beat.fired_t is not None and t >= beat.fired_t + int(s * 10)


# ---------------------------------------------------------------- actions


def say(text: str) -> Callable:
    return lambda w, t: text


def slide_block(which: Callable[[SimWorld], str], dx: float = 14.0) -> Callable:
    """Slide a block on the table about dx cm sideways, to a free spot."""
    def do(w, t):
        b = w.blocks[which(w)]
        x, y = w.free_spot(near=(b.x + dx if b.x + dx < 90 else b.x - dx, b.y), exclude=(b.id,))
        b.x, b.y, b.where = x, y, "table"
    return do


def take_out(place_id: str) -> Callable:
    """Take the block out of a place and put it on the table."""
    def do(w, t):
        bid = next((b.id for b in w.blocks.values() if b.where == place_id), None)
        if bid:
            x, y = w.free_spot(near=(45.0, 30.0), exclude=(bid,))
            b = w.blocks[bid]
            b.x, b.y, b.where = x, y, "table"
    return do


def put_into(bid: str, place_id: str) -> Callable:
    def do(w, t):
        b = w.blocks[bid]
        if b.where in ("gripper", "person"):
            return None
        occ = next((o for o in w.blocks.values() if o.where == place_id and o.id != bid), None)
        if occ is None:
            x, y = w.place_xy(place_id, bid)
            b.x, b.y, b.where = x, y, place_id
    return do


def reach_into_path(w: SimWorld, t: int):
    """A hand comes in from the side toward a point 12 cm ahead of the gripper on its carry path,
    stays 3 s and withdraws (E12's reach, with the v2 hand)."""
    a = w.arm
    tx, ty = intent(w)["dest"]
    L = dxy((a.x, a.y), (tx, ty)) or 1.0
    ux, uy = (tx - a.x) / L, (ty - a.y) / L
    px, py = a.x + 12 * ux, a.y + 12 * uy
    for sgn in (1, -1):
        sx, sy = px - sgn * 25 * uy, py + sgn * 25 * ux
        if 2 <= sx <= 98 and 2 <= sy <= 70:
            break
    w.hand = PersonHand(start=(sx, sy), target=(px, py), stay_ticks=30, speed=4.0, z=TRAVEL_Z, purpose="reach")


def hold_out_hand(x: float = 70.0, y: float = 48.0, stay_s: float = 40.0) -> Callable:
    def do(w, t):
        w.hand = PersonHand(start=(x, 72.0), target=(x, y), stay_ticks=int(stay_s * 10), speed=3.0, z=TRAVEL_Z, purpose="take")
    return do


def slide_next_to(fid: str, side_cm: float = 3.2) -> Callable:
    """Slide block `fid` right up against the block the arm is going for."""
    def do(w, t):
        tb = w.blocks[intent(w)["object"]]
        s = side_cm if tb.x < 80 else -side_cm
        f = w.blocks[fid]
        f.x, f.y, f.where = tb.x + s, tb.y, "table"
    return do
