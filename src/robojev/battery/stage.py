"""The person and the perception in front of the robot server, for one episode.

    Perception   a RobotServer wrapped around the stub or the sim (mocks.ScriptedPerception, so every
                 tool call and arm mode is logged with the wall time): the brain's view of the world.
                 Letters not on the table are HIDDEN here (a cube the sim cannot delete is parked out
                 of reach and left out of world()), hands are laid over it, and every change the
                 person makes is announced with a scene_change, as a perception process would.
    Director     plays the episode's events: polls the true world, fires each event when its When
                 holds (or its fallback time comes), moves cubes, shows hands, speaks lines, presses
                 STOP. Records what it did and when (`fired`).
    BatteryUser  the brain's UserChannel: lines the Director speaks, STOP, and an answer to any
                 question the brain asks (the episode's `reply`, a moment later).

The Director reads the TRUE world (the server's own world(), hidden cubes included) to judge
conditions and pick cubes; the brain never sees it.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from robojev.brain.loop import UserChannel
from robojev.brain.mocks import ScriptedPerception
from robojev.protocol import RobotEvent, RobotServer, WorldState

from .generate import Episode, Event, When
from .scenes import Layout

FAR = (0.75, -0.40)                   # where a taken cube goes: off every arm's workspace, clear of scene.PARK


class Perception(ScriptedPerception):
    """ScriptedPerception plus hidden objects. `hide`/`show` change what world() reports and announce
    it; `announce` reports a change the person made to a visible cube."""

    def __init__(self, server: RobotServer, hidden: frozenset[str] | set[str] = frozenset(), arm: str = "arm_0"):
        super().__init__(server, arm)
        self.hidden: set[str] = set(hidden)

    def world(self) -> WorldState:
        w = super().world()
        if self.hidden:
            with self._lock:
                hidden = set(self.hidden)
            w.objects = {k: v for k, v in w.objects.items() if k not in hidden}
        return w

    def truth(self) -> WorldState:
        """The server's own world: hidden cubes included, no scripted hands."""
        return self.inner.world()

    def hide(self, oid: str, text: str) -> None:
        with self._lock:
            self.hidden.add(oid)
        self.announce(text)

    def show(self, oid: str, text: str) -> None:
        with self._lock:
            self.hidden.discard(oid)
        self.announce(text)

    def announce(self, text: str) -> None:
        self._emit(RobotEvent(self.inner.world().t, "scene_change", text))


@dataclass
class Fired:
    """One thing the Director did: `what` is text | go_on | taken | returned | added | nudged |
    placed | skipped | hand | stop, at wall time `t` (time.monotonic) and `t_s` on the episode clock."""
    t: float
    t_s: float
    index: int
    kind: str
    what: str
    text: str | None = None
    object: str | None = None
    note: str | None = None


class BatteryUser(UserChannel):
    """The brain's user: speaks what the Director hands it, presses STOP, answers questions with the
    episode's reply after `reply_after_s`."""

    def __init__(self, reply: str, reply_after_s: float = 0.8):
        self.reply, self.reply_after_s = reply, reply_after_s
        self.asked: list[tuple[float, str]] = []
        self.said: list[tuple[float, str]] = []
        self.on_reply: Callable[[str], None] | None = None
        self._say: Callable[[str], None] | None = None
        self._stop: Callable[[], None] | None = None
        self._timers: list[threading.Timer] = []

    def attach(self, say: Callable[[str], None], press_stop: Callable[[], None]) -> None:
        self._say, self._stop = say, press_stop

    @property
    def attached(self) -> bool:
        return self._say is not None

    def say(self, text: str) -> None:
        if self._say is None:
            raise RuntimeError("the brain has not attached the user channel yet")
        self.said.append((time.monotonic(), text))
        self._say(text)

    def press_stop(self) -> None:
        if self._stop is None:
            raise RuntimeError("the brain has not attached the user channel yet")
        self._stop()

    def ask(self, question: str) -> None:
        self.asked.append((time.monotonic(), question))
        t = threading.Timer(self.reply_after_s, self._answer)
        t.daemon = True
        self._timers.append(t)
        t.start()

    def _answer(self) -> None:
        if self._say is None:
            return
        self.say(self.reply)
        if self.on_reply is not None:
            self.on_reply(self.reply)

    def close(self) -> None:
        for t in self._timers:
            t.cancel()


def placed(word: str, world: WorldState, positions: list[str]) -> int:
    """How many positions of `word` read right in `world`: position i holds a cube labelled word[i]."""
    occ = {o.where: o.label for o in world.objects.values()}
    return sum(1 for i, ch in enumerate(word) if i < len(positions) and occ.get(positions[i]) == ch)


def carrying(world: WorldState, server: RobotServer) -> bool:
    """Some arm is carrying a cube: holding it, with its skill past the grasp (phase "carry" where the
    server reports phases; the stub's skill is unnamed on the arm, so holding is enough there)."""
    for a in world.arms.values():
        if a.holding is None:
            continue
        if a.skill is None:
            return True
        try:
            return server.status(a.skill).phase == "carry"
        except KeyError:
            return True
    return False


class Director:
    """Plays `ep` on `perception` (over the true world), speaking through `user`. Start it once the
    brain has attached the user channel; it runs on its own thread until the episode is over."""

    def __init__(self, ep: Episode, lay: Layout, perception: Perception, user: BatteryUser, pick_s: float,
                 poll_s: float = 0.02):
        self.ep, self.lay, self.p, self.user, self.pick_s, self.poll_s = ep, lay, perception, user, pick_s, poll_s
        self.word = ep.word                   # the word in force
        self.fired: list[Fired] = []
        self._pending: list[tuple[float, Callable[[], None]]] = []    # (fire at wall time, action)
        self._i = 0
        self._t0 = 0.0
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.origin = {b.name: b.xy for b in lay.blocks}
        self.labels = {b.name: b.label for b in lay.blocks}
        user.on_reply = lambda text: self._record(-1, "reply", "text", text=text)

    # ---------------------------------------------------------------- lifecycle
    def start(self, t0: float | None = None) -> "Director":
        self._t0 = time.monotonic() if t0 is None else t0
        self._thread = threading.Thread(target=self._run, name="director", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def now(self) -> float:
        return time.monotonic() - self._t0

    @property
    def done(self) -> bool:
        return self._i >= len(self.ep.events) and not self._pending

    # ---------------------------------------------------------------- the loop
    def _run(self) -> None:
        while not self._halt.wait(self.poll_s):
            if not self.user.attached:
                continue
            now = time.monotonic()
            with self._lock:
                due = [a for t, a in self._pending if t <= now]
                self._pending = [(t, a) for t, a in self._pending if t > now]
            for a in due:
                a()
            if self._i < len(self.ep.events):
                ev = self.ep.events[self._i]
                if self._holds(ev.when):
                    self._i += 1
                    self._fire(ev)

    def _holds(self, w: When) -> bool:
        t = self.now()
        if t >= w.by_picks * self.pick_s:
            return True
        if w.kind == "at":
            return t >= w.k * self.pick_s
        world = self.p.truth()
        n = placed(self.word, world, self.lay.positions)
        if w.kind == "placed":
            return n >= w.k
        return n >= w.k and carrying(world, self.p.inner)

    def _later(self, after_s: float, action: Callable[[], None]) -> None:
        with self._lock:
            self._pending.append((time.monotonic() + after_s, action))

    def _record(self, index: int, kind: str, what: str, **kw) -> Fired:
        f = Fired(time.monotonic(), self.now(), index, kind, what, **kw)
        self.fired.append(f)
        return f

    # ---------------------------------------------------------------- firing
    def _fire(self, ev: Event) -> None:
        k = ev.kind
        if k == "new_word":
            self.word = ev.word
            self.user.say(ev.text)
            self._record(ev.index, k, "text", text=ev.text)
        elif k == "chatter":
            self.user.say(ev.text)
            self._record(ev.index, k, "text", text=ev.text)
        elif k == "wait_go_on":
            self.user.say(ev.text)
            self._record(ev.index, k, "text", text=ev.text)
            self._later(ev.go_on_after_s, lambda: self._go_on(ev))
        elif k == "stop":
            self.user.press_stop()
            self._record(ev.index, k, "stop")
        elif k == "hand_near":
            self.p.hand_now(ev.hand_for_s, ev.hand_dist_m)
            self._record(ev.index, k, "hand")
            self._remark(ev)
        elif k == "letter_taken":
            self._take(ev)
        elif k == "letter_added":
            self._add(ev)
        elif k == "letter_nudged":
            self._nudge(ev)
        elif k == "letter_placed":
            self._place(ev)

    def _go_on(self, ev: Event) -> None:
        self.user.say(ev.go_on_text)
        self._record(ev.index, ev.kind, "go_on", text=ev.go_on_text)

    def _remark(self, ev: Event) -> None:
        if ev.text:
            self.user.say(ev.text)
            self._record(ev.index, ev.kind, "text", text=ev.text)

    def _free_cube(self, label: str, world: WorldState) -> str | None:
        """A visible cube labelled `label` that is on the table (not in a position, not in a gripper)."""
        held = {a.holding for a in world.arms.values() if a.holding}
        cands = [o for o in world.objects.values() if o.label == label and o.id not in self.p.hidden
                 and o.where == "table" and o.id not in held]
        return cands[0].id if cands else None

    def _any_free(self, world: WorldState) -> str | None:
        held = {a.holding for a in world.arms.values() if a.holding}
        cands = [o for o in world.objects.values() if o.id not in self.p.hidden and o.where == "table" and o.id not in held]
        return cands[0].id if cands else None

    def _take(self, ev: Event) -> None:
        world = self.p.truth()
        oid = self._free_cube(ev.label, world) or self._any_free(world)
        if oid is None:
            self._record(ev.index, ev.kind, "skipped", note="no free cube to take")
            return
        label = self.labels[oid]
        self.p.inner.move(oid, *FAR)
        self.p.hide(oid, f"a person took {world.objects[oid].name} off the table")
        self._record(ev.index, ev.kind, "taken", object=oid)
        self._remark(ev)
        need = self.word.count(label)
        have = sum(1 for o in self.p.truth().objects.values() if o.label == label and o.id not in self.p.hidden)
        if ev.returns == "always" or have < need:
            self._later(ev.return_after_picks * self.pick_s, lambda: self._give(ev, oid, "returned"))

    def _give(self, ev: Event, oid: str, what: str) -> None:
        xy = self.origin[oid] if what == "returned" else self._letter_spot()
        if xy is None:
            self._record(ev.index, ev.kind, "skipped", object=oid, note="no free letter spot")
            return
        self.p.inner.move(oid, *xy)
        name = self.p.truth().objects[oid].name
        self.p.show(oid, f"a person put {name} on the table" + (" again" if what == "returned" else ""))
        self._record(ev.index, ev.kind, what, object=oid)
        text = ev.return_text if what == "returned" else ev.text
        if text:
            self.user.say(text)
            self._record(ev.index, ev.kind, "text", text=text)

    def _letter_spot(self) -> tuple[float, float] | None:
        """A letter spot of the layout with no visible cube within 6 cm (an empty one, or one a
        placed letter left)."""
        world = self.p.truth()
        taken = [(o.x, o.y) for o in world.objects.values() if o.id not in self.p.hidden]
        for x, y in self.lay.spots:
            if all((x - ox) ** 2 + (y - oy) ** 2 > 0.06 ** 2 for ox, oy in taken):
                return (x, y)
        return None

    def _add(self, ev: Event) -> None:
        oid = next((o for o in sorted(self.lay.parked) if o in self.p.hidden and self.labels[o] == ev.label), None) or \
            next((o for o in sorted(self.lay.parked) if o in self.p.hidden), None)
        if oid is None:
            self._record(ev.index, ev.kind, "skipped", note="no parked cube to add")
            return
        self._give(ev, oid, "added")

    def _nudge(self, ev: Event) -> None:
        world = self.p.truth()
        oid = self._free_cube(ev.label, world) or self._any_free(world)
        if oid is None:
            self._record(ev.index, ev.kind, "skipped", note="no free cube to nudge")
            return
        o = world.objects[oid]
        rows = sorted({b.xy[0] for b in self.lay.blocks if b.name not in self.lay.parked})
        mid = (rows[0] + rows[-1]) / 2
        dx = -ev.nudge_m if o.x <= mid else ev.nudge_m        # away from the other row, into clear table
        self.p.inner.move(oid, o.x + dx, o.y)
        self.p.announce(f"a person nudged {o.name}")
        self._record(ev.index, ev.kind, "nudged", object=oid, note=f"{dx * 100:+.0f} cm in x")
        self._remark(ev)

    def _place(self, ev: Event) -> None:
        world = self.p.truth()
        j = min(ev.position if ev.position is not None else 0, len(self.word) - 1)
        slot = self.lay.positions[j]
        occupied = any(o.where == slot for o in world.objects.values())
        oid = None if occupied else self._free_cube(self.word[j], world)
        if oid is None:
            self._record(ev.index, ev.kind, "skipped", note="position taken" if occupied else f"no free {self.word[j]}")
            return
        p = world.places[slot]
        self.p.inner.move(oid, p.x, p.y)
        self.p.announce(f"a person put {world.objects[oid].name} into {p.name}")
        self._record(ev.index, ev.kind, "placed", object=oid, note=slot)
        self._remark(ev)
