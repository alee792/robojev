"""Offline stand-ins for the brain's models and user: no API keys, no network. The Brain never
imports this module; tests and spike runs do.

e12v2's mock Jev and mock LLM are reused as they are. They read an evaluation oracle (the true world,
a goal per task version, the truth about each line the user types), so the adapters here give them
that from the stub server, and make them behave as models would behind this brain:
  - StubTruth      the stub's world() as e12v2's cm WorldState (`.truth()`, what the mocks read).
  - TimedJev/LLM   the mocks report a latency but return at once; these sleep it, in the worker
                   thread, so a real clock sees decisions take ~150 ms and plans seconds.
  - CatalogLLM     e12v2's mock LLM writes e12v2's step names and fields ("move_object", no push
                   distance); a real LLM given the brain's schema writes the catalog's. This renames
                   steps and fills fields the schema has and the mock lacks with null.
  - ScriptedLLM    fixed answers in turn, for robots the oracle cannot plan for (the push-only arm).
  - ScriptedUser   lines typed at times or when conditions hold; the STOP button as a line.
  - ScriptedPerception  a RobotServer in front of another: everything passes through, every tool
                   call and every arm mode is logged with the wall time, and a scripted hand can be
                   put into world() (a PERCEPTION OVERRIDE, not physics: the MuJoCo server has no
                   person in its scene and reports no hands; on the real robot hands come from
                   perception, in front of the motion stack, which is where this sits).
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .core.combine import Gates, JevDecider
from .core.data import WorldState
from .eval.mocks import MockJev, MockLLM
from .eval.oracle import Goal
from .eval.scenarios import Scenario, Utt

from ..protocol import HandObs, RobotEvent, RobotServer
from .adapt import core_state, geometry
from .steps import to_catalog

PRESS_STOP = "<the STOP button>"      # a ScriptedUser line that presses STOP instead of typing


class StubTruth:
    """Ground truth for e12v2's oracle, read from the server (a stub has no perception error)."""

    def __init__(self, server: RobotServer, arm: str = "arm_0", lookahead_s: float = 0.2):
        spec = next(a for a in server.manifest().arms if a.id == arm)
        self.server, self.arm, self.lookahead_s = server, arm, lookahead_s
        self.geom = geometry(spec, lookahead_s)

    def truth(self) -> WorldState:
        return core_state(self.server.world(), self.arm, self.geom, self.lookahead_s)


def oracle_scenario(versions: Sequence[Goal], utts: dict[str, Utt] | None = None, task: str = "") -> Scenario:
    """e12v2's Scenario, for what its mocks read: goal_for(user messages) and the utterance truths.
    Its world and scripted person belong to e12v2's toy world and are not used here."""
    return Scenario(name="stub", about="", task=task, world=None, person=None, versions=list(versions), utts=dict(utts or {}))


class TimedJev:
    """A DecisionBackend that takes as long as it says it does."""

    def __init__(self, backend):
        self.backend, self.live = backend, False

    def answer(self, state: dict, questions: dict, ref=None) -> dict:
        r = self.backend.answer(state, questions, ref=ref)
        time.sleep(r.get("latency_ms", 0.0) / 1000.0)
        return r


class TimedLLM:
    """An LLMBackend that takes as long as it says it does."""

    def __init__(self, backend):
        self.backend = backend

    def call(self, req, ref=None):
        r = self.backend.call(req, ref=ref)
        time.sleep(r.latency_ms / 1000.0)
        return r


@dataclass
class LLMReply:
    """An LLMBackend result in e12v2's shape."""
    raw: str | None
    latency_ms: float
    in_tok: int = 0
    out_tok: int = 0
    error: str | None = None
    model: str = "scripted"


class CatalogLLM:
    """Puts an e12v2-vocabulary LLM's answers into the catalog vocabulary of the request's schema."""

    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def _step_fields(schema: dict) -> list[str]:
        props = schema["properties"]
        steps = props.get("steps") or props.get("new_steps")
        return list(steps["items"]["properties"])

    def call(self, req, ref=None):
        r = self.backend.call(req, ref=ref)
        if r.error is not None or not r.raw:
            return r
        try:
            out = json.loads(r.raw)
        except json.JSONDecodeError:
            return r
        fields = self._step_fields(req.schema)
        key = "steps" if "steps" in out else "new_steps"
        out[key] = [{f: None for f in fields} | to_catalog(s) for s in out.get(key, [])]
        return LLMReply(json.dumps(out), r.latency_ms, r.in_tok, r.out_tok, None, getattr(r, "model", "mock"))


class ScriptedLLM:
    """Answers in turn (the last one repeats). `requests` records every request received."""

    def __init__(self, answers: Sequence[dict], latency_ms: float = 100.0):
        self.answers, self.latency_ms = list(answers), latency_ms
        self.requests: list = []

    def call(self, req, ref=None):
        self.requests.append(req)
        out = self.answers[min(len(self.requests), len(self.answers)) - 1]
        time.sleep(self.latency_ms / 1000.0)
        return LLMReply(json.dumps(out), self.latency_ms)


def oracle_decider(server: RobotServer, versions: Sequence[Goal], utts: dict[str, Utt] | None = None, *,
                   arm: str = "arm_0", seed: int = 0, lookahead_s: float = 0.2) -> JevDecider:
    """e12v2's real JevDecider (request building, combine(), its gates) over its mock Jev with no
    errors and full confidence (the oracle arm), taking Jev's measured latency."""
    sc, truth = oracle_scenario(versions, utts), StubTruth(server, arm, lookahead_s)
    return JevDecider(TimedJev(MockJev(sc, truth, error=0.0, conf_noise=0.0, seed=seed)), Gates())


def oracle_models(server: RobotServer, versions: Sequence[Goal], utts: dict[str, Utt] | None = None, *,
                  arm: str = "arm_0", llm_median_ms: float = 300.0, seed: int = 0,
                  lookahead_s: float = 0.2) -> tuple[JevDecider, dict]:
    """oracle_decider(), and e12v2's mock LLM for both planner routes, both reading the stub.
    -> (decider, planner backends)."""
    sc, truth = oracle_scenario(versions, utts), StubTruth(server, arm, lookahead_s)
    llm = CatalogLLM(TimedLLM(MockLLM(sc, truth, seed=seed, median_ms=llm_median_ms)))
    decider = oracle_decider(server, versions, utts, arm=arm, seed=seed, lookahead_s=lookahead_s)
    return decider, {"fast_llm": llm, "capable_llm": llm}


@dataclass
class Line:
    """One thing the user does: typed `text` (or PRESS_STOP) at `when` s after attach, or when `when()` holds."""
    when: float | Callable[[], bool]
    text: str


class ScriptedUser:
    """A UserChannel that plays its lines in order, each when its time or condition comes."""

    def __init__(self, lines: Iterable[Line | tuple] = (), poll_s: float = 0.02):
        self.lines = [x if isinstance(x, Line) else Line(*x) for x in lines]
        self.poll_s = poll_s
        self.said: list[tuple[float, str]] = []
        self.asked: list[str] = []
        self._task: asyncio.Task | None = None

    def attach(self, say: Callable[[str], None], press_stop: Callable[[], None]) -> None:
        self._say, self._press_stop, self.t0 = say, press_stop, time.monotonic()
        self._task = asyncio.get_running_loop().create_task(self._play())

    async def _play(self) -> None:
        for line in self.lines:
            while not (line.when() if callable(line.when) else time.monotonic() - self.t0 >= line.when):
                await asyncio.sleep(self.poll_s)
            self.said.append((time.monotonic() - self.t0, line.text))
            if line.text == PRESS_STOP:
                self._press_stop()
            else:
                self._say(line.text)

    def ask(self, question: str) -> None:
        self.asked.append(question)


# ---------------------------------------------------------------- a perception override in front of a server


class ScriptedPerception:
    """A RobotServer wrapped around another, for spike runs and tests on the physics server.

    Pass-through for the whole protocol, plus three things a test wants from the outside:
      - `calls`: every tool call (start/hold/pause/resume/retarget/stop) with the wall time;
      - `history`: the arm's mode, what it holds and every object's (where, z), sampled on a thread
        at `poll_s` and recorded on change, with the wall time;
      - `hand_when(cond, ...)`: a PERCEPTION OVERRIDE. The MuJoCo server has no hands: nothing in
        its scene is a person and its world() reports none. On the real robot a hand is a perception
        output (P3) laid over the robot's own state, in front of the motion stack: that is where this
        sits. The scripted hand is reported in world().hands and announced with a scene_change
        notification, as a perception process would; it is not a body in the physics, so "no contact"
        is judged from `min_hand_dist`, the closest the gripper came to it while it was there.
    Times are time.monotonic(), the clock ScriptedUser and the brain use.
    """

    def __init__(self, server: RobotServer, arm: str = "arm_0", poll_s: float = 0.01):
        self.inner, self.arm, self.poll_s = server, arm, poll_s
        self.calls: list[tuple[float, str, tuple]] = []
        self.history: list[tuple[float, str, str | None, dict]] = []
        self.hand_log: list[tuple[float, str]] = []
        self.min_hand_dist = math.inf
        self._hands: list[HandObs] = []
        self._hand_until: float | None = None
        self._pending: list[tuple[Callable[["ScriptedPerception"], bool], float, float, float, bool]] = []
        self.notify = True
        self._subs: list[Callable[[RobotEvent], None]] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        server.subscribe(self._relay)

    # -- lifecycle (not protocol) -----------------------------------------------------------------------
    def open(self) -> "ScriptedPerception":
        if self._thread is None:
            self._thread = threading.Thread(target=self._poll, name="perception", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.inner.unsubscribe(self._relay)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- the script -------------------------------------------------------------------------------------
    def hand_when(self, cond: Callable[["ScriptedPerception"], bool], for_s: float = 1.5, dist: float = 0.15,
                  speed: float = 0.1, notify: bool = True) -> None:
        """When `cond(self)` first holds, a hand appears `dist` m from the gripper, ahead of it along
        the skill's heading (in the arm's path; toward the far side of the table if nothing is
        heading anywhere), reported reaching toward the gripper at `speed` m/s, and leaves `for_s`
        s later. The hand does not move. `notify=False` sends no scene_change: the brain then finds
        the hand on its own world() poll, as it would behind a server that only updates the resource."""
        with self._lock:
            self._pending.append((cond, for_s, dist, speed, notify))

    def _place_hand(self, for_s: float, dist: float, speed: float) -> None:
        w = self.inner.world()
        a = w.arms[self.arm]
        heading = None
        if a.skill is not None:
            try:
                heading = self.inner.status(a.skill).heading
            except KeyError:
                heading = None
        if heading is not None and math.hypot(heading[0] - a.x, heading[1] - a.y) > 0.02:
            ang = math.atan2(heading[1] - a.y, heading[0] - a.x)
        else:
            ang = math.atan2(-a.y, 0.6 - a.x)
        ux, uy = math.cos(ang), math.sin(ang)
        hand = HandObs(a.x + dist * ux, a.y + dist * uy, a.z, -speed * ux, -speed * uy)
        with self._lock:
            self._hands = [hand]
            self._hand_until = time.monotonic() + for_s
            self.min_hand_dist = dist
        self.hand_log.append((time.monotonic(), "appears"))
        if self.notify:
            self._emit(RobotEvent(w.t, "scene_change", f"a person's hand appeared {dist:.2f} m from the gripper"))

    def _remove_hand(self) -> None:
        with self._lock:
            self._hands, self._hand_until = [], None
        self.hand_log.append((time.monotonic(), "leaves"))
        if self.notify:
            self._emit(RobotEvent(self.inner.world().t, "scene_change", "the person's hand has gone out of view"))

    def _poll(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            w = self.inner.world()
            a = w.arms[self.arm]
            objs = {o.id: (o.where, round(o.z, 4)) for o in w.objects.values()}
            rec = (a.mode, a.holding, objs)
            if not self.history or self.history[-1][1:] != rec:
                self.history.append((now, *rec))
            with self._lock:
                hands, until, pending = list(self._hands), self._hand_until, list(self._pending)
            if hands:
                h = hands[0]
                self.min_hand_dist = min(self.min_hand_dist, math.dist((h.x, h.y, h.z), (a.x, a.y, a.z)))
                if until is not None and now >= until:
                    self._remove_hand()
            elif pending:
                cond, for_s, dist, speed, notify = pending[0]
                if cond(self):
                    with self._lock:
                        self._pending.pop(0)
                        self.notify = notify
                    self._place_hand(for_s, dist, speed)
            time.sleep(self.poll_s)

    # -- notifications ----------------------------------------------------------------------------------
    def _relay(self, ev: RobotEvent) -> None:
        self._emit(ev)

    def _emit(self, ev: RobotEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            cb(ev)

    def subscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    def unsubscribe(self, callback: Callable[[RobotEvent], None]) -> None:
        with self._lock:
            if callback in self._subs:
                self._subs.remove(callback)

    # -- the protocol, passed through ------------------------------------------------------------------
    def _log(self, tool: str, *args) -> None:
        self.calls.append((time.monotonic(), tool, args))

    def manifest(self):
        return self.inner.manifest()

    def world(self):
        w = self.inner.world()
        with self._lock:
            w.hands = list(self._hands)
        return w

    def status(self, skill_id: str):
        return self.inner.status(skill_id)

    def precondition(self, arm: str, skill: str, args: dict):
        return self.inner.precondition(arm, skill, args)

    def start(self, arm: str, skill: str, args: dict) -> str:
        self._log("start", arm, skill, dict(args))
        return self.inner.start(arm, skill, args)

    def hold(self, arm: str) -> None:
        self._log("hold", arm)
        self.inner.hold(arm)

    def pause(self, arm: str) -> None:
        self._log("pause", arm)
        self.inner.pause(arm)

    def resume(self, arm: str) -> None:
        self._log("resume", arm)
        self.inner.resume(arm)

    def retarget(self, skill_id: str) -> None:
        self._log("retarget", skill_id)
        self.inner.retarget(skill_id)

    def stop(self) -> None:
        self._log("stop")
        self.inner.stop()

    def heartbeat(self) -> None:
        self.inner.heartbeat()
