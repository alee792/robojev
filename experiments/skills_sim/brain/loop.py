"""H1: the brain on a real clock, as a client of the robot protocol (docs/harness-spikes.md "Order").

    plan = planner(task, world)                 # the one planned wait
    start step 1; decide("plan arrived")        # the plan check runs alongside
    on every event: decide -> right-now at once; in-plan fix if stay_local and confident;
                    else the planner replans while the arm carries on or holds, per right-now

One asyncio queue (messages.py) carries everything: server notifications (bridged from the server's
thread), user text, decisions and planner results (from worker threads), and a heartbeat tick every
200 ms. The loop handles one message at a time and never waits on a model: every decision and
planner call runs in a thread pool and comes back as a message, so the arm carries on or holds, as
the right-now answer said, while a plan is in flight, and several decisions can be in flight at once.

What happens on each message is e12v2's harness logic (docs/v2.md), re-expressed for seconds and
protocol calls instead of ticks and motor commands. The pure parts are imported: the decision
request and combine() (inside the decider), the plan format, validator and diffs, done checks, the
change detector (behind changes.py's settling), the loop detector and is_stop. tracker.py holds the plan bookkeeping, rules.py the
code rules, arm.py every call that moves the arm, text.py the literal event text.

Rules (rules.py has the why): STOP and typed "stop" act at once in code; models can only pause;
one decision per event (a plan trusted in a replan loop is the one event not decided); the 0.8 gate
on the weakest confidence; the replan-loop limit (3 rejections since the user last spoke); a
heartbeat every 200 ms; hold on safety_trip; a reconnecting brain carries on (below).

Reconnecting. The server owns the state, so a new Brain on a server whose arm is mid-skill lets that
skill run: it reads world() and the arm's mode, plans only once the arm is free (so the plan starts
from the world the skill leaves, and nothing is redone), and never start()s over it. The protocol
gives no way to learn that skill's id (ArmObs has no skill id and there is no list of skills), so
the brain recognises its end as the first skill_done/skill_failed for that arm with an id it did not
start, or, failing that, as the arm reading neither running, holding, paused nor tripped on a
heartbeat tick (an end event can race the subscription, and a server may report an arm frozen by
an earlier hold() as "holding" with no skill on it at all). A held arm resumes after the usual
hold; a paused or tripped one waits for a decision.

Robot events: skill_done/failed for the brain's own skill end the step (others are noted and
ignored: another arm, a skill the brain stopped following). A start() the server refuses raises
ValueError: the step could not start, an event for the decision loop like a failed precondition. scene_change is a wake-up: the brain re-reads the world and
e12v2's change detector, settled (changes.py), decides what changed that the robot didn't cause;
each change is one event.
safety_trip holds the arm in code and becomes an event. heartbeat_lost (the server already held the
arm) is the brain's own link failing, not news about the world: the arm resumes after the usual hold.
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

from e12v2.core import plan as P
from e12v2.core.changes import hand_facts
from e12v2.core.data import LLM_ROUTES, Event
from e12v2.core.decision import View

from .. import catalog
from ..protocol import RobotEvent, RobotServer
from .adapt import core_state, geometry, names, xy_cm
from .arm import ArmControl
from .changes import SettledChanges
from .connect import connect
from .messages import Crashed, Decided, FromRobot, Message, Planned, Tick, UserSaid, Wake
from .planner import ManifestPlanner, PlanSnapshot
from .rules import RIGHT_NOW_ACTION, LoopDetector, ReplanGuard, enforce_gate, is_stop
from .steps import CATALOG_NAME, call_args
from .text import between_steps, change_text, where_text
from .tracker import PlanTracker

END = ("done", "stopped", "gave_up", "timeout")
ARM_BUSY = ("running", "holding", "paused", "tripped")


class UserChannel(Protocol):
    """Where typed text comes from and questions go."""

    def attach(self, say: Callable[[str], None], press_stop: Callable[[], None]) -> None:
        """Called once, from inside the running loop. `say` and `press_stop` are safe from any thread."""

    def ask(self, question: str) -> None:
        """Put a question to the user; the answer arrives later through `say`."""


@dataclass
class BrainConfig:
    max_s: float = 120.0              # wall-clock limit for the episode
    heartbeat_s: float = 0.2          # heartbeat and world-check period
    hold_s: float = 1.0               # a hold with no replan pending lasts this long
    paused_idle_s: float = 5.0        # a pause with nothing happening becomes a "paused_idle" event
    max_llm_requests: int = 12        # per episode; past this the episode is given up
    loop_repeats: int = 3             # the same two steps alternating this many times = a loop
    max_step_failures: int = 3        # one step failing this many times = a loop
    max_plan_rejections: int = 3      # rules.ReplanGuard
    stay_local_gate: float = 0.8      # rules.enforce_gate
    settle_s: float = 0.4             # an object change must hold this long to be an event (changes.py)
    arm: str | None = None            # the arm to drive; None = the manifest's first (H2 widens this)
    workers: int = 4                  # threads for model calls


@dataclass
class EpisodeResult:
    outcome: str = "running"          # done | stopped | gave_up | timeout
    wall_s: float = 0.0
    done_s: float | None = None
    decisions: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)       # one record per planner request that came back
    events: Counter = field(default_factory=Counter)
    steps_started: int = 0
    steps_done: int = 0
    steps_failed: int = 0
    loops: int = 0
    stale_plans: int = 0
    failed_plans: int = 0
    log: list[dict] = field(default_factory=list)


class Brain:
    """One episode against one robot server.

    `decider`: an e12v2 Decider (decide(event, view) -> {"combined": Combined, ...}; e12v2's
    JevDecider over any DecisionBackend). `planner`: route -> e12v2 LLMBackend; the brain builds
    every request from the connected manifest itself, so no caller can hand it another robot's
    schema. Construction connects: a non-conforming manifest raises connect.ManifestRefused.
    """

    def __init__(self, server: RobotServer, decider, planner: Mapping[str, object], user: UserChannel | None = None,
                 cfg: BrainConfig | None = None):
        self.cfg = cfg or BrainConfig()
        self.server, self.decider, self.user = server, decider, user
        self.manifest, spec = connect(server, self.cfg.arm)
        self.arm_id = spec.id
        self.arm = ArmControl(server, spec.id)
        self.planner = ManifestPlanner(planner, self.manifest, spec.id)
        self.geom = geometry(spec, self.cfg.heartbeat_s)
        self.plan = PlanTracker()
        self.changes = SettledChanges(self.cfg.settle_s)
        self.loops = LoopDetector(self.cfg.loop_repeats)
        self.replans = ReplanGuard(self.cfg.max_plan_rejections)
        self.r = EpisodeResult()
        self.task = ""
        self.user_messages: list[str] = []
        self.mode = "waiting"             # waiting | running | holding | paused | done | stopped | gave_up | timeout
        self.pause_reason: str | None = None
        self.hold_until: float | None = None
        self.awaiting: int | None = None  # the event whose decision gates the next step
        self.replan_seq = 0               # the newest planner request; older results are stale
        self.replan_pending = False
        self.llm_requests = 0
        self.misfits = 0
        self.world = self.state = None
        self.names: dict[str, str] = {}
        self._new_events: list[Event] = []
        self._ev_id = 0
        self._in_flight = 0               # decisions asked and not yet back
        self._last_event_t = 0.0
        self._robot_moved: set[str] = set()   # objects the robot moved since the change detector last looked
        self._stop_why: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._q: asyncio.Queue[Message] | None = None
        self._closed = self._ran = False
        self._pool = ThreadPoolExecutor(max_workers=self.cfg.workers, thread_name_prefix="brain")
        self._t0 = time.monotonic()

    # ================================================================ inputs (any thread)
    def say(self, text: str) -> None:
        """Typed text. A bare "stop" is STOP, handled here in code before anything queues."""
        if is_stop(text):
            self._stop(f'the user typed "{text}"')
        else:
            self._post(UserSaid(text))

    def press_stop(self) -> None:
        self._stop("the STOP button")

    def _stop(self, why: str) -> None:
        if self.arm.stopped.is_set():
            return
        self._stop_why = why
        self.arm.stop()
        self._post(Wake())

    def _from_robot(self, ev: RobotEvent) -> None:
        """The server's subscription callback: runs on its thread, returns at once."""
        self._post(FromRobot(ev))

    def _post(self, msg: Message) -> None:
        loop = self._loop
        if loop is None or self._closed:
            return
        try:
            loop.call_soon_threadsafe(self._q.put_nowait, msg)
        except RuntimeError:              # the loop has closed: the episode is over
            pass

    def _in_thread(self, fn: Callable[[], object], wrap: Callable[[object], Message]) -> None:
        """Run a model call in the pool; its result (or its exception) comes back as a message."""
        fut = self._loop.run_in_executor(self._pool, fn)

        def back(f: asyncio.Future) -> None:
            if f.cancelled() or self._closed:
                return
            err = f.exception()
            self._q.put_nowait(Crashed(err) if err is not None else wrap(f.result()))
        fut.add_done_callback(back)

    # ================================================================ the episode
    async def run(self, task: str) -> EpisodeResult:
        if self._ran:
            raise RuntimeError("a Brain runs one episode; a new Brain picks up where the server is")
        self._ran = True
        self._loop, self._q = asyncio.get_running_loop(), asyncio.Queue()
        self._t0, self.task = time.monotonic(), task
        self.server.subscribe(self._from_robot)
        self._observe()
        self.changes.update(self.state, 0.0)
        self._take_over_arm()
        if self.user is not None:
            self.user.attach(self.say, self.press_stop)
        if self.mode == "waiting":
            self._request_plan()
        beat = asyncio.create_task(self._heartbeat())
        try:
            while self.mode not in END:
                try:
                    msg = await asyncio.wait_for(self._q.get(), timeout=max(0.0, self.cfg.max_s - self.now()))
                except TimeoutError:
                    self.mode = "timeout"
                    break
                if self.arm.stopped.is_set():
                    self.mode = "stopped"
                    self._note("stop", why=self._stop_why)
                    break
                self._handle(msg)
                if self.mode in END:
                    break
                self._advance()
                self._dispatch()
        finally:
            beat.cancel()
            self._closed = True
            self._pool.shutdown(wait=False, cancel_futures=True)
            if self.mode not in ("done", "stopped"):
                self.arm.set_motion("holding")      # timeout, given up or a bug: leave the arm still, grip kept
        self.r.outcome, self.r.wall_s = self.mode, self.now()
        self._note("end", outcome=self.mode)
        return self.r

    async def _heartbeat(self) -> None:
        while True:
            self.server.heartbeat()
            self._q.put_nowait(Tick())
            await asyncio.sleep(self.cfg.heartbeat_s)

    def _handle(self, msg: Message) -> None:
        match msg:
            case FromRobot(event=ev):
                self._robot_event(ev)
            case UserSaid(text=text):
                self._user_said(text)
            case Tick():
                self._observe_changes()
                if self.arm.foreign and self.world.arms[self.arm_id].mode not in ARM_BUSY:
                    self._found_skill_ended(None)
                self._timers()
            case Decided(event=ev, out=out):
                self._apply_decision(ev, out)
            case Planned():
                self._apply_plan(msg)
            case Crashed(error=err):
                raise err
            case Wake():
                pass

    def _advance(self) -> None:
        """Start the next step when nothing holds it back."""
        if self.mode == "running" and not self.arm.busy and self.awaiting is None and self.plan.plan is not None:
            self._start_next()

    def _dispatch(self) -> None:
        """One decision per new event, each in a thread, each with the view as it is now."""
        evs, self._new_events = self._new_events, []
        for ev in evs:
            if ev.kind == "plan_arrived" and self.replans.trusting:
                self._note("plan_trusted", why="replan loop: running the new plan without deciding on it")
                continue
            view = self._view(ev)
            self._in_flight += 1
            self._in_thread(partial(self.decider.decide, ev, view), partial(Decided, ev))

    # ================================================================ helpers
    def now(self) -> float:
        return time.monotonic() - self._t0

    def name(self, oid: str) -> str:
        return self.names.get(oid, oid)

    def _note(self, note: str, /, **rec) -> None:
        self.r.log.append({"t": round(self.now(), 3), "note": note, **rec})

    def _observe(self) -> None:
        self.world = self.server.world()
        self.state = core_state(self.world, self.arm_id, self.geom, self.cfg.heartbeat_s)
        self.names = names(self.world)

    def _arm_to(self, mode: str) -> None:
        """The brain's mode, and the server call that makes the arm match it (arm.py)."""
        if self.mode == mode:
            return
        before, self.mode = self.mode, mode
        self.arm.set_motion(mode)
        self._note("mode", before=before, after=mode)

    def _heading(self) -> tuple[float, float] | None:
        """Where our skill is heading (cm), for "in the arm's path"; None while paused or between steps."""
        if self.arm.skill_id is None or self.mode == "paused":
            return None
        return xy_cm(self.arm.status().heading)

    @staticmethod
    def _obj(step: dict) -> str | None:
        return step["object"] if step["object"] != P.NONE else None

    @staticmethod
    def _call(step: dict) -> tuple[str, dict]:
        """A step -> (catalog skill name, start() arguments)."""
        name = CATALOG_NAME[step["skill"]]
        return name, call_args(step, catalog.BY_NAME[name])

    def _why_not(self, step: dict) -> str | None:
        return self.arm.precondition(*self._call(step))

    def _runnable_other(self, ev: Event | None) -> str | None:
        """A pending step, other than the current one and the event's, that could start now."""
        if self.plan.plan is None:
            return None
        skip = ({ev.step_id} if ev is not None and ev.step_id else set()) | ({self.plan.cur["id"]} if self.plan.cur else set())
        steps = self.plan.steps()
        return next((sid for sid in self.plan.pending(skip) if self._why_not(steps[sid]) is None), None)

    def _view(self, ev: Event | None) -> View:
        """What the decider sees (e12v2's View), a snapshot: it is read in another thread."""
        st, cur = self.state, self.plan.cur
        if cur is None and ev is not None and ev.step_id and self.plan.plan:
            cur = self.plan.steps().get(ev.step_id)
        if self.arm.skill_id is not None:
            phase = self.arm.status().phase_text
        elif self.arm.foreign:
            phase = "partway through a skill started before this brain connected"
        elif self.plan.last_step is not None:
            last = self.plan.last_step
            phase = between_steps(P.step_text(last, self.names), self.plan.status.get(last["id"]) == "done",
                                  self.name(st.arm.holding) if st.arm.holding else None)
        else:
            phase = "not started on any step yet" if self.plan.plan else "waiting for the first plan"
        path, other = self._heading(), self._runnable_other(ev)
        return View(t=self.now(), task=self.task, user_messages=list(self.user_messages), state=st, plan=self.plan.plan,
                    queue=list(self.plan.queue), status=dict(self.plan.status), current=cur, phase=phase, mode=self.mode,
                    pause_reason=self.pause_reason, replan_pending=self.replan_pending, names=dict(self.names),
                    path_to=path, hand=hand_facts(st, path),
                    runnable_other=P.step_text(self.plan.steps()[other], self.names) if other else None,
                    tries=dict(self.plan.tries),
                    grasped=bool(cur and st.arm.holding and st.arm.holding == cur.get("object")))

    # ================================================================ events
    def event(self, kind: str, text: str, obj: str | None = None, step_id: str | None = None, /, **data) -> Event:
        """A new event; it gets one decision when the current message is done. `data` goes to the
        decider as Event.data (e.g. text= for user_text: what the user said, verbatim)."""
        self._ev_id += 1
        ev = Event(self._ev_id, self.now(), kind, text, obj, step_id, data)
        self._new_events.append(ev)
        self.r.events[kind] += 1
        self._last_event_t = self.now()
        self._note("event", id=ev.id, event=kind, text=text)
        return ev

    def _take_over_arm(self) -> None:
        """At connect: a skill already on the arm is left to run (module docstring, "Reconnecting")."""
        a = self.world.arms[self.arm_id]
        if a.mode == "stopped":
            self.mode, self._stop_why = "stopped", "the arm was already stopped (STOP) when the brain connected"
            self._note("stop", why=self._stop_why)
            return
        if a.mode not in ARM_BUSY:
            return
        self.arm.foreign = True
        self._note("found_skill", mode=a.mode, holding=a.holding)
        if a.mode == "running":
            self.mode = "running"
        elif a.mode == "holding":
            self.mode, self.hold_until = "holding", self.now() + self.cfg.hold_s
        else:
            self.mode, self.pause_reason = "paused", f"found {a.mode} when the brain connected"
            held = f", holding {self.name(a.holding)}" if a.holding else ""
            self.event("arm_found", f"when the brain connected, the arm was {a.mode} partway through a skill it did not start{held}")

    def _robot_event(self, ev: RobotEvent) -> None:
        if ev.arm is not None and ev.arm != self.arm_id:
            self._note("robot_event_ignored", event=ev.kind, arm=ev.arm, why="another arm")
            return
        if ev.kind in ("skill_done", "skill_failed"):
            ok = ev.kind == "skill_done"
            if ev.skill_id is not None and ev.skill_id == self.arm.skill_id:
                reason = None if ok else self._failure_reason(ev)
                self.arm.ended()
                self._observe()
                self._step_ended(ok, reason)
            elif self.arm.foreign and ev.skill_id is not None:
                self._found_skill_ended(ev)
            else:
                self._note("robot_event_ignored", event=ev.kind, skill_id=ev.skill_id, why="not our current skill")
        elif ev.kind == "scene_change":
            self._observe_changes()
        elif ev.kind == "safety_trip":
            self._tripped(ev)
        elif ev.kind == "heartbeat_lost":
            self._note("heartbeat_lost", text=ev.text)
            if self.mode == "running":       # the server already holds the arm: resume after the usual hold
                self.mode, self.hold_until = "holding", self.now() + self.cfg.hold_s
        else:
            self._note("robot_event_ignored", event=ev.kind, why="unknown kind")

    def _failure_reason(self, ev: RobotEvent) -> str:
        """The protocol puts "<code>: <text>" in status().reason; the event text is the fallback."""
        try:
            return self.server.status(ev.skill_id).reason or ev.text
        except KeyError:
            return ev.text

    def _tripped(self, ev: RobotEvent) -> None:
        """The safety filter outranks every answer: hold now, in code, then decide what it means. The
        brain counts the arm as paused, so no timer resumes it: only a decision or the user does."""
        self.arm.set_motion("holding")
        if self.mode in END:
            return
        self.mode, self.pause_reason, self.hold_until = "paused", "the robot's safety filter tripped", None
        self._note("mode", after="paused", because="safety_trip")
        self.event("safety_trip", f"the robot's safety filter tripped ({ev.text}); the arm is holding where it was")

    def _found_skill_ended(self, ev: RobotEvent | None) -> None:
        """The skill found at connect ended (its end event, or None: the arm reads free): the arm is
        ours. Plan from the world it left."""
        self.arm.foreign = False
        what, obj = "the arm reads " + self.world.arms[self.arm_id].mode, None
        if ev is not None:
            try:
                st = self.server.status(ev.skill_id)
                what, obj = f"{st.name}: {st.phase_text}", st.args.get("object")
            except KeyError:
                what, obj = ev.text, ev.object
        if obj:
            self._robot_moved.add(obj)
        self._note("found_skill_ended", skill_id=ev.skill_id if ev else None, event=ev.kind if ev else None, what=what)
        self._observe()
        self.mode, self.pause_reason, self.hold_until = "waiting", None, None
        self._request_plan()

    def _user_said(self, text: str) -> None:
        self.user_messages.append(text)
        self.replans.user_spoke()
        if self.mode == "paused" and (self.pause_reason or "").startswith("waiting for the user's answer"):
            self._arm_to("holding")               # the answer ends that pause; hold while it is decided
            self.pause_reason, self.hold_until = None, self.now() + self.cfg.hold_s
        self.event("user_text", f'the user typed: "{text}"', text=text)

    def _observe_changes(self) -> None:
        """Re-read the world; each settled change the robot didn't cause is one scene_change event."""
        self._observe()
        for ch in self.changes.update(self.state, self.now(), self._heading(), self._robot_touched()):
            self.event("scene_change", change_text(ch, self.name), ch.get("object"), change=ch)

    def _robot_touched(self) -> set[str]:
        """Objects the robot itself moved since the detector last looked: those of steps that ended,
        and the running step's object once it is in the gripper or at the step's goal (its end event
        may still be on its way) or, for push, at all. Anything else that moved, someone else moved."""
        out, self._robot_moved = self._robot_moved, set()
        cur = self.plan.cur
        if cur is not None and cur["object"] in self.state.objects:
            where = self.state.objects[cur["object"]].where
            goal = P.step_goal(cur)
            if cur["skill"] == "push" or where == "gripper" or (goal is not None and where == goal[1]):
                out.add(cur["object"])
        return out

    def _timers(self) -> None:
        t = self.now()
        if self.mode == "holding" and not self.replan_pending and self.hold_until is not None and t >= self.hold_until:
            self.hold_until = None
            self._arm_to("running")
        if self.mode == "paused" and t - self._last_event_t >= self.cfg.paused_idle_s and not self._in_flight:
            self.event("paused_idle", f"the arm has been paused for {self.cfg.paused_idle_s:.0f} s ({self.pause_reason}) "
                                      "and nothing has changed")

    # ================================================================ steps
    def _start_next(self) -> None:
        sid = self.plan.next_pending()
        if sid is None:
            self._check_done()
            return
        s = self.plan.steps()[sid]
        name, args = self._call(s)
        why = self.arm.precondition(name, args)
        if why is None:
            try:
                self.arm.start(name, args)
            except ValueError as e:               # refused at start: same as a failed precondition
                why = str(e)
        txt = P.step_text(s, self.names)
        if why is not None:
            self.plan.could_not_start(sid)
            self.r.steps_failed += 1
            ev = self.event("step_failed", f"step could not start: {txt}: {why}", self._obj(s), sid,
                            reason=f"could not start: {why}")
            self.awaiting = ev.id
        else:
            self.plan.begin(sid)
            self.r.steps_started += 1
            self._note("step_start", step=txt, skill_id=self.arm.skill_id)
        self._loop_check(s, sid)

    def _loop_check(self, s: dict, sid: str) -> None:
        if not (self.loops.record((s["skill"], s["object"], s["target"], s["direction"]))
                or self.plan.tries[sid] >= self.cfg.max_step_failures):
            return
        self.r.loops += 1
        self.plan.tries[sid] = 0
        txt = P.step_text(s, self.names)
        self._note("loop", step=txt)
        self._arm_to("holding")
        if not self.replan_pending:
            ev = Event(0, self.now(), "loop", f"loop: {txt} keeps coming back", self._obj(s), sid)
            self._start_replan("capable_llm", f"a loop: the plan keeps returning to {txt} without progress", ev)

    def _step_ended(self, ok: bool, reason: str | None) -> None:
        s = self.plan.end(ok, self.now())
        obj = self._obj(s)
        if obj:
            self._robot_moved.add(obj)
        txt = P.step_text(s, self.names)
        if ok:
            self.r.steps_done += 1
            g = P.step_goal(s)
            after = (f"; {self.name(g[0])} is now {where_text(self.state.objects[g[0]].where, self.name)}"
                     if g and g[0] in self.state.objects else "")
            ev = self.event("step_done", f"step finished: {txt}{after}", obj, s["id"])
        else:
            self.r.steps_failed += 1
            ev = self.event("step_failed", f"step failed: {txt}: {reason}", obj, s["id"], reason=reason)
        self._note("step_end", step=txt, ok=ok, reason=reason)
        self.awaiting = ev.id

    def _check_done(self) -> None:
        if self.replan_pending:
            return                                # a plan is on its way: not done until it arrives and says so
        self._observe()
        unmet = self.plan.unmet(self.state)
        if not unmet:
            self.mode, self.r.done_s = "done", self.now()
            self._note("done")
        elif not self.plan.unmet_raised:
            self.plan.unmet_raised = True
            obj = unmet[0]["object"] if unmet[0]["object"] in self.state.objects else None
            ev = self.event("plan_done_unmet", "every planned step has run, but these done conditions are false: " +
                            "; ".join(c["text"] for c in unmet), obj, unmet=[c["text"] for c in unmet])
            self.awaiting = ev.id

    # ================================================================ decisions
    def _apply_decision(self, ev: Event, out: dict) -> None:
        self._in_flight -= 1
        c = enforce_gate(out["combined"], self.cfg.stay_local_gate)
        self.r.decisions.append({"t": self.now(), "t_event": ev.t, "event_id": ev.id, "event": ev.kind, "text": ev.text,
                                 "combined": c, "latency_ms": out.get("latency_ms", 0.0)})
        self._note("decision", event_id=ev.id, event=ev.kind, right_now=c.right_now, fix=c.fix, route=c.route, reason=c.reason)
        if ev.id == self.awaiting:
            self.awaiting = None
        if self.mode in END:
            return
        self.plan.done_answers.update(c.done)
        self._right_now(c.right_now, ev)
        if c.route == "stay_local":
            self._fix(c.fix, ev)
        elif c.route in LLM_ROUTES:
            self._start_replan(c.route, ev.text + ("; plan check: " + "; ".join(c.problems) if c.problems else ""), ev)
        elif c.route == "ask_user":
            self._arm_to("paused")
            self.pause_reason = "waiting for the user's answer to a question"
            if self.user is not None:
                self.user.ask(f"About \"{ev.data.get('text', ev.text)}\": what should the end result be?")
        if self.mode == "holding" and not self.replan_pending:
            self.hold_until = self.now() + self.cfg.hold_s

    def _right_now(self, answer: str, ev: Event) -> None:
        """Apply the right-now answer. It can hold, pause, resume or retarget; nothing a model says stops."""
        if self.mode == "waiting" and answer not in ("pause", "back_off"):
            return
        action = RIGHT_NOW_ACTION.get(answer)
        if action == "holding" and self.mode == "running":
            self._arm_to("holding")
        elif action == "paused" and self.mode != "paused":
            self._arm_to("paused")
            # the user's own words, not an interpretation (e12v2: "asked to wait" kept a cautious pause paused)
            self.pause_reason = (f'the user said "{ev.data.get("text", "")}"' if ev.kind == "user_text" else
                                 "a person's hand came near" if "hand" in ev.text else f"paused after: {ev.text}")
        elif action == "retarget":
            self.arm.retarget()
        elif action == "running" and self.mode in ("paused", "holding"):
            self.hold_until, self.pause_reason = None, None
            self._arm_to("running")

    def _fix(self, fix: str, ev: Event) -> None:
        other = self._runnable_other(ev) if fix == "another_step_first" else None
        if self.plan.apply_fix(fix, ev, self.state.arm.holding, other):
            self.arm.release()
        if fix != "none":
            self._note("fix", fix=fix, queue=list(self.plan.queue))

    # ================================================================ the planner
    def _ref(self, ev: Event | None) -> dict:
        snap = PlanSnapshot(copy.deepcopy(self.plan.plan), list(self.plan.queue)) if self.plan.plan else None
        return {"episode": snap, "event": ev}

    def _new_request(self) -> int | None:
        if self.llm_requests >= self.cfg.max_llm_requests:
            self.mode = "gave_up"
            self._note("gave_up", why=f"{self.cfg.max_llm_requests} planner requests")
            return None
        self.llm_requests += 1
        self.replan_seq += 1
        self.replan_pending = True
        return self.replan_seq

    def _request_plan(self) -> None:
        seq = self._new_request()
        if seq is None:
            return
        task, said, world, core, ref, t = self.task, list(self.user_messages), self.world, self.state, self._ref(None), self.now()
        self._note("llm_request", request="plan", route="fast_llm")
        self._in_thread(lambda: self.planner.plan(task, said, world, core, ref=ref), lambda res: Planned(seq, t, None, res))

    def _start_replan(self, route: str, why: str, ev: Event | None) -> None:
        if self.plan.plan is None:
            return                                # the first plan is on its way; it will read user_messages
        if ev is not None and ev.kind == "plan_arrived" and self.replans.rejected():
            self.r.loops += 1
            self._note("loop", loop="replans", rejections=self.replans.rejections)
        seq = self._new_request()
        if seq is None:
            return
        st = self.arm.status()
        arm_now = {"current_step": self.plan.cur["id"] if self.plan.cur else "none",
                   "doing": st.phase_text if st is not None else "between steps",
                   "while_you_plan": {"holding": "holding still", "paused": "paused"}.get(self.mode, "carrying on")}
        args = (route, self.task, list(self.user_messages), self.world, self.state, copy.deepcopy(self.plan.plan),
                list(self.plan.queue), dict(self.plan.status), arm_now, why)
        ref, t = self._ref(ev), self.now()
        self._note("llm_request", request="replan", route=route, why=why)
        self._in_thread(lambda: self.planner.replan(*args, ref=ref), lambda res: Planned(seq, t, ev, res))

    def _apply_plan(self, msg: Planned) -> None:
        res = msg.result
        self.r.llm.append({"t_request": msg.t_request, "t": self.now(), "kind": res.kind, "route": res.route, "ok": res.ok,
                           "attempts": res.attempts, "event": msg.event.kind if msg.event else None})
        if msg.seq != self.replan_seq:
            self.r.stale_plans += 1
            self._note("plan_stale", seq=msg.seq)
            return
        self.replan_pending = False
        self._observe()
        if not res.ok:
            self.r.failed_plans += 1
            self._note("plan_failed", request=res.kind, errors=[a["errors"] for a in res.attempts])
            if res.kind == "plan":
                if self.r.failed_plans >= 2:
                    self.mode = "gave_up"
                else:
                    self._request_plan()
            else:
                self.plan.unmet_raised = False
                if self.mode == "holding":
                    self._arm_to("running")
            return
        if res.kind == "plan":
            self.plan.load(res.plan)
            if self.mode == "waiting":
                self.mode = "running"
            self._note("plan_applied", plan=res.plan)
            self.event("plan_arrived", "the first plan arrived: " + res.plan.get("reading", ""))
            return
        held = self.state.arm.holding
        newp, q, check = self.plan.merge(res.diff, msg.t_request, held)
        errs = P.validate(newp, self.state, held, check)
        if errs and self.misfits < 2:
            # the world moved on while the LLM worked: ask again from where things stand now
            self.misfits += 1
            txt = "the new plan no longer fits the world: " + "; ".join(errs[:3])
            self._note("plan_misfit", errors=errs)
            self._start_replan(res.route, txt, Event(0, self.now(), "misfit", txt))
            return
        self.misfits = 0
        if self.plan.install(newp, q):
            self.arm.release()
        if self.mode == "holding":
            self.hold_until = None
            self._arm_to("running")
        self._note("plan_applied", diff=res.diff, queue=q)
        self.event("plan_arrived", "a new plan arrived: " + (res.diff.get("reading") or ""))
