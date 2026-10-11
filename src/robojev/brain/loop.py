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

Several arms (docs/widowx-pair.md "Crawl: one brain, two arms"; docs/architecture.md choice 5). One
brain, one planner, one plan; each step names its arm (planner.py). Each driven arm has its own
ArmRun: its mode (running, holding, paused), pause reason, hold timer, the decision that gates its
next step, its skill, its change detector for hands, its loop detector. What is shared stays one:
the queue, the planner (one request in flight), the heartbeat, STOP (one stop() parks every arm).
  - Steps: each arm starts its next step when it is free, so steps on different arms run side by
    side; a step whose object is held or claimed by another arm, or that uses an object or slot
    another arm's step is using, waits (arms.py). The shared-zone lock is the server's, not ours.
  - Events: every event becomes one core Event per arm it concerns, each decided once, with that
    arm's view; Event.data carries "arm" and "about" (the event the copies share). A step's end, a
    safety trip, a found skill, paused_idle: its own arm. User text and a new plan: every arm. An
    object moved: the arms that reach it, where it was or is, or that have a step for it. A hand:
    the arm it is nearest, any arm it is near or in the path of, and an arm that was told of it
    (until it hears it leave). A plan_done_unmet: the arm of the last step for its object.
  - Views: the "arm" facts (position, holding, phase text, mode) are the arm's own; the plan's
    position lists that arm's steps (a plan_arrived check sees the whole plan); every other arm is
    one more thing in the scene, listed among nearby objects with its phase text.
  - Decisions: right-now and in-plan fixes apply to the arm asked about. A replan or a question to
    the user happens once per event, however many of its copies ask for it.
With one arm (or BrainConfig.arm naming one) this is exactly the one-arm loop: one ArmRun, events
with no arm data, the whole plan in every view.
"""
from __future__ import annotations

import asyncio
import copy
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Protocol

from .core import plan as P
from .core.changes import hand_facts
from .core.data import LLM_ROUTES, Event, Geometry, Obj
from .core.data import WorldState as CoreWorld
from .core.decision import View

from .. import catalog
from ..protocol import ArmSpec, RobotEvent, RobotServer
from .adapt import CM, core_state, geometry, names, xy_cm
from .arm import ArmControl
from .arms import reaches, waits_for
from .changes import SettledChanges
from .connect import connect
from .messages import Crashed, Decided, FromRobot, Message, Planned, Tick, UserSaid, Wake
from .planner import ManifestPlanner, PlanSnapshot
from .rules import RIGHT_NOW_ACTION, LoopDetector, ReplanGuard, enforce_gate, is_stop, is_transient, transient_mentions
from .steps import CATALOG_NAME, call_args
from .text import between_steps, change_text, where_text
from .tracker import PlanTracker

END = ("done", "stopped", "gave_up", "timeout")
ARM_BUSY = ("running", "holding", "paused", "tripped")
NEAR_BANDS = ("very close", "near")       # core.changes.hand_band: a hand this close concerns any arm


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
    arm: str | None = None            # the one arm to drive; None = every arm in the manifest
    workers: int = 4                  # threads for model calls


@dataclass
class EpisodeResult:
    outcome: str = "running"          # done | stopped | gave_up | timeout
    wall_s: float = 0.0
    done_s: float | None = None
    decisions: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)       # one record per planner request that came back
    events: Counter = field(default_factory=Counter)    # per event, however many arms it concerned
    steps_started: int = 0
    steps_done: int = 0
    steps_failed: int = 0
    loops: int = 0
    stale_plans: int = 0
    failed_plans: int = 0
    log: list[dict] = field(default_factory=list)


@dataclass
class ArmRun:
    """One driven arm's share of the episode (module docstring, "Several arms")."""
    id: str
    spec: ArmSpec
    ctl: ArmControl
    geom: Geometry
    changes: SettledChanges
    loops: LoopDetector
    mode: str = "waiting"                  # waiting | running | holding | paused: what the brain wants the arm doing
    pause_reason: str | None = None
    hold_until: float | None = None
    awaiting: int | None = None            # the event whose decision gates this arm's next step
    state: CoreWorld | None = None         # the world as e12v2 reads it from this arm (cm; its gripper is "gripper")
    last_event_t: float = 0.0              # the last event concerning this arm (paused_idle)
    in_flight: int = 0                     # decisions asked about this arm and not yet back
    hand_told: bool = False                # it has been told of the hand in view (and must hear it leave)
    waiting_on: str | None = None          # why its next step waits for another arm, while it does


class Brain:
    """One episode against one robot server, driving every arm (or BrainConfig.arm alone).

    `decider`: an e12v2 Decider (decide(event, view) -> {"combined": Combined, ...}; e12v2's
    JevDecider over any DecisionBackend). `planner`: route -> e12v2 LLMBackend; the brain builds
    every request from the connected manifest itself, so no caller can hand it another robot's
    schema. Construction connects: a non-conforming manifest raises connect.ManifestRefused.
    """

    def __init__(self, server: RobotServer, decider, planner: Mapping[str, object], user: UserChannel | None = None,
                 cfg: BrainConfig | None = None):
        self.cfg = cfg or BrainConfig()
        self.server, self.decider, self.user = server, decider, user
        self.manifest, specs = connect(server, self.cfg.arm)
        self.stopped = threading.Event()
        self.arms = {a.id: ArmRun(a.id, a, ArmControl(server, a.id, self.stopped), geometry(a, self.cfg.heartbeat_s),
                                  SettledChanges(self.cfg.settle_s), LoopDetector(self.cfg.loop_repeats))
                     for a in specs}
        self.primary = specs[0].id
        self.multi = len(self.arms) > 1
        self.planner = ManifestPlanner(planner, self.manifest, tuple(self.arms))
        self.plan = PlanTracker(self.primary)
        self.replans = ReplanGuard(self.cfg.max_plan_rejections)
        self.r = EpisodeResult()
        self.task = ""
        self.user_messages: list[str] = []
        self.user_words: list[str] = []   # the task, then each correction taken as about the end result, verbatim (View.user_words)
        self.mode = "waiting"             # the episode: waiting (no plan yet) | running | done | stopped | gave_up | timeout
        self.replan_seq = 0               # the newest planner request; older results are stale
        self.replan_pending = False
        self.llm_requests = 0
        self.misfits = 0
        self.world = self.state = None    # the protocol world; the primary arm's core world (done checks, schema ids)
        self.names: dict[str, str] = {}
        self._new_events: list[Event] = []
        self._ev_id = 0
        self._once_keys: set[tuple[int, str]] = set()   # (event, "replan" | "ask") already acted on
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
        if self.stopped.is_set():
            return
        self._stop_why = why
        self.arms[self.primary].ctl.stop()      # the server's stop() parks every arm; the flag is shared
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
        self.user_words = [task]
        self.server.subscribe(self._from_robot)
        self._observe()
        for arm in self.arms.values():
            arm.changes.update(arm.state, 0.0)
        self._take_over_arms()
        if self.user is not None:
            self.user.attach(self.say, self.press_stop)
        if self.mode == "waiting" and not any(a.ctl.foreign for a in self.arms.values()):
            self._request_plan()
        beat = asyncio.create_task(self._heartbeat())
        try:
            while self.mode not in END:
                try:
                    msg = await asyncio.wait_for(self._q.get(), timeout=max(0.0, self.cfg.max_s - self.now()))
                except TimeoutError:
                    self.mode = "timeout"
                    break
                if self.stopped.is_set():
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
                for arm in self.arms.values():      # timeout, given up or a bug: leave every arm still, grip kept
                    arm.ctl.set_motion("holding")
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
                for arm in self.arms.values():
                    if arm.ctl.foreign and self.world.arms[arm.id].mode not in ARM_BUSY:
                        self._found_skill_ended(arm, None)
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
        """Start each free arm's next step when nothing holds it back."""
        if self.plan.plan is None:
            return
        for arm in self.arms.values():
            if self.mode in END:
                return
            if arm.mode == "running" and not arm.ctl.busy and arm.awaiting is None:
                self._start_next(arm)

    def _dispatch(self) -> None:
        """One decision per new event (per arm it concerns), each in a thread, with the view as it is now."""
        evs, self._new_events = self._new_events, []
        for ev in evs:
            if ev.kind == "plan_arrived" and self.replans.trusting:
                self._note("plan_trusted", why="replan loop: running the new plan without deciding on it")
                continue
            arm = self._arm_of(ev)
            view = self._view(ev, arm)
            arm.in_flight += 1
            self._in_thread(partial(self.decider.decide, ev, view), partial(Decided, ev))

    # ================================================================ helpers
    def now(self) -> float:
        return time.monotonic() - self._t0

    def name(self, oid: str) -> str:
        return self.names.get(oid, oid)

    def _note(self, note: str, /, **rec) -> None:
        self.r.log.append({"t": round(self.now(), 3), "note": note, **rec})

    def _arm_of(self, ev: Event) -> ArmRun:
        return self.arms[ev.data.get("arm", self.primary)]

    def _once(self, ev: Event, what: str) -> bool:
        """True the first time `what` is done for this event (any of its per-arm copies)."""
        key = (ev.data.get("about", ev.id), what)
        if key in self._once_keys:
            return False
        self._once_keys.add(key)
        return True

    def _observe(self) -> None:
        self.world = self.server.world()
        for arm in self.arms.values():
            arm.state = core_state(self.world, arm.id, arm.geom, self.cfg.heartbeat_s)
        self.state = self.arms[self.primary].state
        self.names = names(self.world)
        if self.multi:
            self.names.update({f"gripper:{a}": f"{a}'s gripper" for a in self.world.arms})

    def _arm_to(self, arm: ArmRun, mode: str) -> None:
        """The arm's mode, and the server call that makes the arm match it (arm.py)."""
        if arm.mode == mode:
            return
        before, arm.mode = arm.mode, mode
        arm.ctl.set_motion(mode)
        self._note("mode", arm=arm.id, before=before, after=mode)

    def _heading(self, arm: ArmRun) -> tuple[float, float] | None:
        """Where the arm's skill is heading (cm), for "in the arm's path"; None while paused or between steps."""
        if arm.ctl.skill_id is None or arm.mode == "paused":
            return None
        return xy_cm(arm.ctl.status().heading)

    @staticmethod
    def _obj(step: dict) -> str | None:
        return step["object"] if step["object"] != P.NONE else None

    @staticmethod
    def _call(step: dict) -> tuple[str, dict]:
        """A step -> (catalog skill name, start() arguments)."""
        name = CATALOG_NAME[step["skill"]]
        return name, call_args(step, catalog.BY_NAME[name])

    def _why_not(self, step: dict) -> str | None:
        return self.arms[self.plan.arm_of(step)].ctl.precondition(*self._call(step))

    def _waits(self, step: dict) -> str | None:
        """Why `step` must wait for another arm (arms.waits_for); never with one arm."""
        if not self.multi:
            return None
        return waits_for(step["id"], self.plan.queue, self.plan.steps(), self.plan.status, self.world, self.primary, self.names)

    def _runnable_other(self, arm: ArmRun, ev: Event | None) -> str | None:
        """A pending step of `arm`, other than its current one and the event's, that could start now."""
        if self.plan.plan is None:
            return None
        cur = self.plan.current(arm.id)
        skip = ({ev.step_id} if ev is not None and ev.step_id else set()) | ({cur["id"]} if cur else set())
        steps = self.plan.steps()
        return next((sid for sid in self.plan.pending(skip) if self.plan.arm_of(steps[sid]) == arm.id
                     and self._why_not(steps[sid]) is None and self._waits(steps[sid]) is None), None)

    def _phase(self, arm: ArmRun) -> str:
        """Where the arm is, literally: its skill's phase text, or between steps."""
        st = arm.state
        if arm.ctl.skill_id is not None:
            return arm.ctl.status().phase_text
        if arm.ctl.foreign:
            return "partway through a skill started before this brain connected"
        last = self.plan.last.get(arm.id)
        if last is not None:
            text = between_steps(P.step_text(last, self.names), self.plan.status.get(last["id"]) == "done",
                                 self.name(st.arm.holding) if st.arm.holding else None)
        else:
            text = "not started on any step yet" if self.plan.plan else "waiting for the first plan"
        if arm.waiting_on is not None:
            nxt = self.plan.next_for(arm.id)
            if nxt is not None:
                text += f"; its next step, {P.step_text(self.plan.steps()[nxt], self.names)}, waits: {arm.waiting_on}"
        return text

    def _view(self, ev: Event | None, arm: ArmRun) -> View:
        """What the decider sees (e12v2's View) about `arm`, a snapshot: it is read in another thread."""
        st, cur = arm.state, self.plan.current(arm.id)
        if cur is None and ev is not None and ev.step_id and self.plan.plan:
            cur = self.plan.steps().get(ev.step_id)
        whole = not self.multi or (ev is not None and ev.kind == "plan_arrived")
        queue = list(self.plan.queue) if whole else [q for q in self.plan.queue if self.plan.arm_of(self.plan.steps()[q]) == arm.id]
        names_ = dict(self.names)
        shown = st if whole else self._with_other_arms(arm, names_)
        path, other = self._heading(arm), self._runnable_other(arm, ev)
        return View(t=self.now(), task=self.task, user_messages=list(self.user_messages), state=shown, plan=self.plan.plan,
                    queue=queue, status=dict(self.plan.status), current=cur, phase=self._phase(arm), mode=arm.mode,
                    pause_reason=arm.pause_reason, replan_pending=self.replan_pending, names=names_,
                    path_to=path, hand=hand_facts(st, path),
                    runnable_other=P.step_text(self.plan.steps()[other], self.names) if other else None,
                    tries=dict(self.plan.tries),
                    grasped=bool(cur and st.arm.holding and st.arm.holding == cur.get("object")),
                    user_words=list(self.user_words))

    def _with_other_arms(self, arm: ArmRun, names_: dict[str, str]) -> CoreWorld:
        """`arm`'s world with every other driven arm as one more object, so e12v2's nearby-objects
        list shows it (distance band, in the path) like anything else moving on the table. Its name
        carries its phase text; it is "in" the shared zone or its own side of the table."""
        objs = dict(arm.state.objects)
        for other in self.arms.values():
            if other is arm:
                continue
            a, oid = other.state.arm, f"arm:{other.id}"
            zone = next((p for p in self.world.places.values() if p.kind == "zone"
                         and (a.x / CM - p.x) ** 2 + (a.y / CM - p.y) ** 2 <= p.radius ** 2), None)
            names_[oid] = f"{other.id}, the other arm ({self._phase(other)})"
            names_[f"side:{other.id}"] = zone.name if zone is not None else f"{other.id}'s side of the table"
            objs[oid] = Obj(oid, names_[oid], "", None, "", a.x, a.y, a.z, f"side:{other.id}" if zone is None else zone.id)
        return replace(arm.state, objects=objs)

    # ================================================================ events
    def event(self, kind: str, text: str, obj: str | None = None, step_id: str | None = None, /, *,
              arms: Iterable[str] | None = None, gate: bool = False, **data) -> list[Event]:
        """A new event: one core Event per arm it concerns (`arms`, default every arm), each decided
        once when the current message is done. `gate`: its decision gates that arm's next step.
        `data` goes to the decider as Event.data (e.g. text= for user_text: what the user said,
        verbatim); with several arms it also carries "arm" and "about" (the shared event id)."""
        about, out = self._ev_id + 1, []
        self.r.events[kind] += 1
        for a in (list(arms) if arms is not None else list(self.arms)):
            arm = self.arms[a]
            self._ev_id += 1
            ev = Event(self._ev_id, self.now(), kind, text, obj, step_id, {**data, "arm": a, "about": about} if self.multi else data)
            self._new_events.append(ev)
            arm.last_event_t = self.now()
            if gate:
                arm.awaiting = ev.id
            self._note("event", id=ev.id, event=kind, text=text, arm=a)
            out.append(ev)
        return out

    def _take_over_arms(self) -> None:
        """At connect: a skill already on an arm is left to run (module docstring, "Reconnecting")."""
        for arm in self.arms.values():
            a = self.world.arms[arm.id]
            if a.mode == "stopped":
                self.mode, self._stop_why = "stopped", "the arm was already stopped (STOP) when the brain connected"
                self._note("stop", why=self._stop_why)
                return
            if a.mode not in ARM_BUSY:
                continue
            arm.ctl.foreign = True
            self._note("found_skill", arm=arm.id, mode=a.mode, holding=a.holding)
            if a.mode == "running":
                arm.mode = "running"
            elif a.mode == "holding":
                arm.mode, arm.hold_until = "holding", self.now() + self.cfg.hold_s
            else:
                arm.mode, arm.pause_reason = "paused", f"found {a.mode} when the brain connected"
                held = f", holding {self.name(a.holding)}" if a.holding else ""
                self.event("arm_found", f"when the brain connected, the arm was {a.mode} partway through a skill it did not start{held}",
                           arms=[arm.id])

    def _skill_arm(self, ev: RobotEvent) -> ArmRun | None:
        """The driven arm a skill event is about: by its arm, else by the skill id."""
        if ev.arm is not None:
            return self.arms.get(ev.arm)
        return next((a for a in self.arms.values() if a.ctl.skill_id == ev.skill_id), None) or \
            next((a for a in self.arms.values() if a.ctl.foreign), None) or self.arms[self.primary]

    def _robot_event(self, ev: RobotEvent) -> None:
        if ev.arm is not None and ev.arm not in self.arms:
            self._note("robot_event_ignored", event=ev.kind, arm=ev.arm, why="another arm")
            return
        if ev.kind in ("skill_done", "skill_failed"):
            ok, arm = ev.kind == "skill_done", self._skill_arm(ev)
            if ev.skill_id is not None and ev.skill_id == arm.ctl.skill_id:
                reason = None if ok else self._failure_reason(ev)
                arm.ctl.ended()
                self._observe()
                self._step_ended(arm, ok, reason)
            elif arm.ctl.foreign and ev.skill_id is not None:
                self._found_skill_ended(arm, ev)
            else:
                self._note("robot_event_ignored", event=ev.kind, skill_id=ev.skill_id, why="not our current skill")
        elif ev.kind == "scene_change":
            self._observe_changes()
        elif ev.kind == "safety_trip":
            for arm in ([self.arms[ev.arm]] if ev.arm is not None else list(self.arms.values())):
                self._tripped(arm, ev)
        elif ev.kind == "heartbeat_lost":
            self._note("heartbeat_lost", text=ev.text)
            for arm in self.arms.values():       # the server already holds every arm: resume after the usual hold
                if arm.mode == "running":
                    arm.mode, arm.hold_until = "holding", self.now() + self.cfg.hold_s
        else:
            self._note("robot_event_ignored", event=ev.kind, why="unknown kind")

    def _failure_reason(self, ev: RobotEvent) -> str:
        """The protocol puts "<code>: <text>" in status().reason; the event text is the fallback."""
        try:
            return self.server.status(ev.skill_id).reason or ev.text
        except KeyError:
            return ev.text

    def _tripped(self, arm: ArmRun, ev: RobotEvent) -> None:
        """The safety filter outranks every answer: hold now, in code, then decide what it means. The
        brain counts the arm as paused, so no timer resumes it: only a decision or the user does."""
        arm.ctl.set_motion("holding")
        if self.mode in END:
            return
        arm.mode, arm.pause_reason, arm.hold_until = "paused", "the robot's safety filter tripped", None
        self._note("mode", arm=arm.id, after="paused", because="safety_trip")
        self.event("safety_trip", f"the robot's safety filter tripped ({ev.text}); the arm is holding where it was", arms=[arm.id])

    def _found_skill_ended(self, arm: ArmRun, ev: RobotEvent | None) -> None:
        """The skill found at connect ended (its end event, or None: the arm reads free): the arm is
        ours. Once no arm runs a found skill, plan from the world they left."""
        arm.ctl.foreign = False
        what, obj = "the arm reads " + self.world.arms[arm.id].mode, None
        if ev is not None:
            try:
                st = self.server.status(ev.skill_id)
                what, obj = f"{st.name}: {st.phase_text}", st.args.get("object")
            except KeyError:
                what, obj = ev.text, ev.object
        if obj:
            self._robot_moved.add(obj)
        self._note("found_skill_ended", arm=arm.id, skill_id=ev.skill_id if ev else None, event=ev.kind if ev else None, what=what)
        self._observe()
        arm.mode, arm.pause_reason, arm.hold_until = "waiting", None, None
        if self.plan.plan is None and not any(a.ctl.foreign for a in self.arms.values()):
            self._request_plan()

    def _user_said(self, text: str) -> None:
        self.user_messages.append(text)
        self.replans.user_spoke()
        for arm in self.arms.values():
            if arm.mode == "paused" and (arm.pause_reason or "").startswith("waiting for the user's answer"):
                self._arm_to(arm, "holding")          # the answer ends that pause; hold while it is decided
                arm.pause_reason, arm.hold_until = None, self.now() + self.cfg.hold_s
        self.event("user_text", f'the user typed: "{text}"', text=text)

    def _observe_changes(self) -> None:
        """Re-read the world; each settled change the robot didn't cause is one scene_change event,
        for the arms it concerns. Objects come from the primary arm's detector; hands from each arm's own."""
        self._observe()
        touched = self._robot_touched()
        for arm in self.arms.values():
            for ch in arm.changes.update(arm.state, self.now(), self._heading(arm), touched):
                if ch["what"] == "hand":
                    if self._hand_concerns(arm, ch):
                        self.event("scene_change", change_text(ch, self.name), None, arms=[arm.id], change=ch)
                elif arm.id == self.primary:
                    self.event("scene_change", change_text(ch, self.name), ch.get("object"),
                               arms=self._object_arms(ch), change=ch)

    def _hand_concerns(self, arm: ArmRun, ch: dict) -> bool:
        """A hand change is `arm`'s decision when the hand is nearest this arm of all, near it or in
        its path, or the arm was told of the hand before (so it hears it go). One arm: always."""
        if not self.multi:
            return True
        now = ch["now"]
        near = now is not None and (now[0] in NEAR_BANDS or now[2] or self._nearest_arm(arm) == arm.id)
        told, arm.hand_told = arm.hand_told, now is not None and (arm.hand_told or near)
        return told or near

    def _nearest_arm(self, arm: ArmRun) -> str | None:
        """Of every driven arm, the one whose gripper is nearest the hand `arm` sees."""
        h = arm.state.hand
        if h is None:
            return None
        return min(self.arms.values(), key=lambda o: ((h.x - o.state.arm.x) ** 2 + (h.y - o.state.arm.y) ** 2 +
                                                      (h.z - o.state.arm.z) ** 2, o is not arm)).id

    def _object_arms(self, ch: dict) -> list[str]:
        """An object change concerns the arms that reach it where it was or is, or have a step for it
        running or to come; failing all, the primary arm."""
        if not self.multi:
            return [self.primary]
        pts = [(xy[0] / CM, xy[1] / CM) for xy in (ch.get("from_xy"), ch.get("to_xy")) if xy]
        o = self.world.objects.get(ch.get("object", ""))
        if o is not None:
            pts.append((o.x, o.y))
        steps = self.plan.steps()
        stepping = {self.plan.arm_of(steps[q]) for q in self.plan.queue if steps[q]["object"] == ch.get("object")}
        out = [a for a, arm in self.arms.items() if a in stepping or any(reaches(arm.spec, x, y) for x, y in pts)]
        return out or [self.primary]

    def _robot_touched(self) -> set[str]:
        """Objects the robot itself moved since the detector last looked: those of steps that ended,
        and each running step's object once it is in the gripper or at the step's goal (its end event
        may still be on its way) or, for push, at all; with several arms, anything in any gripper.
        Anything else that moved, someone else moved."""
        out, self._robot_moved = self._robot_moved, set()
        for arm in self.arms.values():
            cur = self.plan.current(arm.id)
            if cur is not None and cur["object"] in arm.state.objects:
                where = arm.state.objects[cur["object"]].where
                goal = P.step_goal(cur)
                if cur["skill"] == "push" or where == "gripper" or (goal is not None and where == goal[1]):
                    out.add(cur["object"])
        if self.multi:
            out |= {o.id for o in self.world.objects.values() if o.where.startswith("gripper:")}
        return out

    def _timers(self) -> None:
        t = self.now()
        for arm in self.arms.values():
            if arm.mode == "holding" and not self.replan_pending and arm.hold_until is not None and t >= arm.hold_until:
                arm.hold_until = None
                self._arm_to(arm, "running")
            if arm.mode == "paused" and t - arm.last_event_t >= self.cfg.paused_idle_s and not arm.in_flight:
                self.event("paused_idle", f"the arm has been paused for {self.cfg.paused_idle_s:.0f} s ({arm.pause_reason}) "
                                          "and nothing has changed", arms=[arm.id])

    # ================================================================ steps
    def _start_next(self, arm: ArmRun) -> None:
        sid = self.plan.next_for(arm.id)
        if sid is None:
            if not self.plan.pending(set()) and all(not a.ctl.busy and a.awaiting is None for a in self.arms.values()):
                self._check_done()
            return
        s = self.plan.steps()[sid]
        wait = self._waits(s)
        if wait != arm.waiting_on:
            arm.waiting_on = wait
            if wait is not None:
                self._note("step_waits", arm=arm.id, step=P.step_text(s, self.names), why=wait)
        if wait is not None:
            return
        name, args = self._call(s)
        why = arm.ctl.precondition(name, args)
        if why is None:
            try:
                arm.ctl.start(name, args)
            except ValueError as e:               # refused at start: same as a failed precondition
                why = str(e)
        txt = P.step_text(s, self.names)
        if why is not None:
            self.plan.could_not_start(sid)
            self.r.steps_failed += 1
            self.event("step_failed", f"step could not start: {txt}: {why}", self._obj(s), sid,
                       arms=[arm.id], gate=True, reason=f"could not start: {why}")
        else:
            self.plan.begin(sid)
            self.r.steps_started += 1
            self._note("step_start", arm=arm.id, step=txt, skill_id=arm.ctl.skill_id)
        self._loop_check(arm, s, sid)

    def _loop_check(self, arm: ArmRun, s: dict, sid: str) -> None:
        if not (arm.loops.record((s["skill"], s["object"], s["target"], s["direction"]))
                or self.plan.tries[sid] >= self.cfg.max_step_failures):
            return
        self.r.loops += 1
        self.plan.tries[sid] = 0
        txt = P.step_text(s, self.names)
        self._note("loop", arm=arm.id, step=txt)
        self._arm_to(arm, "holding")
        if not self.replan_pending:
            ev = Event(0, self.now(), "loop", f"loop: {txt} keeps coming back", self._obj(s), sid)
            self._start_replan("capable_llm", f"a loop: the plan keeps returning to {txt} without progress", ev)

    def _step_ended(self, arm: ArmRun, ok: bool, reason: str | None) -> None:
        s = self.plan.end(arm.id, ok, self.now())
        obj = self._obj(s)
        if obj:
            self._robot_moved.add(obj)
        txt = P.step_text(s, self.names)
        if ok:
            self.r.steps_done += 1
            g = P.step_goal(s)
            objs = arm.state.objects
            after = f"; {self.name(g[0])} is now {where_text(objs[g[0]].where, self.name)}" if g and g[0] in objs else ""
            self.event("step_done", f"step finished: {txt}{after}", obj, s["id"], arms=[arm.id], gate=True)
        else:
            self.r.steps_failed += 1
            self.event("step_failed", f"step failed: {txt}: {reason}", obj, s["id"], arms=[arm.id], gate=True, reason=reason)
        self._note("step_end", arm=arm.id, step=txt, ok=ok, reason=reason)

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
            moves = [s for s in self.plan.plan["steps"] if obj and s["object"] == obj and P.step_goal(s)]
            arm = self.plan.arm_of(moves[-1]) if moves else self.primary
            self.event("plan_done_unmet", "every planned step has run, but these done conditions are false: " +
                       "; ".join(c["text"] for c in unmet), obj, arms=[arm], gate=True, unmet=[c["text"] for c in unmet])

    # ================================================================ decisions
    def _apply_decision(self, ev: Event, out: dict) -> None:
        arm = self._arm_of(ev)
        arm.in_flight -= 1
        c = enforce_gate(out["combined"], self.cfg.stay_local_gate)
        self.r.decisions.append({"t": self.now(), "t_event": ev.t, "event_id": ev.id, "event": ev.kind, "text": ev.text,
                                 "arm": arm.id, "combined": c, "latency_ms": out.get("latency_ms", 0.0)})
        self._note("decision", event_id=ev.id, arm=arm.id, event=ev.kind, right_now=c.right_now, fix=c.fix, route=c.route,
                   reason=c.reason)
        if ev.id == arm.awaiting:
            arm.awaiting = None
        if self.mode in END:
            return
        self.plan.done_answers.update(c.done)
        self._right_now(arm, c.right_now, ev)
        if ev.kind == "user_text" and c.route != "stay_local" and self._once(ev, "words"):
            self._user_words(ev.data.get("text", ""))
        if c.route == "stay_local":
            self._fix(arm, c.fix, ev)
        elif c.route in LLM_ROUTES:
            if self._once(ev, "replan"):
                self._start_replan(c.route, ev.text + ("; plan check: " + "; ".join(c.problems) if c.problems else ""), ev)
        elif c.route == "ask_user":
            self._arm_to(arm, "paused")
            arm.pause_reason = "waiting for the user's answer to a question"
            if self.user is not None and self._once(ev, "ask"):
                self.user.ask(f"About \"{ev.data.get('text', ev.text)}\": what should the end result be?")
        if arm.mode == "holding" and not self.replan_pending:
            arm.hold_until = self.now() + self.cfg.hold_s

    def _right_now(self, arm: ArmRun, answer: str, ev: Event) -> None:
        """Apply the right-now answer to the arm it was asked about. It can hold, pause, resume or
        retarget; nothing a model says stops."""
        if arm.mode == "waiting" and answer not in ("pause", "back_off"):
            return
        action = RIGHT_NOW_ACTION.get(answer)
        if action == "holding" and arm.mode == "running":
            self._arm_to(arm, "holding")
        elif action == "paused" and arm.mode != "paused":
            self._arm_to(arm, "paused")
            # the user's own words, not an interpretation (e12v2: "asked to wait" kept a cautious pause paused)
            arm.pause_reason = (f'the user said "{ev.data.get("text", "")}"' if ev.kind == "user_text" else
                                "a person's hand came near" if "hand" in ev.text else f"paused after: {ev.text}")
        elif action == "retarget":
            arm.ctl.retarget()
        elif action == "running" and arm.mode in ("paused", "holding"):
            arm.hold_until, arm.pause_reason = None, None
            self._arm_to(arm, "running")

    def _user_words(self, text: str) -> None:
        """A typed line the decision took as about the end result (a planner or ask_user route) joins
        the user's task-level words, verbatim, newest last; a transient command never does, even when
        an unsure route sends it to the planner (rules.is_transient)."""
        if text and not is_transient(text):
            self.user_words.append(text)
            self._note("user_words", words=list(self.user_words))

    def _keep_reading(self, out: dict, kind: str) -> None:
        """The reading guard (rules.transient_mentions): a plan whose restated task repeats a transient
        line the user typed ("…after the user said 'hang on a second'…") keeps the previous reading
        instead (the current plan's, or the task before any plan), and the log says why."""
        leaked = transient_mentions(out.get("reading") or "", self.user_messages)
        if not leaked:
            return
        kept = (self.plan.plan.get("reading") if self.plan.plan else "") or self.task
        self._note("reading_kept", request=kind, leaked=leaked, rejected=out.get("reading"), kept=kept,
                   why="the restated task repeated a transient command; the task holds only the end result")
        out["reading"] = kept

    def _fix(self, arm: ArmRun, fix: str, ev: Event) -> None:
        other = self._runnable_other(arm, ev) if fix == "another_step_first" else None
        if self.plan.apply_fix(fix, ev, arm.id, arm.state.arm.holding, other):
            arm.ctl.release()
        if fix != "none":
            self._note("fix", arm=arm.id, fix=fix, queue=list(self.plan.queue))

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

    def _arm_now(self, arm: ArmRun) -> dict:
        """What an arm is doing, for a replan request."""
        st, cur = arm.ctl.status(), self.plan.current(arm.id)
        return {"current_step": cur["id"] if cur else "none",
                "doing": st.phase_text if st is not None else "between steps",
                "while_you_plan": {"holding": "holding still", "paused": "paused"}.get(arm.mode, "carrying on")}

    def _start_replan(self, route: str, why: str, ev: Event | None) -> None:
        if self.plan.plan is None:
            return                                # the first plan is on its way; it will read user_messages
        if ev is not None and ev.kind == "plan_arrived" and self.replans.rejected():
            self.r.loops += 1
            self._note("loop", loop="replans", rejections=self.replans.rejections)
        seq = self._new_request()
        if seq is None:
            return
        arm_now = ({a: self._arm_now(arm) for a, arm in self.arms.items()} if self.multi else
                   self._arm_now(self.arms[self.primary]))
        args = (route, self.task, list(self.user_messages), self.world, self.state, copy.deepcopy(self.plan.plan),
                list(self.plan.queue), dict(self.plan.status), arm_now, why)
        ref, t = self._ref(ev), self.now()
        self._note("llm_request", request="replan", route=route, why=why)
        self._in_thread(lambda: self.planner.replan(*args, ref=ref), lambda res: Planned(seq, t, ev, res))

    def _resume_held(self) -> None:
        """A plan came back: arms holding for it carry on."""
        for arm in self.arms.values():
            if arm.mode == "holding":
                arm.hold_until = None
                self._arm_to(arm, "running")

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
                self._resume_held()
            return
        if res.kind == "plan":
            self._keep_reading(res.plan, "plan")
            self.plan.load(res.plan)
            self.mode = "running"
            for arm in self.arms.values():
                if arm.mode == "waiting":
                    arm.mode = "running"
            self._note("plan_applied", plan=res.plan)
            self.event("plan_arrived", "the first plan arrived: " + res.plan.get("reading", ""))
            return
        self._keep_reading(res.diff, "replan")
        holdings = {a: arm.state.arm.holding for a, arm in self.arms.items()}
        newp, q, check = self.plan.merge(res.diff, msg.t_request, holdings)
        errs = self.planner.check(newp, self.state, self.world, check)
        if errs and self.misfits < 2:
            # the world moved on while the LLM worked: ask again from where things stand now
            self.misfits += 1
            txt = "the new plan no longer fits the world: " + "; ".join(errs[:3])
            self._note("plan_misfit", errors=errs)
            self._start_replan(res.route, txt, Event(0, self.now(), "misfit", txt))
            return
        self.misfits = 0
        for a in self.plan.install(newp, q):
            self.arms[a].ctl.release()
        self._resume_held()
        self._note("plan_applied", diff=res.diff, queue=q)
        self.event("plan_arrived", "a new plan arrived: " + (res.diff.get("reading") or ""))
