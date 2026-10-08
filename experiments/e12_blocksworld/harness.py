"""The closed loop: human -> perception -> apply due Picks -> Arbiter + skill step -> change-driven requests.

Per tick (10 ticks/s of sim time):
  1. the scripted human acts (events fire on triggers; the hand moves)
  2. perception renders the Scene
  3. Picks whose simulated latency has elapsed are applied (gated on confidence), and escalations
     whose Planner delay has elapsed are resolved
  4. the Arbiter decides what the arm does: Spotter beats Sequencer, a pause beats both; the running
     skill takes one step through the Limiter
  5. each layer decides whether this tick deserves a request (change-driven), builds its State and
     Questions, and the backend answers now; the answer applies `latency_ticks` later
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from common import payload

from . import layers as L
from . import plan as P
from .scenarios import Scenario, Utterance
from .skills import EVASIONS, SKILLS, Limiter, SkillFailed, offered
from .world import HAND_CONTACT_CM, LOW_CM, TICKS_PER_S, Noise, Perception, dxy

GATES = {"next_skill": 0.3, "task_status": 0.5, "escalate": 0.5, "action": 0.3, "order": 0.3, "intent": 0.3, "knob": 0.3}
CAUTION = ["continue", "slow", "hold", "back_off", "evade_up"]
HOLDLIKE = {"hold", *EVASIONS}
TASK_SKILLS = {"move_above", "pick", "carry_to", "place", "release", "retreat", "nudge_clear"}
INTERRUPTIBLE = {"move_above", "carry_to", "hold", "survey"}
STOP_WORDS = re.compile(r"\b(stop|freeze)\b", re.I)
PRICE_PER_TOKEN = 0.042e-6


@dataclass
class Variant:
    facts: str = "solved"
    spotter: bool = True
    listener: bool = True

    @property
    def name(self) -> str:
        return self.facts + ("" if self.spotter else "-nospotter") + ("" if self.listener else "-nolistener")


SWEEP = [Variant("solved"), Variant("raw"), Variant("solved", spotter=False), Variant("solved", listener=False)]


@dataclass
class Pending:
    due: int
    layer: str
    answers: dict
    utt: Utterance | None = None
    rid: int = 0


@dataclass
class Metrics:
    ticks: int = 0
    done_tick: int | None = None
    completed: bool = False
    skills: int = 0
    failed: int = 0
    recoveries: int = 0          # failed / interrupted skills followed by a successful task skill
    disturbances: int = 0        # human events that moved the world
    recovered: int = 0           # disturbances whose moved blocks all ended at their goal, task finished
    requests: dict = field(default_factory=lambda: {"sequencer": 0, "spotter": 0, "listener": 0})
    request_ticks: int = 0
    request_errors: int = 0
    escalations: int = 0
    gated: int = 0
    forbidden_touch: int = 0
    hand_contact: int = 0
    floor_stops: int = 0
    limiter_refusals: int = 0
    utt_to_change: list = field(default_factory=list)   # ticks, per utterance that should change behaviour
    utt_missed: int = 0                                  # should have changed behaviour and never did
    utt_spurious: int = 0                                # should not have, and did
    agree: dict = field(default_factory=dict)            # "layer.question" -> [agree, n]
    budget_hit: bool = False     # stopped at --max-requests
    chars: int = 0
    tokens: int = 0
    latency_ms: list = field(default_factory=list)


class Runtime:
    def __init__(self, sc: Scenario, backend, variant: Variant, latency_ticks: int | str = 2, planner_delay_s: float = 2.0,
                 noise: Noise | None = None, seed: int = 0, max_s: float = 300.0, log=None, capture: dict | None = None,
                 escalation: str = "oracle", max_requests: int = 600):
        self.sc, self.world, self.plan = sc, sc.world, sc.plan
        self.max_requests = max_requests       # cost insurance for live runs: the scenario stops (incomplete) here
        self.backend, self.variant = backend, variant
        self.latency_ticks, self.planner_delay = latency_ticks, int(round(planner_delay_s * TICKS_PER_S))
        self.max_ticks = int(max_s * TICKS_PER_S)
        self.log, self.capture, self.escalation = log, capture, escalation
        self.knobs = P.default_knobs(self.plan)
        self.deferred: set[str] = set()
        self.limiter = Limiter(self.plan)
        self.perception = Perception(noise, seed)
        self.scene = self.truth_scene = None
        self.skill = None                      # (key, generator)
        self.spot_action, self.spot_streak, self.evade_pending = "continue", {}, None
        self.order_off_streak: dict[str, int] = {}
        self.paused = False
        self.pending: list[Pending] = []
        self.escalations: list = []            # (due, kind, utt, resolution)
        self.seq_inflight = self.esc_seq_open = False
        self.seq_gated = self.seq_disagree = 0
        self.unrecovered = 0
        self.spot_snap, self.spot_last_tick = None, -10 ** 9
        self.moved_at: dict[str, int] = {}
        self.disturbed: list[list[str]] = []   # per disturbance: the bound blocks it moved
        self.order_bound = {o["enforce"]["near"] for o in self.limiter.conditional.values() if "near" in o["enforce"]}
        self.arm_moved: set[str] = set()       # blocks the arm itself held or dragged since the Spotter last looked
        self.touching = self.contact = False
        self.finished = False
        self.tick = 0
        self.rid = 0
        self.m = Metrics()
        self._tick_rec: dict = {"requests": [], "applied": [], "events": []}
        self._claude = None

    # ---------------------------------------------------------------- text helpers (Scene names)
    def name(self, bid) -> str:
        e = (self.scene or {}).get("blocks", {}).get(bid)
        if e:
            return e["name"]
        b = self.world.blocks.get(bid)
        return b and (f"block {b.number}" if b.number is not None else f"{b.colour} block") or str(bid)

    def dest_name(self, dest: str) -> str:
        kind, _, arg = dest.partition(":")
        return {"slot": f"tray slot {arg}", "bin": f"the {arg} bin", "park": "a free spot on the table"}.get(kind, dest)

    def where_text(self, where: str, e: dict | None = None) -> str:
        if where == "table":
            return f"on the table at ({e['x']:.0f}, {e['y']:.0f}) cm" if e else "on the table"
        if where == "gripper":
            return "in the gripper"
        return "in " + self.dest_name(where)

    def skill_text(self, key: str | None) -> str:
        if not key:
            return "none"
        name, _, arg = key.partition(":")
        if not arg:
            return name
        return f"{name}({self.dest_name(arg) if name == 'carry_to' else self.name(arg)})"

    @property
    def recently_moved(self) -> set:
        return {b for b, t in self.moved_at.items() if self.tick - t <= TICKS_PER_S}

    def pace(self) -> str:
        slow = self.knobs.get("pace") == "slow" or self.spot_action == "slow" or self.limiter.pace_cap(self.scene, self.world.arm)
        return "slow" if slow else "normal"

    # ---------------------------------------------------------------- the loop
    def run(self) -> Metrics:
        w = self.world
        self._perceive()
        for t in range(self.max_ticks):
            self.tick = w.tick = t
            self._tick_rec = {"requests": [], "applied": [], "events": []}
            self.human()
            self._perceive()
            for p in sorted([p for p in self.pending if p.due <= t], key=lambda p: (p.due, p.rid)):
                self.pending.remove(p)
                self.apply(p)
            for e in [e for e in self.escalations if e[0] <= t]:
                self.escalations.remove(e)
                self.resolve(*e[1:])
            if self.finished:
                break
            if sum(self.m.requests.values()) >= self.max_requests:
                self.m.budget_hit = True
                break
            self.act()
            if self.world.arm.holding:
                self.arm_moved.add(self.world.arm.holding)
            if self.skill and self.skill[0].startswith("nudge_clear:"):
                self.arm_moved.add(self.skill[0].split(":", 1)[1])
            self._perceive()
            self.check_violations()
            self.requests()
            self.m.request_ticks += bool(self._tick_rec["requests"])
            self._log_tick()
        self.m.ticks = self.tick + 1
        final = {**P.default_knobs(self.plan), **self.sc.final_knobs}
        truth = Perception.truth(w)
        self.m.completed = self.finished and P.complete(self.plan, final, truth, w.arm)
        goals = P.goals(self.plan, final, truth)
        for ids in self.disturbed:   # recovered: every block it moved ended at its goal and the task finished
            self.m.recovered += self.finished and all(truth["blocks"][b]["where"] == goals[b] for b in ids)
        self.m.floor_stops, self.m.limiter_refusals = self.limiter.floor_stops, self.limiter.refusals
        for ev in self.sc.events:
            u = ev.utt
            if u is None:
                continue
            if u.expect_change:
                if u.changed_at is None:
                    self.m.utt_missed += 1
                else:
                    self.m.utt_to_change.append(u.changed_at - u.tick)
            elif u.changed_at is not None:
                self.m.utt_spurious += 1
        return self.m

    def _perceive(self):
        self.truth_scene = Perception.truth(self.world)
        self.scene = self.perception.observe(self.world)

    # ---------------------------------------------------------------- the human
    def human(self):
        w = self.world
        for ev in self.sc.events:
            if ev.fired or not ev.when(w):
                continue
            ev.fired = True
            before = {b.id: (b.x, b.y) for b in w.blocks.values()}
            out = ev.do(w)
            moved = [b.id for b in w.blocks.values() if dxy(before[b.id], (b.x, b.y)) > 2]
            for b in moved:
                self.moved_at[b] = self.tick
            if ev.disturbs:
                self.m.disturbances += 1
                self.disturbed.append([b for b in moved if b in self.plan["bindings"]["blocks"]])
            self._tick_rec["events"].append(ev.name)
            if isinstance(out, Utterance):
                ev.utt = out
                out.tick = self.tick
                self.on_utterance(out)
        if w.hand is not None:
            w.hand.step()
            if w.hand.phase == "gone":
                w.hand = None

    def on_utterance(self, utt: Utterance):
        if STOP_WORDS.search(utt.text):       # STOP never depends on a model
            self.paused = True
            self._interrupt("stop word")
            utt.changed_at = utt.changed_at if utt.changed_at is not None else self.tick
        if self.variant.listener:
            self.ask("listener", *L.listener(self, self.variant.facts, utt), reason="utterance", utt=utt)
        else:
            self.escalate("listener", utt, "no Listener: every utterance goes to the Planner")

    # ---------------------------------------------------------------- requests (change-driven)
    def requests(self):
        if self.finished:
            return
        if self.seq_allowed() and self.skill is None and not self.seq_inflight and not self.esc_seq_open:
            self.seq_inflight = True
            self.ask("sequencer", *L.sequencer(self, self.variant.facts), reason="arm idle")
        if self.variant.spotter:
            why = self.spotter_should_ask()
            if why:
                self.ask("spotter", *L.spotter(self, self.variant.facts), reason=why)

    def seq_allowed(self) -> bool:
        return not self.finished and not self.paused and self.spot_action not in HOLDLIKE

    def _spot_snapshot(self) -> dict:
        s = {b: (e["x"], e["y"]) for b, e in self.scene["blocks"].items() if e["where"] != "gripper"}
        h = self.scene.get("hand")
        s["hand"] = (h["x"], h["y"]) if h else None
        arm = self.world.arm
        s["near"] = frozenset(b for b in self.order_bound
                              if b in self.scene["blocks"] and dxy((arm.x, arm.y), (self.scene["blocks"][b]["x"], self.scene["blocks"][b]["y"])) < 20)
        return s

    def spotter_should_ask(self) -> str | None:
        snap = self._spot_snapshot()
        if self.spot_snap is None:
            self.spot_snap = snap
            return None
        old, arm = self.spot_snap, self.world.arm
        why = None
        if (old["hand"] is None) != (snap["hand"] is None):
            why = "hand appeared" if snap["hand"] else "hand vanished"
        elif snap["hand"] and dxy(old["hand"], snap["hand"]) > 2:
            why = "hand moved"
        elif snap["near"] != old["near"]:
            why = "the gripper came near / left " + ", ".join(self.name(b) for b in sorted(snap["near"] ^ old["near"]))
        hd = L.heading(self, self.scene)
        for b, xy in snap.items():
            if b in ("hand", "near") or b not in old or why:
                continue
            if dxy(old[b], xy) > 2:
                if b in self.arm_moved:
                    old[b] = xy            # the arm's own doing: not a disturbance
                    self.arm_moved.discard(b)
                elif min(dxy((arm.x, arm.y), xy), dxy((arm.x, arm.y), old[b])) < 30 or (hd and dxy(hd, xy) < 10):
                    why = f"{self.name(b)} moved near the arm"
                else:
                    old[b] = xy            # far from the arm: absorb silently
        if why:
            self.spot_snap = snap
            return why
        watching = snap["hand"] is not None or self.spot_action != "continue" or self.limiter.active or snap["near"]
        if watching and self.tick - self.spot_last_tick >= TICKS_PER_S:
            self.spot_snap = snap
            return "1 s silence while watching"
        return None

    def ask(self, layer, state, qs, truth, reason="", utt=None):
        self.rid += 1
        body = payload(state, qs)
        self.m.chars += len(json.dumps(body))
        self.m.requests[layer] += 1
        if layer == "spotter":
            self.spot_last_tick = self.tick
        if self.capture is not None and not self.capture.get(layer, {}).get("telling"):
            # keep the first request made after the scenario's first event (or, with no events, while
            # holding a block): tick 0 of every scenario looks the same
            fired = [e for e in self.sc.events if e.fired]
            telling = bool(fired) if self.sc.events else bool(self.world.arm.holding)
            self.capture[layer] = {"scenario": self.sc.name, "variant": self.variant.name, "tick": self.tick, "reason": reason,
                                   "telling": telling, "request": body, "truth": truth}
        answers, ms, tok, meta = self.backend.answer(layer, state, qs, truth)
        rec = {"layer": layer, "rid": self.rid, "reason": reason, "state": state, "questions": qs, "truth": truth, "meta": meta}
        self._tick_rec["requests"].append(rec)
        if answers is None:
            self.m.request_errors += 1
            if layer == "sequencer":
                self.seq_inflight = False
            elif layer == "listener":
                self.escalate("listener", utt, "Listener request failed")
            return
        self.m.tokens += tok
        self.m.latency_ms.append(ms)
        lat = self.latency_ticks if isinstance(self.latency_ticks, int) else max(1, math.ceil(ms / (1000 / TICKS_PER_S)))
        rec["answers"], rec["due"] = answers, self.tick + lat
        for q in qs:
            g = q.split(":")[0] if q.startswith("order:") else q
            a = self.m.agree.setdefault(f"{layer}.{g}", [0, 0])
            a[0] += (answers.get(q) or {}).get("choice") in truth[q]
            a[1] += 1
        self.pending.append(Pending(self.tick + lat, layer, answers, utt, self.rid))

    # ---------------------------------------------------------------- applying Picks
    def _pick(self, answers, q) -> tuple[str | None, float]:
        a = answers.get(q) or {}
        c = a.get("confidence")
        return a.get("choice"), 1.0 if c is None else c

    def _gate(self, q) -> float:
        return GATES.get(q.split(":")[0], 0.0)

    def apply(self, p: Pending):
        effect = getattr(self, f"apply_{p.layer}")(p)
        self._tick_rec["applied"].append({"layer": p.layer, "rid": p.rid,
                                          "picks": {q: [a.get("choice"), a.get("confidence")] for q, a in p.answers.items()},
                                          "effect": effect})

    def apply_sequencer(self, p: Pending) -> str:
        self.seq_inflight = False
        if self.skill is not None or not self.seq_allowed():
            return "stale: arm busy or held"
        a = p.answers
        esc, c = self._pick(a, "escalate")
        if esc == "ask_a_smarter_model" and c >= GATES["escalate"]:
            self.escalate("sequencer", None, "Sequencer asked for a smarter model")
            return "escalated"
        code_done = P.complete(self.plan, self.knobs, self.scene, self.world.arm)
        ts, c = self._pick(a, "task_status")
        if code_done:
            if ts == "complete" and c >= GATES["task_status"]:
                self.finish()
                return "task complete"
            self.seq_disagree += 1
            if self.seq_disagree >= 3:
                self.escalate("sequencer", None, "code says complete, the Sequencer does not")
                return "escalated: disagree on done"
        ns, c = self._pick(a, "next_skill")
        if c < GATES["next_skill"]:
            self.m.gated += 1
            self.seq_gated += 1
            if self.seq_gated >= 3:
                self.seq_gated = 0
                self.escalate("sequencer", None, "3 low-confidence picks in a row")
                return "gated: escalated"
            return "gated"
        self.seq_gated = 0
        if ns not in offered(self):
            return f"stale option {ns}"
        self.dispatch(ns)
        return f"dispatch {ns}"

    def apply_spotter(self, p: Pending) -> str:
        a = p.answers
        act, c = self._pick(a, "action")
        effect = []
        if act in CAUTION and c >= GATES["action"]:
            effect.append(self._set_spot(act))
        elif act in CAUTION:
            self.m.gated += 1
            effect.append("action gated")
        for q in a:
            if not q.startswith("order:"):
                continue
            oid = q.split(":", 1)[1]
            ch, c = self._pick(a, q)
            if oid not in self.limiter.conditional or c < GATES["order"]:
                continue
            if ch == "on":
                self.order_off_streak[oid] = 0
                if oid not in self.limiter.active:
                    self.limiter.active.add(oid)
                    effect.append(f"order on: {oid}")
            elif ch == "off" and oid in self.limiter.active:
                self.order_off_streak[oid] = self.order_off_streak.get(oid, 0) + 1
                if self.order_off_streak[oid] >= 2:      # switching a constraint off needs two picks in a row
                    self.limiter.active.discard(oid)
                    effect.append(f"order off: {oid}")
        return "; ".join(e for e in effect if e) or "no change"

    def _set_spot(self, new: str) -> str:
        cur = self.spot_action
        running_evasion = self.skill is not None and self.skill[0] in EVASIONS
        if new == cur:
            self.spot_streak = {}
            if new in EVASIONS and not running_evasion:
                self.evade_pending = new
                return f"spotter: {new} again"
            return ""
        if CAUTION.index(new) < CAUTION.index(cur):       # less cautious: needs two picks in a row
            n = self.spot_streak.get(new, 0) + 1
            self.spot_streak = {new: n}
            if n < 2:
                return f"spotter: {new} pending (1/2)"
        self.spot_streak = {}
        self.spot_action = new
        if new in EVASIONS:
            self.evade_pending = new
        return f"spotter: {cur} -> {new}"

    def apply_listener(self, p: Pending) -> str:
        a, utt = p.answers, p.utt
        esc, c = self._pick(a, "escalate")
        if esc == "ask_a_smarter_model" and c >= GATES["escalate"]:
            self.escalate("listener", utt, "Listener asked for a smarter model")
            return "escalated"
        intent, c = self._pick(a, "intent")
        if intent not in L.INTENTS or c < GATES["intent"]:
            self.m.gated += 1
            self.escalate("listener", utt, "low-confidence intent")
            return "gated: escalated"
        knobs = {}
        for k in self.plan["knobs"]:
            v, c = self._pick(a, f"knob:{k}")
            if v is not None and c >= GATES["knob"]:
                knobs[k] = v
        when, _ = self._pick(a, "when")
        return self.apply_intent(intent, knobs, when, utt, "listener")

    def apply_intent(self, intent: str, knobs: dict, when: str, utt: Utterance | None, source: str) -> str:
        changed, effect = False, f"intent {intent}"
        if intent == "adjust":
            ch = {k: v for k, v in knobs.items() if v != P.UNCHANGED and k in self.plan["knobs"]
                  and v in self.plan["knobs"][k]["values"] and v != self.knobs.get(k)}
            if not ch:
                if source == "listener":
                    self.escalate("listener", utt, "adjust, but no knob changes")
                    return "adjust with no knob change: escalated"
                return "adjust with no knob change"
            for k, v in ch.items():
                if k == "skip":
                    tgt = self.world.arm.target
                    if tgt and tgt not in self.deferred and self.world.blocks[tgt].where != "gripper":
                        self.deferred.add(tgt)
                        changed = True
                else:
                    self.knobs[k] = v
                    changed = True
            effect += f" {ch}"
            if changed and when == "now" and self.skill and self.skill[0].split(":")[0] in INTERRUPTIBLE:
                self._interrupt("plan changed")
        elif intent in ("stop", "pause"):
            changed, self.paused = not self.paused, True
        elif intent == "resume":
            changed, self.paused = self.paused, False
        elif intent == "new_task" and source == "listener":
            self.escalate("listener", utt, "new task")
            return "new_task: escalated"
        if changed and utt is not None and utt.changed_at is None:
            utt.changed_at = self.tick
        return effect + (" (behaviour changed)" if changed else "")

    # ---------------------------------------------------------------- escalation to the Planner
    def escalate(self, kind: str, utt: Utterance | None, why: str):
        self.m.escalations += 1
        if kind == "sequencer":
            self.esc_seq_open = True
        res, delay = None, self.planner_delay
        if self.escalation == "claude":
            res, ms = self._claude_resolve(kind, utt)
            delay = max(delay, math.ceil(ms / (1000 / TICKS_PER_S)))
        self.escalations.append((self.tick + delay, kind, utt, res))
        self._tick_rec.setdefault("escalations", []).append({"kind": kind, "why": why, "due": self.tick + delay})

    def resolve(self, kind: str, utt: Utterance | None, res: dict | None):
        """Oracle Planner: resolves from ground truth after the delay. (claude: from its answers.)"""
        arm = self.world.arm
        if kind == "sequencer":
            self.esc_seq_open = False
            self.seq_disagree = 0
            done = P.complete(self.plan, self.knobs, self.truth_scene, arm)
            ns = res.get("next_skill") if res else P.next_skill(self.plan, self.knobs, self.truth_scene, arm, self.deferred)
            if (res and res.get("task_status") == "complete" and done) or (res is None and done):
                self.finish()
            elif self.skill is None and self.seq_allowed() and ns in offered(self):
                self.dispatch(ns)
            effect = f"planner: {ns}"
        else:
            if utt is None:
                return
            r = res or utt.resolution()
            effect = "planner: " + self.apply_intent(r["intent"], r["knobs"], r.get("when", "now"), utt, "planner")
        self._tick_rec["applied"].append({"layer": "planner", "effect": effect})

    def _claude_resolve(self, kind, utt):
        from .backends import Claude
        if self._claude is None:
            self._claude = Claude()
        if kind == "sequencer":
            st, qs, tr = L.sequencer(self, "solved")
        else:
            st, qs, tr = L.listener(self, "solved", utt)
        st = {"role": "You are the planner the fast layer escalated to. Resolve it.", **st}
        ans, ms, _, _ = self._claude.answer(kind, st, qs, tr)
        if not ans:
            return None, ms
        if kind == "sequencer":
            return {"next_skill": ans["next_skill"]["choice"], "task_status": ans["task_status"]["choice"]}, ms
        knobs = {k: ans[f"knob:{k}"]["choice"] for k in self.plan["knobs"]}
        return {"intent": ans["intent"]["choice"], "knobs": knobs, "when": ans["when"]["choice"]}, ms

    # ---------------------------------------------------------------- Arbiter + skills
    def dispatch(self, key: str):
        name = key.split(":")[0]
        why = self.limiter.check_skill(key, self.scene)
        if why:
            self.world.arm.last_result = f"{self.skill_text(key)}: failed: refused by the Limiter ({why})"
            self.m.failed += 1
            self.unrecovered += 1
            return
        _, _, arg = key.partition(":")
        self.skill = (key, SKILLS[name](self, arg))
        self.world.arm.skill = key
        self.m.skills += 1

    def _finish_skill(self, result: str, ok: bool):
        key = self.skill[0]
        self.skill, self.world.arm.skill = None, None
        self.world.arm.last_result = f"{self.skill_text(key)}: {result}"
        if not ok:
            self.m.failed += 1
            self.unrecovered += 1
        elif key.split(":")[0] in TASK_SKILLS and self.unrecovered:
            self.m.recoveries += self.unrecovered
            self.unrecovered = 0

    def _interrupt(self, why: str):
        if self.skill:
            self.skill[1].close()
            self._finish_skill(f"failed: interrupted ({why})", ok=False)

    def act(self):
        if self.evade_pending:
            a, self.evade_pending = self.evade_pending, None
            if self.skill and self.skill[0] not in EVASIONS:
                self._interrupt(f"Spotter: {a}")
            if not self.skill:
                self.dispatch(a)
        if not self.skill:
            return
        evasion = self.skill[0] in EVASIONS
        if not evasion and (self.paused or self.spot_action in HOLDLIKE):
            return   # Arbiter: the Spotter (or a pause) beats the Sequencer's skill
        try:
            next(self.skill[1])
        except StopIteration as e:
            self._finish_skill(e.value or "done", ok=True)
        except SkillFailed as e:
            self._finish_skill(f"failed: {e}", ok=False)

    def finish(self):
        self.finished = True
        self.m.done_tick = self.tick

    def check_violations(self):
        w, arm = self.world, self.world.arm
        touching = arm.holding in self.limiter.forbidden or (arm.z < LOW_CM and any(
            w.blocks[f].where != "gripper" and dxy(arm.xyz, (w.blocks[f].x, w.blocks[f].y)) < arm.footprint
            for f in self.limiter.forbidden))
        self.m.forbidden_touch += touching and not self.touching
        self.touching = touching
        contact = w.hand is not None and math.dist(arm.xyz, w.hand.xyz) < HAND_CONTACT_CM
        self.m.hand_contact += contact and not self.contact
        self.contact = contact

    def _log_tick(self):
        if self.log is None:
            return
        arm = self.world.arm
        self.log.write(kind="tick", scenario=self.sc.name, backend=self.backend.name, variant=self.variant.name, tick=self.tick,
                       arm=[round(arm.x, 1), round(arm.y, 1), round(arm.z, 1)], holding=arm.holding, skill=arm.skill,
                       last_result=arm.last_result, spotter=self.spot_action, orders_on=sorted(self.limiter.active),
                       knobs=self.knobs, paused=self.paused, hand=(self.world.hand and [round(v, 1) for v in self.world.hand.xyz]),
                       **{k: v for k, v in self._tick_rec.items() if v})


def run_one(sc: Scenario, backend, variant: Variant, **kw) -> Metrics:
    rt = Runtime(sc, backend, variant, **kw)
    if rt.log is not None:
        rt.log.write(kind="run", scenario=sc.name, backend=backend.name, variant=variant.name, plan=sc.plan,
                     latency_ticks=rt.latency_ticks, planner_delay_ticks=rt.planner_delay)
    m = rt.run()
    if rt.log is not None:
        rt.log.write(kind="summary", scenario=sc.name, backend=backend.name, variant=variant.name, metrics=m.__dict__)
    return m
