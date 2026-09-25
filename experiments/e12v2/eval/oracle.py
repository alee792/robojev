"""The evaluation oracle. The system under test (core/, the controls) must never import this.

  Goal            where each object must end up for one task version, plus its constraints
  oracle_plan()   a correct plan from the TRUE world for a goal (the mock LLM's plans; the ceiling)
  truth()         the acceptable answers for one decision (per group, and per progress / plan-check
                  Noul), from the true world, the scenario's script and the harness's view
  achieved()      are the goal's conditions true in the true world (completion)
Thresholds here (hand distances, the gripper footprint) are the oracle's own; the Jev questions
describe the same situations in words and code-computed bands, without these numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core import plan as P
from ..core.data import dxy, seg_dist
from ..sim.world import FINGER_CM

HAND_BACK_OFF_CM = 10.0     # the oracle's own numbers
HAND_WATCH_CM = 45.0
HAND_CLEAR_CM = 25.0
PATH_CM = 8.0


@dataclass
class Goal:
    assign: dict                                   # object -> place id | "on:<object>" | "person"
    constraints: list = field(default_factory=list)  # plan-format constraints (no ids)
    stays: list = field(default_factory=list)      # objects the task names that should not move (a tower's base)

    def mentions(self) -> set:
        m = set(self.assign) | {c["object"] for c in self.constraints} | set(self.stays)
        m |= {v[3:] for v in self.assign.values() if v.startswith("on:")}
        return m

    def done_when(self) -> list[dict]:
        out = []
        for i, (o, p) in enumerate(self.assign.items(), 1):
            if p.startswith("on:"):
                out.append({"id": f"d{i}", "text": f"{o} is on {p[3:]}", "object": o, "relation": "on", "target": p[3:]})
            elif p == "person":
                out.append({"id": f"d{i}", "text": f"the person is holding {o}", "object": o, "relation": "in", "target": "person"})
            else:
                out.append({"id": f"d{i}", "text": f"{o} is in {p}", "object": o, "relation": "in", "target": p})
        for j, c in enumerate(self.constraints, len(out) + 1):
            if c["kind"] == "keep_out_of":
                out.append({"id": f"d{j}", "text": f"{c['object']} is not in {c['place']}", "object": c["object"],
                            "relation": "not_in", "target": c["place"]})
        return out

    def plan_constraints(self) -> list[dict]:
        return [{"id": f"c{i}", **c} for i, c in enumerate(self.constraints, 1)]


def achieved(goal: Goal, truth_state) -> bool:
    for c in goal.done_when():
        if not P.measure(c, truth_state):
            return False
    return truth_state.arm.holding is None


# ---------------------------------------------------------------- oracle plans


def _push_dir(o, f) -> str:
    dx, dy = o.x - f.x, o.y - f.y
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "away_from_robot" if dy > 0 else "toward_robot"


def oracle_plan(goal: Goal, st, holding: str | None = None, reading: str = "") -> dict:
    """A correct step list from the true state `st`: the held object first, then clear what is in the
    way (constraint breakers, wrong stack tops, slot blockers), then place in slot / stack order."""
    where = {o.id: o.where for o in st.objects.values()}
    if holding:
        where[holding] = "gripper"
    places = st.places
    forb = {c["object"] for c in goal.constraints if c["kind"] == "dont_touch"}
    kout = [(c["object"], c["place"]) for c in goal.constraints if c["kind"] == "keep_out_of"]
    steps: list[dict] = []
    pushed = set()

    def inside(o, p):
        pl = places.get(p)
        return where.get(o) in pl.members if pl is not None and pl.kind == "group" else where.get(o) == p

    def occupant(p, but):
        return next((o for o, w in where.items() if w == p and o != but), None)

    def top(o, but=None):
        return next((x for x, w in where.items() if w == f"on:{o}" and x != but), None)

    def add(skill, o, t="none", d="none"):
        steps.append({"id": f"s{len(steps) + 1}", "skill": skill, "object": o, "target": t, "direction": d})

    def maybe_push(o):
        ob = st.objects[o]
        for f in forb:
            fo = st.objects.get(f)
            if fo and o not in pushed and where.get(o) == "table" and dxy((ob.x, ob.y), (fo.x, fo.y)) < FINGER_CM + 1.0:
                add("push", o, d=_push_dir(ob, fo))
                pushed.add(o)

    def move(o, dest):
        if where[o] != "gripper":
            maybe_push(o)
        if dest.startswith("on:"):
            add("stack_on", o, dest[3:])
        elif dest == "person":
            add("hand_over", o, "person")
        else:
            add("move_object", o, dest)
        where[o] = dest

    def ready(o, dest):
        if dest.startswith("on:"):
            base = dest[3:]
            ok_base = base not in goal.assign or where.get(base) == goal.assign[base]
            return ok_base and top(base, but=o) is None and where.get(base) not in ("gripper", "person")
        pl = places.get(dest)
        return pl is None or pl.capacity != 1 or occupant(dest, o) is None

    def wrong(o):
        return o in goal.assign and where.get(o) != goal.assign[o]

    # 1. the held object
    if holding:
        d = goal.assign.get(holding)
        if d and ready(holding, d):
            move(holding, d)
        else:
            bad = any(inside(holding, kp) for ko, kp in kout if ko == holding)
            move(holding, "table" if not bad else "table")
    # 2. constraint breakers out
    for ko, kp in kout:
        if inside(ko, kp):
            if top(ko):
                move(top(ko), "table")
            move(ko, "table")
    # 3. stacks: take off tops that are wrong, top-down
    changed = True
    while changed:
        changed = False
        for o, w in list(where.items()):
            if w.startswith("on:") and top(o) is None and (goal.assign.get(o) != w or wrong(w[3:])):
                move(o, "table")
                changed = True
    # 4. place, in slot order / bottom-up; break blocker cycles via the table
    order = sorted([o for o in goal.assign], key=lambda o: _rank(goal.assign[o], goal))
    for _ in range(4 * len(order) + 4):
        todo = [o for o in order if wrong(o)]
        if not todo:
            break
        o = next((x for x in todo if ready(x, goal.assign[x])), None)
        if o is not None:
            move(o, goal.assign[o])
            continue
        d = goal.assign[todo[0]]
        blocker = occupant(d, todo[0]) if not d.startswith("on:") else top(d[3:], but=todo[0])
        if blocker is None or blocker in forb:
            break
        move(blocker, "table")
    return {"reading": reading or "oracle plan", "steps": steps, "done_when": goal.done_when(), "constraints": goal.plan_constraints()}


def _rank(dest: str, goal: Goal) -> tuple:
    if dest.startswith("tray_slot_"):
        return (0, int(dest.rsplit("_", 1)[1]))
    if dest.startswith("on:"):
        depth, b = 0, dest[3:]
        while goal.assign.get(b, "").startswith("on:"):
            depth, b = depth + 1, goal.assign[b][3:]
        return (1, depth)
    if dest == "person":
        return (3, 0)
    return (2, 0)


# ---------------------------------------------------------------- decision truth


def _hand(truth_state, heading):
    h, a = truth_state.hand, truth_state.arm
    if h is None:
        return None
    d = math.dist((h.x, h.y, h.z), (a.x, a.y, a.z))
    nd = math.dist((h.x + h.vx, h.y + h.vy, h.z), (a.x, a.y, a.z))
    in_path = heading is not None and seg_dist((h.x, h.y), (a.x, a.y), heading) < PATH_CM and dxy((a.x, a.y), heading) > 1
    return {"d": d, "approaching": nd < d - 0.5, "in_path": in_path, "held_out": h.held_out}


def _hand_clear(hf, handing_over: bool) -> bool:
    return hf is None or (hf["held_out"] and handing_over) or (hf["d"] >= HAND_CLEAR_CM and not hf["approaching"] and not hf["in_path"])


def _latest_step(view, oid):
    best = None
    for s in (view.plan or {}).get("steps", []):
        if s["object"] == oid and P.step_goal(s):
            st = view.status.get(s["id"])
            if st in (None, "dropped") and s["id"] not in view.queue:
                continue
            best = s
    return best


def truth(ev, view, scenario, world) -> dict:
    """Acceptable answers (lists, preferred first) for right_now / in_plan_fix / route, and booleans for
    done.<id> and check.<object>."""
    ts = world.truth()
    goal = scenario.goal_for(view.user_messages)
    heading = view.path_to
    hf = _hand(ts, heading)
    cur = view.current
    handing_over = bool(cur and cur["skill"] == "hand_over")
    user_wait = (view.pause_reason or "").startswith("the user asked")
    if user_wait:   # lifted if the user's latest wait/go line is a "go on"
        kinds = [scenario.utts[m].kind for m in view.user_messages if m in scenario.utts and scenario.utts[m].kind in ("wait", "go_on")]
        user_wait = not kinds or kinds[-1] == "wait"
    hand_pause = view.mode == "paused" and not user_wait and "waiting for the user" not in (view.pause_reason or "")
    resumable = view.mode == "paused" and hand_pause and _hand_clear(hf, handing_over)
    stay = ["carry_on"] + (["resume"] if resumable else [])
    T = {"right_now": ["resume", "carry_on"] if resumable else ["carry_on"], "in_plan_fix": ["none"], "route": ["stay_local"]}
    llm = ["fast_llm", "capable_llm"]
    o = ev.object
    k = ev.kind

    if k == "plan_arrived":
        plan = view.plan
        m = P.mentioned(plan)
        for oid in ts.objects:
            if oid not in m:
                T[f"check.{oid}"] = oid in goal.mentions()
        # does the plan reach the goal? (symbolic run of its queue from the true state)
        ok = _plan_reaches(plan, view.queue, ts, goal)
        if not ok or any(v for kk, v in T.items() if kk.startswith("check.")):
            T["route"] = llm
    elif k == "step_done":
        if view.plan:
            for c in view.plan["done_when"]:
                if c["relation"] == "other":
                    T[f"done.{c['id']}"] = True
    elif k == "step_failed":
        r = ev.data.get("reason", "")
        if "must not be touched" in r or "constraint" in r:
            T.update(right_now=["hold", "carry_on"], route=llm)
        elif "occupied by" in r or "on top of it" in r:
            blocker = _blocker(r, ts)
            handled = blocker is not None and _pending_for(view, blocker)
            if handled and view.runnable_other:
                T.update(in_plan_fix=["another_step_first"])
            else:
                T.update(route=llm)
        elif "not in view" in r or "held by the person" in r or "gripper is holding" in r or "not a place" in r:
            T.update(in_plan_fix=["skip"], route=llm)
        else:   # nothing grasped (moved), a hand in the way, a handover that timed out
            T.update(in_plan_fix=["retry"])
        if hf is not None and not _hand_clear(hf, handing_over):
            T["right_now"] = ["pause", "carry_on"] if view.mode != "paused" else ["carry_on", "pause"]
    elif k == "scene_change":
        ch = ev.data.get("change", {})
        if ch.get("what") == "hand":
            if hf is None:
                T["right_now"] = ["resume"] if hand_pause else ["carry_on"]
            elif hf["held_out"] and handing_over:
                T["right_now"] = stay if not hand_pause else ["resume", "carry_on"]
            elif hf["d"] < HAND_BACK_OFF_CM:
                T["right_now"] = ["back_off", "pause"] if view.mode != "paused" or view.pause_reason.startswith("a person") else ["back_off", "pause", "carry_on"]
            elif (hf["approaching"] and hf["d"] < HAND_WATCH_CM) or hf["in_path"]:
                T["right_now"] = ["pause", "carry_on"] if view.mode == "paused" else ["pause"]
            else:
                T["right_now"] = ["resume", "carry_on"] if resumable else ["carry_on"]
        elif o is not None:
            forb = {c["object"] for c in goal.constraints if c["kind"] == "dont_touch"}
            kout = {c["object"]: c["place"] for c in goal.constraints if c["kind"] == "keep_out_of"}
            step = _latest_step(view, o)
            st = view.status.get(step["id"]) if step else None
            obj = ts.objects.get(o)
            if cur and cur.get("object") == o and not view.grasped and st == "running":
                T["right_now"] = ["re_target"]
            elif o in forb and _crowds_pending(view, ts, o):
                T.update(right_now=["hold", "carry_on"], route=llm)
            elif o in kout and obj is not None and ts.in_place(o, kout[o]):
                if step is not None and st in ("done", "failed", "skipped"):
                    T["in_plan_fix"] = ["re_queue"]
                elif step is None or st not in ("pending", "running"):
                    T["route"] = llm
            elif step is not None and st == "done" and obj is not None and obj.where != P.step_goal(step)[1]:
                T["in_plan_fix"] = ["re_queue"]
            elif step is None and o in goal.assign and obj is not None and obj.where != goal.assign[o]:
                T["route"] = llm
    elif k == "user_text":
        u = scenario.utts.get(ev.data.get("text", ""))
        kind = u.kind if u else None
        if u is None:   # an answer to a question: treat as its correction, or as go on
            corr = next((x for x in scenario.utts.values() if x.clarify == ev.data.get("text")), None)
            kind = "correction" if corr else "go_on"
        if kind == "correction":
            wrong = _running_wrong(view, goal, ts)
            T.update(right_now=["hold"] if wrong else ["hold", "carry_on"], route=llm)
            if view.mode == "paused":
                T["right_now"] = ["hold", "resume", "carry_on"]
        elif kind == "wait":
            T["right_now"] = ["pause"] if view.mode != "paused" else ["carry_on", "pause"]
        elif kind == "go_on":
            T["right_now"] = ["resume"] if view.mode in ("paused", "holding") and _hand_clear(hf, handing_over) else ["carry_on", "resume"]
        # chatter: the defaults (carry on, no fix, stay local)
    elif k == "paused_idle":
        T["right_now"] = ["resume"] if resumable else ["carry_on", "pause"]
    elif k == "plan_done_unmet":
        step = _latest_step(view, o) if o else None
        if step is not None and view.status.get(step["id"]) in ("done", "failed", "skipped") and goal.assign.get(o) in (P.step_goal(step)[1], None):
            T["in_plan_fix"] = ["re_queue"]
        else:
            T["route"] = llm
    return T


def _blocker(reason: str, ts):
    for oid, ob in ts.objects.items():
        if ob.name in reason.split("occupied by")[-1] or ob.name in reason.split("has")[-1]:
            return oid
    return None


def _pending_for(view, oid) -> bool:
    return any(s["object"] == oid and view.status.get(s["id"]) == "pending" for s in (view.plan or {}).get("steps", []))


def _crowds_pending(view, ts, f) -> bool:
    fo = ts.objects[f]
    for sid in view.queue:
        s = P.steps_by_id(view.plan)[sid]
        if view.status.get(sid) in ("pending", "running") and s["object"] in ts.objects and s["object"] != f:
            ob = ts.objects[s["object"]]
            if ob.where == "table" and dxy((ob.x, ob.y), (fo.x, fo.y)) < FINGER_CM + 1.0 and s["skill"] != "push":
                return True
    return False


def _running_wrong(view, goal: Goal, ts) -> bool:
    """Would the step in progress (or the held object's destination) be wrong under the new goal?"""
    cur = view.current
    if cur is None:
        return ts.arm.holding is not None
    g = P.step_goal(cur)
    if g is None:
        return False
    return goal.assign.get(g[0]) != g[1]


def _plan_reaches(plan: dict, queue: list, ts, goal: Goal) -> bool:
    if plan is None:
        return False
    byid = P.steps_by_id(plan)
    where = {o.id: o.where for o in ts.objects.values()}
    if ts.arm.holding:
        where[ts.arm.holding] = "gripper"
    for sid in queue:
        g = P.step_goal(byid[sid])
        if g:
            where[g[0]] = g[1]
    for o, d in goal.assign.items():
        if where.get(o) != d:
            return False
    for c in goal.constraints:
        if c["kind"] == "keep_out_of":
            pl = ts.places.get(c["place"])
            mem = pl.members if pl is not None and pl.kind == "group" else (c["place"],)
            if where.get(c["object"]) in mem:
                return False
    return True
