"""The decision request: one Jev request per event, three question groups (docs/v2.md "The decision").

State = the event + where the arm is in the plan (a literal fact) + the objects near it, filtered to
what the event is about. Code does the arithmetic (distances become named bands, "in the arm's path",
"approaching"), looks up what it can (which step an object belongs to, whether the object is the
current step's), and offers only options it has checked apply. Questions:

  right_now      Spotter   Choice: carry_on / hold / pause / back_off / re_target / resume
  in_plan_fix    Sequencer Choice: none / retry / skip / re_queue / another_step_first
  route          Router    Choice: stay_local / fast_llm / capable_llm / ask_user
  done.<id>      progress  Noul per done condition code cannot measure ("is it true that ...?")
  check.<object> plan check Noul per object the plan never mentions (yes = something is wrong)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .changes import hand_band
from .plan import NONE, mentioned, step_goal, step_text, steps_by_id
from .data import WorldState, dxy, seg_dist

# ---------------------------------------------------------------- what the harness shows a decider


@dataclass
class View:
    t: int
    task: str
    user_messages: list
    state: WorldState
    plan: dict | None
    queue: list
    status: dict                    # step id -> pending | running | done | failed | skipped
    current: dict | None            # the running step, or the one the event is about
    phase: str                      # literal: where the arm is in the current step
    mode: str                       # waiting | running | paused | holding | stopped | done
    pause_reason: str | None
    replan_pending: bool
    names: dict
    path_to: tuple | None           # where the arm is heading in this step (x, y)
    hand: dict | None               # core.changes.hand_facts
    runnable_other: str | None      # a pending step (not the current one) whose precondition holds
    tries: dict = field(default_factory=dict)   # step id -> failures so far
    grasped: bool = False           # the current step's object is in the gripper

    def name(self, oid) -> str:
        return self.names.get(oid, oid)


# ---------------------------------------------------------------- criteria (Jev reads literally)

RIGHT_NOW_CRITERIA = {
    "carry_on": {
        "what": "No change: the arm keeps doing what it is doing now (moving on with the current step, or staying paused or holding if it already is).",
        "not_for": "a person's hand coming toward the gripper or into the arm's path while the arm moves; the user asking the robot to wait; a change that makes the move in progress wrong",
    },
    "hold": {
        "what": "Stop at a safe point, keeping anything in the gripper, because the move in progress may be wrong now: the task or the plan is about to change.",
        "not_for": "a person's hand nearby (use pause or back_off); the user asking the robot to wait (use pause); an object moved that the plan still handles as written",
    },
    "pause": {
        "what": "Stop where it is and stay stopped until told to go on: a person's hand is coming toward the gripper or is in the arm's path, or the user asks the robot to wait.",
        "not_for": "a hand that is already very close to the gripper (use back_off); a hand held out, still, to take an object the robot is handing over",
    },
    "back_off": {
        "what": "Move the gripper away from a person's hand that is very close to it, then stay stopped.",
        "not_for": "a hand that is not very close to the gripper; a hand held out, still, to take an object the robot is handing over",
    },
    "re_target": {
        "what": "The object the current step is going for has been moved and is not in the gripper yet: aim at where it is now and carry on.",
        "not_for": "an object other than the current step's; an object already in the gripper",
    },
    "resume": {
        "what": "The arm is paused or holding and the reason has gone (the hand has left or is no longer coming toward the gripper or in its path, or the user said to go on): start moving again.",
        "not_for": "a hand still coming toward the gripper or in its path; the user asked to wait and has not said to go on; the arm is not paused or holding",
    },
}

FIX_CRITERIA = {
    "none": {
        "what": "No fix within the plan: nothing needs fixing (a step finished normally, a remark, a pause, a hand), or no fix below fits.",
        "not_for": "a failed step that trying again would fix; an object the plan already moved that was put back where it started",
    },
    "retry": {
        "what": "Run the step that failed again: the failure looks temporary (e.g. the object was moved and can be tried where it is now, or a hand was in the way).",
        "not_for": "a step that cannot succeed as written (its place is occupied by something the plan does not move, or it would break a constraint)",
    },
    "skip": {
        "what": "Drop the current step: it is no longer needed (e.g. its object is already where the step would put it).",
        "not_for": "a step that is still needed",
    },
    "re_queue": {
        "what": "`event_object` has a step that already ran, but the object is no longer where that step put it: run that step again.",
        "not_for": "an object whose step has not run yet; an object with no step in the plan",
    },
    "another_step_first": {
        "what": "The current step cannot run yet (e.g. its place is occupied by an object that a later step moves) but another planned step can: run that one first.",
        "not_for": "a step that failed for a temporary reason (use retry); an obstacle that no step in the plan moves",
    },
}

ROUTE_CRITERIA = {
    "stay_local": {
        "what": "The plan still holds as written, with at most the fix above: steps finishing, objects moved that their own steps will still handle, a placed object to be put back, hands, waits, remarks, temporary failures.",
        "not_for": "the user asking for a different end result; an object the plan now has to deal with but has no step for; a plan that would break a constraint or leaves out an object the task asks about",
    },
    "fast_llm": {
        "what": "The plan must change: the user asks for a different end result, or the world changed so the plan needs new or different steps. Most replans.",
        "not_for": "anything the plan already covers; requests that could mean two different things",
    },
    "capable_llm": {
        "what": "The plan must change and working out how is hard: several constraints conflict, the same step keeps failing, or the request needs careful multi-step reasoning.",
        "not_for": "an ordinary correction or a single new step (use fast_llm)",
    },
    "ask_user": {
        "what": "The user's words could reasonably mean two different end results, and guessing wrong would undo work.",
        "not_for": "a clear request, however big the change; remarks that ask for nothing",
    },
}


# ---------------------------------------------------------------- facts


def _where_text(v: View, where: str) -> str:
    if where == "gripper":
        return "in the gripper"
    if where == "table":
        return "on the table"
    if where == "person":
        return "held by the person"
    if where.startswith("on:"):
        return f"on top of {v.name(where[3:])}"
    return f"in {v.name(where)}"


def _mode_text(v: View) -> str:
    if v.mode == "paused":
        return f"paused ({v.pause_reason or 'no reason recorded'}); it stays paused until resumed"
    if v.mode == "holding":
        return "holding still at a safe point" + (" while a new plan is worked out" if v.replan_pending else "")
    if v.mode == "waiting":
        return "waiting for the first plan"
    return {"running": "running the plan", "done": "finished", "stopped": "stopped"}.get(v.mode, v.mode)


def object_step(v: View, oid: str) -> tuple[str, dict | None]:
    """(status, step) of the latest step in the plan that moves `oid`."""
    if not v.plan:
        return "no step", None
    best = None
    for s in v.plan["steps"]:
        if s["object"] == oid and step_goal(s):
            st = v.status.get(s["id"], "dropped")
            if st == "dropped" and s["id"] not in v.queue:
                continue
            best = (st, s)
    return best or ("no step", None)


def object_fact(v: View, oid: str) -> dict:
    o = v.state.obj(oid)
    a = v.state.arm
    st, s = object_step(v, oid)
    f = {"name": v.name(oid), "where": _where_text(v, o.where) if o else "not in view"}
    if o and o.where != "gripper":
        f["distance_from_gripper"] = hand_band(dxy((a.x, a.y), (o.x, o.y)))
        f["in_the_arms_path"] = "yes" if v.path_to and seg_dist((o.x, o.y), (a.x, a.y), v.path_to) < 6 and o.where != "gripper" else "no"
    if s is None:
        f["its_step"] = "the plan has no step for it"
    else:
        tgt = step_goal(s)[1]
        at_goal = o is not None and (o.where == tgt)
        f["its_step"] = f"{step_text(s, v.names)}: {st}" + ("" if st != "done" else (", and it is still there" if at_goal else ", but it is no longer there"))
    f["is_the_current_steps_object"] = "yes" if v.current and v.current.get("object") == oid else "no"
    return f


def nearby(v: View, focus: set, k: int = 5) -> list[dict]:
    """Objects within 20 cm of the gripper or of where the step is heading, plus the event's objects."""
    a = v.state.arm
    pts = [(a.x, a.y)] + ([v.path_to] if v.path_to else [])
    near = []
    for o in v.state.objects.values():
        if o.id in focus or o.where == "gripper":
            continue
        d = min(dxy(p, (o.x, o.y)) for p in pts)
        if d < 20:
            near.append((d, o.id))
    return [object_fact(v, oid) for _, oid in sorted(near)[:k]]


def plan_position(v: View) -> dict:
    p = {"current_step": step_text(v.current, v.names) if v.current else "none",
         "arm_in_step": v.phase}
    if v.plan:
        byid = steps_by_id(v.plan)
        nxt = [step_text(byid[i], v.names) for i in v.queue if v.status.get(i) == "pending" and (not v.current or i != v.current["id"])]
        p["steps_done"] = sum(1 for s in v.status.values() if s == "done")
        p["next_steps"] = nxt[:2]
        p["steps_left_after_current"] = len(nxt)
    return p


def build_state(event, v: View) -> dict:
    s = {"event": {"kind": event.kind.replace("_", " "), "what_happened": event.text}}
    if event.kind == "user_text":
        s["event"]["user_just_said"] = event.data.get("text", "")
    s["task"] = v.task
    if len(v.user_messages) > 1 and event.kind != "user_text":
        s["user_said_earlier"] = v.user_messages[-3:]
    elif event.kind == "user_text" and len(v.user_messages) > 1:
        s["user_said_earlier"] = v.user_messages[-4:-1]
    s["robot"] = {"mode": _mode_text(v), "holding": v.name(v.state.arm.holding) if v.state.arm.holding else "nothing",
                  "new_plan_being_worked_out": "yes" if v.replan_pending else "no"}
    s["plan_position"] = plan_position(v)
    focus = set()
    if event.object and (event.object in v.state.objects):
        s["event_object"] = object_fact(v, event.object)
        focus.add(event.object)
    if event.kind == "step_failed":
        s["failure"] = {"reason": event.data.get("reason", ""), "failures_of_this_step_so_far": v.tries.get(event.step_id, 0),
                        "another_planned_step_can_run_now": "yes: " + v.runnable_other if v.runnable_other else "no"}
    if event.kind == "plan_done_unmet":
        s["unmet_done_conditions"] = event.data.get("unmet", [])
    if event.kind == "plan_arrived" and v.plan:
        byid = steps_by_id(v.plan)
        s["plan"] = {"reading": v.plan.get("reading", ""),
                     "steps_to_run": [step_text(byid[i], v.names) for i in v.queue],
                     "done_when": [c["text"] for c in v.plan["done_when"]],
                     "objects_the_plan_never_mentions": [v.name(o) for o in unmentioned(v)]}
    if v.hand:
        h = v.hand
        s["hand"] = {"distance_from_gripper": h["band"], "coming_closer": "yes" if h["approaching"] else "no",
                     "in_the_arms_path": "yes" if h["in_path"] else "no",
                     "held_out_still_as_if_to_take_something": "yes" if h["held_out"] else "no"}
    else:
        s["hand"] = "no person's hand in view"
    s["nearby_objects"] = nearby(v, focus)
    if v.plan and v.plan["constraints"]:
        s["constraints"] = [c["text"] for c in v.plan["constraints"]]
    return s


def unmentioned(v: View) -> list[str]:
    if not v.plan:
        return []
    m = mentioned(v.plan)
    return sorted(o for o in v.state.objects if o not in m)


# ---------------------------------------------------------------- options code has checked


def offered_right_now(v: View) -> list[str]:
    opts = ["carry_on", "hold", "pause"]
    if v.hand:
        opts.append("back_off")
    if v.current and v.current.get("object", NONE) != NONE and not v.grasped:
        opts.append("re_target")
    if v.mode in ("paused", "holding"):
        opts.append("resume")
    return opts


def offered_fixes(event, v: View) -> list[str]:
    opts = ["none"]
    if event.kind == "step_failed":
        opts.append("retry")
    if event.kind == "step_failed" or (v.current is not None and v.status.get(v.current["id"]) == "running" and not v.grasped):
        opts.append("skip")
    oid = event.object
    if oid:
        st, _ = object_step(v, oid)
        if st in ("done", "failed", "skipped"):
            opts.append("re_queue")
    if event.kind == "step_failed" and v.runnable_other:
        opts.append("another_step_first")
    return opts


# ---------------------------------------------------------------- questions


def build_questions(event, v: View, groups=("right_now", "in_plan_fix", "route")) -> dict:
    qs = {}
    if "right_now" in groups:
        qs["right_now"] = {
            "type": "choice",
            "instructions": {
                "question": "Given `event`, what should the arm do right now?",
                "context": "`plan_position.arm_in_step` says literally where the arm is in its current step; `robot.mode` says whether it is moving, paused or holding. The code's own stop distance for hands applies regardless of this answer.",
            },
            "criteria": {k: RIGHT_NOW_CRITERIA[k] for k in offered_right_now(v)},
        }
    fixes = offered_fixes(event, v)
    if "in_plan_fix" in groups and len(fixes) > 1:     # only "none" applies: code knows the answer, don't ask
        qs["in_plan_fix"] = {
            "type": "choice",
            "instructions": {"question": "Can `event` be absorbed within the current plan with one of these fixes, without writing new steps or changing the task?"},
            "criteria": {k: FIX_CRITERIA[k] for k in fixes},
        }
    if "route" in groups:
        qs["route"] = {
            "type": "choice",
            "instructions": {"question": "Who should handle `event`: the robot on its own with the current plan, or a planner that writes a new plan, or the user?",
                             "ask_about": "whether the end result or the steps must change, not what the event is about"},
            "criteria": ROUTE_CRITERIA,
        }
    if v.plan and event.kind in ("step_done", "plan_done_unmet"):
        for c in v.plan["done_when"]:
            if c["relation"] == "other":
                qs[f"done.{c['id']}"] = {
                    "type": "noul",
                    "instructions": f"Is it true, right now, that {c['text']}?",
                    "criteria": {"true": f"{c['text']}: yes, as the objects are now.", "false": "No, or not yet."},
                }
    if event.kind == "plan_arrived":
        for oid in unmentioned(v):
            nm = v.name(oid)
            qs[f"check.{oid}"] = {
                "type": "noul",
                "instructions": f"Is {nm} something `task` asks about (something to move, arrange, hand over or keep away from somewhere) but `plan` leaves out?",
                "criteria": {"true": f"The task covers {nm} (by name, number, colour or as one of a group it describes) and the plan has no step, done condition or constraint for it.",
                             "false": f"The task does not ask for anything about {nm}; the plan is right to leave it alone."},
            }
    return qs
