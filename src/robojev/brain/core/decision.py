"""The decision request: one Jev request per event, three question groups (docs/v2.md "The decision").

State = the event + where the arm is in the plan (a literal fact) + the objects near it, filtered to
what the event is about. Code does the arithmetic (distances become named bands, "in the arm's path",
"approaching"), looks up what it can (which step an object belongs to, whether the object is the
current step's), and offers only options it has checked apply. Questions:

  right_now      Spotter   Choice: carry_on / hold / pause / back_off / re_target / resume
  in_plan_fix    Sequencer Choice: none / retry / skip / re_queue / another_step_first / swap_target:<object>
  route          Router    Choice: stay_local / fast_llm / capable_llm / ask_user
  done.<id>      progress  Noul per done condition code cannot measure ("is it true that ...?")
  check.<object> plan check Noul per object the plan never mentions (yes = something is wrong)

The task is shown twice. `user_words` is what the user asked for, in their own words: the task they
typed and every correction the brain took as about the end result, newest last, never a transient
"wait" or "go on". `task` is how the current plan reads it (the planner's restatement, corrections
folded in, which stops the old replan loop). A correction is judged against `user_words`: the
restatement can paraphrase ("ascending numeric order") in a way that hides the words a correction
contradicts ("lowest on the left" vs "highest on the left"), and the live run showed Jev confidently
carrying on through such a correction (docs/live-results.md, section 1).

On `plan_arrived` only `right_now` and the plan-check Nouls are asked: a new plan is presumed good
and goes back to the planner only on a specific plan-check yes, never on the generic route question
(docs/v2.md "The decision"; core/combine.py has the numbers).

`swap_target:<object>` is the one in-plan fix built per event. Mid-reach the user says "actually, the
scalpel": the right answer is to go for the other object now, same destination, nothing else in the
plan touched, and it has to happen in the fast loop (about 150 ms), not after a planner call. Code
offers it only when it is obvious: a typed line while a grasping step is running whose object is
not yet in the gripper, for each other object (at most three) the line names (its name or label as
a word; a one-letter label only as the capital written on it, "a" being an article) or, when the
line says "other" or "another", each object sharing the current one's label (two A cubes). The
decider picks one or none; the loop re-validates the rewritten plan in code and sends anything that
does not fit to the planner instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .changes import hand_band
from .plan import GRASPING, NONE, mentioned, step_goal, step_text, steps_by_id
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
    user_words: list = field(default_factory=list)   # the task and each correction, verbatim, newest last (no transients)
    unplanned_words: list = field(default_factory=list)   # the tail of user_words no plan has read yet (a correction being planned)

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

SWAP = "swap_target"
_OTHER_RE = re.compile(r"\b(other|another)\b", re.I)
MAX_SWAP_CANDIDATES = 3


def swap_fix(oid: str) -> str:
    return f"{SWAP}:{oid}"


def swap_object(fix: str) -> str | None:
    """The object id a swap_target fix names, or None for any other fix."""
    return fix[len(SWAP) + 1:] if fix.startswith(SWAP + ":") else None


def _word_in(text: str, phrase: str | None, exact_case: bool = False) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, 0 if exact_case else re.I) is not None


_LETTER_I = re.compile(r"\b(letter|the|other|an)\s+I(?!\w)", re.I)


def names_in(text: str, o) -> bool:
    """Whether a typed line names object `o`: its name as a phrase, or its label as a word. A
    one-character label is matched as written (the capital on a letter cube), so the article "a"
    does not name the letter A; the letter I is named only as "the I", "letter I", "the other I",
    since a bare capital I is the pronoun."""
    if _word_in(text, o.name):
        return True
    lab = o.label or ""
    if len(lab) != 1:
        return _word_in(text, lab)
    if lab == "I":
        return _LETTER_I.search(text) is not None
    return _word_in(text, lab, exact_case=True)


def swap_candidates(text: str, v: View) -> list[str]:
    """Objects the running grasping step could go for instead (module doc): named in `text`, or, when
    it says "other"/"another", sharing the current object's label. Never the current object, nothing
    held or taken by the person; the named ones first, then by id; at most MAX_SWAP_CANDIDATES."""
    cur = v.current
    if not text or cur is None or cur["skill"] not in GRASPING or v.grasped or v.status.get(cur["id"]) != "running":
        return []
    cur_o = v.state.obj(cur["object"])
    other = _OTHER_RE.search(text) is not None and cur_o is not None and bool(cur_o.label)
    named, same = [], []
    for oid, o in sorted(v.state.objects.items()):
        if oid == cur["object"] or o.where in ("gripper", "person") or o.where.startswith("gripper:"):
            continue
        if names_in(text, o):
            named.append(oid)
        elif other and o.label == cur_o.label:
            same.append(oid)
    return (named + same)[:MAX_SWAP_CANDIDATES]


def fix_criteria(fix: str, v: View) -> dict:
    """The criteria text for one offered fix: the table above, or, for a swap, the text written for
    this event's objects (Jev reads literally: it must name them)."""
    oid = swap_object(fix)
    if oid is None:
        return FIX_CRITERIA[fix]
    cur = v.name(v.current["object"]) if v.current else "the current object"
    new = v.name(oid)
    return {
        "what": f"The user wants {new} handled instead of {cur}, for the same destination; the rest of the plan stands. "
                f"The arm has not grasped {cur} yet, so it can go for {new} now.",
        "not_for": f"a line that only mentions {new} (a remark, a question, a word about something else); the user wanting a "
                   f"different destination or a different end result (that is a planner's job); {cur} already in the gripper",
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
    what = _shape_text(o)
    if what:
        f["shape"] = what
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


def _shape_text(o) -> str | None:
    """"a rod, its front pointing left": what the object is and which way it faces, when known."""
    if o is None:
        return None
    shape, front = getattr(o, "shape", None), getattr(o, "front", None)
    bits = [f"a {shape}"] if shape else []
    if front:
        bits.append(f"its front pointing {front.replace('_', ' ')}")
    return ", ".join(bits) or None


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
    # What the user asked for, in their words (the task, then each correction the brain took as about
    # the end result, newest last; never a transient "wait" or "go on"), and the task as the current
    # plan reads it: the planner restates it with every plan, corrections folded in (docs/v2.md "The
    # decision"). A correction is judged against the user's words, not the restatement (module doc).
    s["user_words"] = list(v.user_words) or [v.task]
    s["task"] = (v.plan.get("reading") if v.plan else "") or v.task
    # Which of those words `task` already answers. The live correction runs (docs/live-results.md,
    # "More planner calls after a correction") showed a stateless decider reading the old task beside
    # the correction as "the user asks for a different end result" on every later step, though the
    # plan had folded the correction in; code knows which lines the current plan has read.
    # Shown only on events about the plan's progress, never on a typed line or a pause re-ask: there
    # the question is the line itself (or whether to go on), and the live run (docs/live-results.md,
    # "Handoff 8") showed the Decisions API reading "every line is folded into task" on "ok, go on" as
    # a reason not to resume (0.88 -> 0.28 with this field and the right-now wording that pointed at it).
    if v.plan and v.user_words and event.kind not in ("user_text", "paused_idle"):
        s["user_words_in_task"] = ("all of them: `task` is the plan's reading of every line in `user_words`, the corrections folded in"
                                   if not v.unplanned_words else
                                   {"not_yet": list(v.unplanned_words), "note": "a new plan for these is being worked out; `task` does not include them yet"})
    if v.replan_pending and v.user_messages and event.kind != "user_text":
        s["change_being_planned"] = {"user_said": v.user_messages[-1],
                                     "note": "a new plan for this is being worked out; `task` does not include it yet"}
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
                     "objects_the_plan_never_mentions": [_unmentioned_text(v, g) for g in unmentioned_groups(v)]}
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


def unmentioned_groups(v: View) -> list[dict]:
    """The objects the plan never mentions, grouped by label (objects.label: a letter or number read
    off them, so two "A" cubes are interchangeable). One plan-check question per group, not per
    object, and each group knows how many objects of its label the plan already uses: a spare letter
    is the normal case on a table with duplicates, and the live spelling runs (docs/live-results.md,
    "Good new plans sent back") showed Jev unsure whether "the second letter A" was something the
    task asks about, with nothing in the question to tell a spare from a missing letter.
    -> [{"ids": [...], "label": str | None, "used": [ids the plan uses with the same label]}]"""
    if not v.plan:
        return []
    m = mentioned(v.plan)
    groups: dict[str, dict] = {}
    for oid in sorted(v.state.objects):
        if oid in m:
            continue
        lab = v.state.objects[oid].label
        key = f"label:{lab}" if lab else f"id:{oid}"
        g = groups.setdefault(key, {"ids": [], "label": lab or None, "used": []})
        g["ids"].append(oid)
    for g in groups.values():
        if g["label"]:
            g["used"] = sorted(o for o in m if o in v.state.objects and v.state.objects[o].label == g["label"])
    return list(groups.values())


def _unmentioned_text(v: View, g: dict) -> str:
    names = ", ".join(v.name(o) for o in g["ids"])
    if not g["label"]:
        return names
    k = len(g["used"])
    return f"{names} (labelled {g['label']}; the plan uses {k} other object{'s' if k != 1 else ''} labelled {g['label']})"


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
    if event.kind == "user_text":
        opts += [swap_fix(o) for o in swap_candidates(event.data.get("text", ""), v)]
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
                "about_the_task": "`user_words` is what the user asked for, in their own words (the task, then each correction, newest last); `task` is how the current plan reads it. If the user just asked for something different from `user_words`, the move in progress may be wrong even where it fits `task`.",
            },
            "criteria": {k: RIGHT_NOW_CRITERIA[k] for k in offered_right_now(v)},
        }
    fixes = offered_fixes(event, v) if event.kind != "plan_arrived" else ["none"]   # a new plan needs no fix within itself
    if "in_plan_fix" in groups and len(fixes) > 1:     # only "none" applies: code knows the answer, don't ask
        qs["in_plan_fix"] = {
            "type": "choice",
            "instructions": {"question": "Can `event` be absorbed within the current plan with one of these fixes, without writing new steps or changing the task?"},
            "criteria": {k: fix_criteria(k, v) for k in fixes},
        }
    if "route" in groups and event.kind != "plan_arrived":     # a new plan: the plan check alone decides (module doc)
        qs["route"] = {
            "type": "choice",
            "instructions": {"question": "Who should handle `event`: the robot on its own with the current plan, or a planner that writes a new plan, or the user?",
                             "ask_about": "whether the end result or the steps must change, not what the event is about. Whether the user asks for a different end result is judged against `user_words` (what the user asked for, in their words), not against `task` (how the current plan reads it)."
                                          + (" A correction that `task` already states (`user_words_in_task`) is not a request for a different end result: the plan has changed for it already, and the steps go on as written unless this event itself calls for a change"
                                             if event.kind not in ("user_text", "paused_idle") else "")},

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
        for g in unmentioned_groups(v):
            qs[f"check.{g['ids'][0]}"] = _check_question(v, g)
    return qs


def _check_question(v: View, g: dict) -> dict:
    """One plan-check Noul for a group of unmentioned objects (unmentioned_groups). With a label the
    plan already uses, the question is about COUNT: does the task need more objects of that label
    than the plan moves? A spare is then a confident "false", not a general doubt."""
    ids, lab, used = g["ids"], g["label"], g["used"]
    nm = v.name(ids[0]) + (f" (and {len(ids) - 1} more labelled {lab})" if len(ids) > 1 else "")
    if lab and used:
        k, n = len(used), len(used) + len(ids)
        plural = "s" if k != 1 else ""
        q = (f"Of the {n} objects labelled {lab} in view, `plan` already uses {k} ({', '.join(v.name(o) for o in used)}). {nm} is another one "
             f"labelled {lab} that `plan` leaves out. Does `task` need more objects labelled {lab} than the {k} the plan uses?")

        return {"type": "noul", "instructions": q,
                "criteria": {"true": f"Counting what the task names, spells or describes, it needs more than {k} object{plural} labelled {lab}, so {nm} is needed and the plan leaves it out.",
                             "false": f"The task needs {k} or fewer objects labelled {lab}: {nm} is a spare, and the plan is right to leave it alone."}}
    what = f"{nm}" + (f" (labelled {lab})" if lab else "")
    return {"type": "noul",
            "instructions": f"Is {what} something `task` asks about (something to move, arrange, hand over or keep away from somewhere) but `plan` leaves out?",
            "criteria": {"true": f"The task covers {what} (by name, number, letter, colour or as one of a group it describes) and the plan has no step, done condition or constraint for it.",
                         "false": f"The task does not ask for anything about {what}; the plan is right to leave it alone."}}

