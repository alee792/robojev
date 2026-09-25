"""Metrics: an Observer on the harness (read-only hooks), per-episode records, tables, the offline gate
sweep and the pass criteria.

Per episode: completed (the oracle's goal for the final task version holds in the true world, nothing
held), sim time, LLM calls and total LLM wait, time holding and paused, disturbances and recoveries,
hand contacts and constraint violations, correction-to-changed-behaviour times, spurious reactions to
chatter, Jev requests / latency / tokens, per-group agreement with the oracle, and every Jev answer
with its truth (for the gate sweep).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from common import pct

from ..core import plan as P
from ..core.data import dxy
from ..sim.world import FINGER_CM, HAND_CONTACT_CM, LOW_CM, TICKS_PER_S
from . import oracle as O

JEV_PRICE = 0.042e-6
GATES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GROUPS = ("right_now", "in_plan_fix", "route")


class Observer:
    def __init__(self, scenario, world):
        self.sc, self.w = scenario, world
        self.contacts = 0
        self._contact = False
        self.violations = 0
        self.violation_notes: list = []
        self.agree: dict = defaultdict(lambda: [0, 0])
        self.answers: list = []      # per Jev decision: {group: (choice, conf, right)}, for the gate sweep
        self.utt: list = []          # {text, kind, t, changed_t}
        self._forb_xy: dict = {}
        self._prev_where: dict = {}
        self.spurious = 0
        self.corr_events: dict = {}

    # ---------------------------------------------------------------- hooks
    def on_event(self, ep, ev):
        if ev.kind == "user_text":
            text = ev.data.get("text", "")
            u = self.sc.utts.get(text)
            self.utt.append({"text": text, "kind": u.kind if u else "answer", "t": ev.t, "changed_t": None,
                             "mode_before": ep.mode, "ev": ev.id, "goal": self.sc.goal_for(ep.user_messages)})

    def on_decision(self, ep, ev, out, view):
        T = O.truth(ev, view, self.sc, self.w)
        out["truth"] = {k: v for k, v in T.items()}
        a = out.get("answers")
        req = out.get("request")
        c = out["combined"]
        if ev.kind == "user_text":
            u = self.sc.utts.get(ev.data.get("text", ""))
            if u is not None and u.kind == "chatter" and (c.route != "stay_local" or c.react or c.right_now in ("hold", "pause", "back_off")):
                self.spurious += 1
        if not a or not req:
            return
        rec = {}
        for g in GROUPS:
            if g in a and g in req["questions"]:
                offered = list(req["questions"][g]["criteria"])
                truth_offered = [t for t in T.get(g, []) if t in offered]
                if not truth_offered:
                    continue
                right = a[g].get("choice") in T[g]
                self.agree[g][0] += right
                self.agree[g][1] += 1
                rec[g] = (a[g].get("choice"), float(a[g].get("confidence") or 0.0), right)
        for k, q in req["questions"].items():
            if q["type"] == "noul" and k in T:
                right = (a[k]["noul"] >= 0.5) == bool(T[k])
                grp = k.split(".")[0]
                self.agree[grp][0] += right
                self.agree[grp][1] += 1
        if rec:
            rec["event"] = ev.kind
            self.answers.append(rec)

    def on_llm(self, ep, res, ev):
        pass

    def on_tick(self, ep):
        ts = self.w.truth()
        a = ts.arm
        # hand contact: the gripper inside a reaching hand's contact zone (a held-out hand taking a block is consent)
        h = self.w.hand
        touching = h is not None and h.purpose == "reach" and math.dist((a.x, a.y, a.z), (h.x, h.y, h.z)) < HAND_CONTACT_CM
        if touching and not self._contact:
            self.contacts += 1
        self._contact = touching
        # constraint violations caused by the robot
        goal = self.sc.goal_now()
        person_moved = {m for b in self.sc.person.fired_now for m in b.moved}
        for c in goal.constraints:
            o = ts.objects.get(c["object"])
            if o is None:
                continue
            if c["kind"] == "dont_touch":
                prev = self._forb_xy.get(o.id)
                if a.holding == o.id or (prev and dxy(prev, (o.x, o.y)) > 0.5 and o.id not in person_moved):
                    self._violate(ep, f"the robot moved {o.name}")
                if a.z < LOW_CM and dxy((a.x, a.y), (o.x, o.y)) < FINGER_CM * 0.6:
                    self._violate(ep, f"the gripper was down on {o.name}")
                self._forb_xy[o.id] = (o.x, o.y)
            elif c["kind"] == "keep_out_of":
                was = self._prev_where.get(o.id)
                if was == "gripper" and ts.in_place(o.id, c["place"]):
                    self._violate(ep, f"the robot put {o.name} into {c['place']}")
        self._prev_where = {oid: ob.where for oid, ob in ts.objects.items()}
        # correction-to-changed-behaviour
        for u in self.utt:
            if u["changed_t"] is not None or ep.t <= u["t"]:
                continue
            if self._changed(ep, u, ts):
                u["changed_t"] = ep.t

    def _violate(self, ep, why):
        self.violations += 1
        self.violation_notes.append((ep.t, why))

    def _changed(self, ep, u, ts) -> bool:
        k = u["kind"]
        if k == "wait":
            return ep.mode in ("paused", "stopped")
        if k == "go_on":
            return ep.mode == "running" and ep.skill is not None
        if k in ("correction", "answer"):
            goal = self.sc.goal_for(ep.user_messages)
            if ep.mode in ("holding", "paused", "stopped"):
                return True
            if ep.mode == "done":
                return True
            if ep.cur is not None:
                g = P.step_goal(ep.cur)
                return g is None or goal.assign.get(g[0], "table") == g[1] or (g[1] == "table" and g[0] in goal.assign)
            return False
        return False


# ---------------------------------------------------------------- per-episode record


def episode_record(arm: str, sc, seed: int, res, obs: Observer, world) -> dict:
    ts = world.truth()
    goal = sc.goal_now()
    completed = O.achieved(goal, ts) and res.outcome == "done"
    tps = TICKS_PER_S
    llm_att = [a for x in res.llm for a in x["attempts"]]
    jev = [d for d in res.decisions if d["jev"]]
    disturb = [b for b in sc.person.beats if b.disturbs and b.fired_t is not None]
    recovered = sum(1 for b in disturb if completed and all(
        (o not in goal.assign) or P.measure({"relation": "on" if goal.assign[o].startswith("on:") else "in", "object": o,
                                             "target": goal.assign[o][3:] if goal.assign[o].startswith("on:") else goal.assign[o]}, ts)
        for o in b.moved))
    corr = [(u["changed_t"] - u["t"]) / tps for u in obs.utt if u["kind"] == "correction" and u["changed_t"] is not None]
    corr_missed = sum(1 for u in obs.utt if u["kind"] == "correction" and u["changed_t"] is None)
    waits = [(u["changed_t"] - u["t"]) / tps for u in obs.utt if u["kind"] in ("wait", "go_on") and u["changed_t"] is not None]
    return {
        "arm": arm, "scenario": sc.name, "held_out": sc.held_out, "interference": sc.interference, "seed": seed,
        "completed": completed, "outcome": res.outcome, "sim_s": res.ticks / tps,
        "llm_calls": len(llm_att), "llm_requests": len(res.llm), "llm_wait_s": sum(a["latency_ms"] for a in llm_att) / 1000,
        "llm_in": sum(a["in_tok"] for a in llm_att), "llm_out": sum(a["out_tok"] for a in llm_att),
        "llm_cached": sum(a.get("cached_tok", 0) for a in llm_att),
        "hold_s": res.mode_ticks.get("holding", 0) / tps, "pause_s": res.mode_ticks.get("paused", 0) / tps,
        "disturbances": len(disturb), "recovered": recovered,
        "contacts": obs.contacts, "violations": obs.violations,
        "corr_s": corr, "corr_missed": corr_missed, "wait_s": waits, "spurious": obs.spurious,
        "jev_requests": len(jev), "jev_ms": [d["latency_ms"] for d in jev], "jev_tok": sum(d["in_tok"] for d in jev),
        "agree": {k: list(v) for k, v in obs.agree.items()}, "answers": obs.answers,
        "loops": res.loops, "steps_failed": res.steps_failed, "refusals": res.refusals, "floor_stops": res.floor_stops,
        "stale_plans": res.stale_plans, "failed_plans": res.failed_plans,
    }


# ---------------------------------------------------------------- tables


def _med(xs):
    return statistics.median(xs) if xs else float("nan")


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    corr = [x for r in recs for x in r["corr_s"]]
    ms = [x for r in recs for x in r["jev_ms"]]
    agree = Counter()
    tot = Counter()
    for r in recs:
        for g, (ok, k) in r["agree"].items():
            agree[g] += ok
            tot[g] += k
    return {
        "n": n, "completed": sum(r["completed"] for r in recs), "sim_s": _med([r["sim_s"] for r in recs]),
        "llm_calls": sum(r["llm_calls"] for r in recs), "llm_wait_s": sum(r["llm_wait_s"] for r in recs),
        "hold_s": sum(r["hold_s"] for r in recs), "pause_s": sum(r["pause_s"] for r in recs),
        "disturbances": sum(r["disturbances"] for r in recs), "recovered": sum(r["recovered"] for r in recs),
        "contacts": sum(r["contacts"] for r in recs), "violations": sum(r["violations"] for r in recs),
        "corr_med": _med(corr), "corr_n": len(corr), "corr_missed": sum(r["corr_missed"] for r in recs),
        "wait_med": _med([x for r in recs for x in r["wait_s"]]), "spurious": sum(r["spurious"] for r in recs),
        "jev_requests": sum(r["jev_requests"] for r in recs), "jev_p50": pct(ms, 50) if ms else float("nan"),
        "jev_p95": pct(ms, 95) if ms else float("nan"), "jev_tok": sum(r["jev_tok"] for r in recs),
        "llm_in": sum(r["llm_in"] for r in recs), "llm_out": sum(r["llm_out"] for r in recs),
        "agree": {g: (agree[g], tot[g]) for g in tot}, "loops": sum(r["loops"] for r in recs),
    }


def _f(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table(recs: list[dict], title: str, arms: list[str], by_scenario: bool = True) -> str:
    out = [f"\n## {title}\n"]
    hdr = "| arm | done | sim s (med) | LLM calls | LLM wait s | hold s | pause s | recovered/dist | contacts | violations | corr→behaviour s (med) | corr missed | wait/go s (med) | chatter spurious | Jev req | Jev p50/p95 ms | agree right-now / fix / route |"
    out += [hdr, "|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|"]
    for arm in arms:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            continue
        s = summarize(rs)
        ag = s["agree"]
        agtxt = " / ".join(f"{ag[g][0]}/{ag[g][1]}" if g in ag else "-" for g in GROUPS)
        out.append(f"| {arm} | {s['completed']}/{s['n']} | {_f(s['sim_s'])} | {s['llm_calls']} | {_f(s['llm_wait_s'])} | {_f(s['hold_s'])} | "
                   f"{_f(s['pause_s'])} | {s['recovered']}/{s['disturbances']} | {s['contacts']} | {s['violations']} | {_f(s['corr_med'], 2)} "
                   f"(n={s['corr_n']}) | {s['corr_missed']} | {_f(s['wait_med'], 2)} | {s['spurious']} | {s['jev_requests']} | "
                   f"{_f(s['jev_p50'], 0)}/{_f(s['jev_p95'], 0)} | {agtxt} |")
    if by_scenario:
        scen = sorted({r["scenario"] for r in recs}, key=lambda n: [r["scenario"] for r in recs].index(n))
        out.append("\nCompleted per scenario (runs completed / runs):\n")
        out.append("| scenario | " + " | ".join(arms) + " |")
        out.append("|" + "|".join(["---"] * (len(arms) + 1)) + "|")
        for sc in scen:
            cells = []
            for arm in arms:
                rs = [r for r in recs if r["arm"] == arm and r["scenario"] == sc]
                if not rs:
                    cells.append("")
                    continue
                c = sum(r["completed"] for r in rs)
                extra = []
                ct = sum(r["contacts"] for r in rs)
                if ct:
                    extra.append(f"{ct} contact")
                cs = [x for r in rs for x in r["corr_s"]]
                if cs:
                    extra.append(f"corr {_med(cs):.1f}s")
                llm = sum(r["llm_calls"] for r in rs)
                extra.append(f"{llm} LLM")
                cells.append(f"{c}/{len(rs)} ({', '.join(extra)})")
            out.append(f"| {sc} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- gate sweep (offline, from answers)


def gate_sweep(answer_recs: list[dict], gates=GATES) -> list[dict]:
    """For each gate: the wrong right-now / in-plan-fix answers caught (confidence below the gate, so
    the cautious fallback or an escalation applies instead), the right answers escalated needlessly,
    and the accuracy of what is acted on. Fixes count only when the route answered stay_local (the
    only time a fix applies); their confidence is the weaker of route and fix."""
    rows = []
    for g in gates:
        wrong = caught = acted = acted_right = right_esc = n = 0
        rn_fallback = 0
        for rec in answer_recs:
            if "right_now" in rec:
                ch, c, ok = rec["right_now"]
                n += 1
                if c < g:
                    rn_fallback += 1
                    caught += not ok
                    right_esc += ok
                else:
                    acted += 1
                    acted_right += ok
                wrong += not ok
            if "in_plan_fix" in rec and rec.get("route", (None,))[0] == "stay_local":
                ch, c, ok = rec["in_plan_fix"]
                rch, rc, rok = rec["route"]
                w = min(c, rc)
                both = ok and rok
                n += 1
                if w < g:
                    caught += not both
                    right_esc += both
                else:
                    acted += 1
                    acted_right += both
                wrong += not both
        rows.append({"gate": g, "answers": n, "wrong": wrong, "caught": caught, "caught_pct": 100 * caught / wrong if wrong else float("nan"),
                     "escalated_right": right_esc, "escalated_pct": 100 * (caught + right_esc) / n if n else float("nan"),
                     "acted_accuracy": 100 * acted_right / acted if acted else float("nan")})
    return rows


def sweep_table(rows) -> str:
    out = ["\n## Gate sweep (right-now + in-plan fix answers; offline, from the answers logged)\n",
           "| gate | answers | wrong | wrong caught | caught % | escalated/fallback % | accuracy of acted-on answers % |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['gate']:.1f} | {r['answers']} | {r['wrong']} | {r['caught']} | {_f(r['caught_pct'], 0)} | "
                   f"{_f(r['escalated_pct'], 1)} | {_f(r['acted_accuracy'], 1)} |")
    return "\n".join(out)


# ---------------------------------------------------------------- pass criteria


def pass_criteria(recs: list[dict], gate: float, answers: list[dict]) -> str:
    core = [r for r in recs if not r["held_out"]]
    held = [r for r in recs if r["held_out"]]

    def S(arm, rs):
        x = [r for r in rs if r["arm"] == arm]
        return summarize(x) if x else None

    lines = ["\n## Pass criteria (decide whether we move to MuJoCo)\n"]
    j = S("jev", core)
    if j is None:
        return "\n".join(lines + ["(no `jev` arm in this run)"])
    c1 = j["completed"] >= 0.9 * j["n"] and j["contacts"] == 0
    lines.append(f"1. `jev` completes >= 90% of core episodes with zero hand contacts: {j['completed']}/{j['n']} completed, "
                 f"{j['contacts']} contacts -> **{'PASS' if c1 else 'FAIL'}**")
    # The rules control is written knowing the core scenarios' phrasing, so criterion 2 is decided on the
    # held-out interference/correction scenarios (changes the rules could not anticipate); core is shown for
    # information.
    hi = [r for r in held if r["interference"]]
    hj, hr = S("jev", hi), S("rules", hi)
    if hj and hr:
        better_c = hj["completed"] > hr["completed"]
        better_t = (not math.isnan(hj["corr_med"])) and (math.isnan(hr["corr_med"]) or hj["corr_med"] < hr["corr_med"])
        c2 = better_c or better_t
        lines.append(f"2. `jev` beats `rules` on completion or correction time on the held-out interference/correction scenarios: "
                     f"completion {hj['completed']}/{hj['n']} vs {hr['completed']}/{hr['n']}; median correction→behaviour "
                     f"{_f(hj['corr_med'], 2)} s vs {_f(hr['corr_med'], 2)} s -> **{'PASS' if c2 else 'FAIL'}**")
        inter = [r for r in core if r["interference"]]
        ji, ri = S("jev", inter), S("rules", inter)
        if ji and ri:
            lines.append(f"   (information, not part of the verdict: on the core scenarios, whose phrasing the rules were written "
                         f"against, completion {ji['completed']}/{ji['n']} vs {ri['completed']}/{ri['n']}; median correction→behaviour "
                         f"{_f(ji['corr_med'], 2)} s vs {_f(ri['corr_med'], 2)} s)")
    elif hr is None and S("rules", core):
        lines.append("2. (no held-out scenarios with the `rules` arm in this run)")
    else:
        lines.append("2. (no `rules` arm in this run)")
    a = S("always_llm", core)
    if a:
        faster = (not math.isnan(j["corr_med"])) and (math.isnan(a["corr_med"]) or j["corr_med"] < a["corr_med"])
        c3 = faster and j["completed"] >= a["completed"]
        lines.append(f"3. `jev` faster than `always_llm` on median correction→behaviour without lower completion: {_f(j['corr_med'], 2)} s vs "
                     f"{_f(a['corr_med'], 2)} s; completion {j['completed']}/{j['n']} vs {a['completed']}/{a['n']} -> **{'PASS' if c3 else 'FAIL'}**")
    else:
        lines.append("3. (no `always_llm` arm in this run)")
    rows = gate_sweep(answers, (gate,))
    r = rows[0]
    c4 = r["wrong"] == 0 or r["caught_pct"] >= 90
    lines.append(f"4. At the chosen gate ({gate}), >= 90% of Jev's wrong in-plan fixes / right-now answers are caught: "
                 f"{r['caught']}/{r['wrong']} ({_f(r['caught_pct'], 0)}%) -> **{'PASS' if c4 else 'FAIL'}**"
                 + (" (no wrong answers in this run)" if r["wrong"] == 0 else ""))
    hj = S("jev", held)
    if hj:
        lines.append(f"5. Held-out completion (`jev`, no threshold): {hj['completed']}/{hj['n']}; "
                     + ", ".join(f"{arm} {S(arm, held)['completed']}/{S(arm, held)['n']}" for arm in sorted({r['arm'] for r in held}) if arm != "jev"))
    return "\n".join(lines)
