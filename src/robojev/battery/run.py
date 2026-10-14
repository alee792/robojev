"""The runner: generated episodes through the brain on the stub or the physics sim, judged, written
out one JSONL per episode, summarised in a table per decider.

    run_episode(ep, opts)  one episode: world, perception layer, director, user, brain; -> Record
    run(opts)              N episodes for one decider; -> Summary (and the JSONL files under opts.out)
    compare(opts)          mock, rules and always_llm on the same episodes, side by side

Each episode's JSONL holds the episode as generated, everything the director did and when, the
brain's log, decisions and planner records, and the judge's record. Nothing in it grades an answer.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from robojev.brain import Brain, BrainConfig

from . import scenes
from .generate import Episode, Grammar, Paraphrase, episodes
from .judge import Record, judge
from .models import DECIDERS, PLANNERS, make_decider, make_planner
from .stage import BatteryUser, Director, Perception

WORLDS = ("stub", "sim")
CONTROLS = ("mock", "rules", "always_llm")
MAX_S = {"stub": 60.0, "sim": 240.0}
PARK_WAIT_S = 12.0           # after STOP, how long to wait for the arms to park before judging


@dataclass(frozen=True)
class Options:
    n: int = 20
    seed: int = 0
    world: str = "stub"
    decider: str = "mock"
    planner: str = "mock"
    arms: int = 1
    split: str = "train"
    out: Path | None = None
    paraphrase: Paraphrase | None = None
    grammar: Grammar | None = None
    pick_s: float | None = None          # the world's pick time, for the episode clock; default scenes.PICK_S
    max_s: float | None = None
    max_llm_requests: int = 40
    hold_s: float = 1.0
    quiet: bool = False

    def __post_init__(self) -> None:
        if self.world not in WORLDS:
            raise ValueError(f"world must be one of {WORLDS}")
        if self.decider not in DECIDERS or self.planner not in PLANNERS:
            raise ValueError(f"decider must be one of {DECIDERS} and planner one of {PLANNERS}")
        if self.arms == 2 and self.world != "stub":
            raise ValueError("two arms run on the stub (docs/battery.md)")


@dataclass
class Summary:
    """One decider over a set of episodes: the table's numbers."""
    decider: str
    planner: str
    world: str
    n: int
    mean_score: float
    finish_rate: float                   # exact and done, over the episodes without a STOP
    judged: int                          # episodes without a STOP
    time_to_change_s: dict[str, float | None]    # median per event kind (the first action of each event)
    planner_called: dict[str, float]             # per event kind, the share of events a planner request followed
    planner_calls: float                 # per episode
    replans: float
    decisions: float
    decision_ms_p50: float | None
    drops: float
    hands: int
    contacts: int
    stops: int
    stops_ok: int
    wall_s: float                        # median
    outcomes: dict[str, int]
    records: list[Record] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("records")
        return d


def _jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _server(ep: Episode, opts: Options):
    if opts.world == "stub":
        return scenes.stub_server(ep, opts.pick_s or scenes.PICK_S["stub"])
    return scenes.sim_server(ep)


def _wait_parked(perception: Perception, deadline_s: float) -> float | None:
    """After STOP: wait until every arm reads "stopped" and has been still for half a second. ->
    seconds it took, or None."""
    t0 = time.monotonic()
    still_since: float | None = None
    last = None
    while time.monotonic() - t0 < deadline_s:
        w = perception.truth()
        pos = tuple((a.id, round(a.x, 3), round(a.y, 3), round(a.z, 3)) for a in w.arms.values())
        stopped = all(a.mode == "stopped" for a in w.arms.values())
        if stopped and pos == last:
            if still_since is None:
                still_since = time.monotonic()
            elif time.monotonic() - still_since >= 0.5:
                return round(still_since - t0, 2)
        else:
            still_since = None
        last = pos
        time.sleep(0.05)
    return None


def run_episode(ep: Episode, opts: Options, decider_name: str | None = None) -> tuple[Record, dict]:
    """One episode. -> (the judge's record, everything for its JSONL)."""
    decider_name = decider_name or opts.decider
    lay = scenes.layout(ep)
    pick_s = opts.pick_s or scenes.PICK_S[opts.world]
    srv = _server(ep, opts)
    srv.open()
    perception = Perception(srv, hidden=lay.parked)
    perception.open()
    user = BatteryUser(ep.reply)
    director = Director(ep, lay, perception, user, pick_s)
    arms = [a.id for a in srv.manifest().arms]
    try:
        decider = make_decider(decider_name, ep, perception.world, arms)
        planner = make_planner(opts.planner, perception, seed=ep.index)
        cfg = BrainConfig(max_s=opts.max_s or MAX_S[opts.world], max_llm_requests=opts.max_llm_requests, hold_s=opts.hold_s)
        brain = Brain(perception, decider, planner, user, cfg)
        t0 = time.monotonic()
        director.start(t0)
        r = asyncio.run(brain.run(ep.opening))
        wall = time.monotonic() - t0
        director.stop()
        parked = None
        if r.outcome == "stopped":
            parked = _wait_parked(perception, PARK_WAIT_S)
        else:
            time.sleep(0.3)                           # a cube still settling after the last release
        world = perception.truth()
        rec = judge(ep, lay.positions, world, r, perception, director.fired, len(user.asked), t0, wall,
                    decider_name, opts.planner, opts.world, parked)
    finally:
        director.stop()
        user.close()
        perception.close()
        srv.close()
    dump = {"episode": _jsonable(ep), "fired": _jsonable(director.fired), "asked": _jsonable(user.asked),
            "brain_log": _jsonable(r.log), "decisions": _jsonable(r.decisions), "llm": _jsonable(r.llm),
            "judge": _jsonable(rec)}
    return rec, dump


def write_episode(path: Path, dump: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for key in ("episode", "judge"):
            f.write(json.dumps({"type": key, **dump[key]}) + "\n")
        for key in ("fired", "asked", "brain_log", "decisions", "llm"):
            for rec in dump[key]:
                f.write(json.dumps({"type": key, "rec": rec}) + "\n")


def _change_key(e) -> str | None:
    """Which row of the table an event's time-to-change goes in: the event kind for the thing itself
    (the line, the taking, the hand...), its own row for a go-on, a letter's return and a reply; None
    for a remark beside a world event (the scene change is the event) and for anything skipped."""
    if e.what == "skipped":
        return None
    if e.what == "go_on":
        return "go_on"
    if e.what == "returned":
        return "letter_returned"
    if e.kind == "reply":
        return "reply"
    if e.what == "text" and e.kind not in ("new_word", "other_letter", "chatter", "wait_go_on"):
        return None
    return e.kind


def summarise(records: Sequence[Record]) -> Summary:
    r0 = records[0]
    judged = [r for r in records if r.stop is None]
    ttc: dict[str, list[float]] = defaultdict(list)
    asked: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        for e in r.events:
            key = _change_key(e)
            if key is None:
                continue
            asked[key].append(bool(e.planner_called))
            if e.time_to_change_s is not None:
                ttc[key].append(e.time_to_change_s)
    lat = [r.decision_ms_p50 for r in records if r.decision_ms_p50 is not None]
    outcomes: dict[str, int] = defaultdict(int)
    for r in records:
        outcomes[r.outcome] += 1
    stops = [r for r in records if r.stop is not None]
    return Summary(r0.decider, r0.planner, r0.world, len(records),
                   round(statistics.mean(r.score for r in judged), 3) if judged else 0.0,
                   round(sum(r.exact and r.outcome == "done" for r in judged) / len(judged), 3) if judged else 0.0, len(judged),
                   {k: round(statistics.median(v), 3) for k, v in sorted(ttc.items())},
                   {k: round(sum(v) / len(v), 2) for k, v in sorted(asked.items())},
                   round(statistics.mean(r.planner_calls for r in records), 2), round(statistics.mean(r.replans for r in records), 2),
                   round(statistics.mean(r.decisions for r in records), 1), round(statistics.median(lat), 1) if lat else None,
                   round(statistics.mean(r.drops for r in records), 2), sum(r.hand_seen for r in records),
                   sum(r.hand_contact for r in records), len(stops),
                   sum(1 for r in stops if r.stop.parked_s is not None and r.stop.nothing_after and r.stop.held_or_set_down and r.stop.drops == 0),
                   round(statistics.median(r.wall_s for r in records), 1), dict(outcomes), list(records))


def run(opts: Options, eps: Sequence[Episode] | None = None, decider: str | None = None) -> Summary:
    decider = decider or opts.decider
    eps = list(eps) if eps is not None else episodes(opts.n, opts.seed, opts.split, opts.grammar, opts.paraphrase)
    records = []
    for ep in eps:
        rec, dump = run_episode(ep, opts, decider)
        records.append(rec)
        if opts.out is not None:
            write_episode(Path(opts.out) / f"ep_{ep.seed}_{ep.index:03d}_{decider}.jsonl", dump)
        if not opts.quiet:
            print(f"  {decider:10s} ep {ep.index:3d} {opts.split:8s} word {rec.word:6s} row {''.join(x or '_' for x in rec.row):8s} "
                  f"score {rec.score:.2f} {rec.outcome:8s} {rec.wall_s:6.1f} s  events {[e.kind for e in ep.events]}", flush=True)
    s = summarise(records)
    if opts.out is not None:
        Path(opts.out).mkdir(parents=True, exist_ok=True)
        (Path(opts.out) / f"summary_{decider}.json").write_text(json.dumps(_jsonable(s.as_dict()), indent=1))
    return s


def compare(opts: Options, deciders: Sequence[str] = CONTROLS, eps: Sequence[Episode] | None = None) -> list[Summary]:
    """The same episodes for each decider, in turn."""
    eps = list(eps) if eps is not None else episodes(opts.n, opts.seed, opts.split, opts.grammar, opts.paraphrase)
    return [run(opts, eps, d) for d in deciders]


def table(summaries: Sequence[Summary]) -> str:
    """The side-by-side table: one column per decider."""
    rows: list[tuple[str, list[str]]] = []

    def row(label: str, f) -> None:
        rows.append((label, [f(s) for s in summaries]))

    def opt(x, fmt: str) -> str:
        return "-" if x is None else format(x, fmt)
    row("episodes (judged / with STOP)", lambda s: f"{s.n} ({s.judged} / {s.stops})")
    row("mean outcome score", lambda s: f"{s.mean_score:.3f}")
    row("finish rate (exact row, done)", lambda s: f"{s.finish_rate:.2f}")
    row("outcomes", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(s.outcomes.items())))
    kinds = sorted({k for s in summaries for k in s.time_to_change_s})
    for k in kinds:
        row(f"median s to first change: {k}", lambda s, k=k: opt(s.time_to_change_s.get(k), ".2f"))
    if any("other_letter" in s.planner_called for s in summaries):
        row("planner asked after: other_letter", lambda s: opt(s.planner_called.get("other_letter"), ".2f"))
    row("planner calls / episode", lambda s: f"{s.planner_calls:.2f}")
    row("replans / episode", lambda s: f"{s.replans:.2f}")
    row("decisions / episode", lambda s: f"{s.decisions:.1f}")
    row("decision latency p50 ms", lambda s: opt(s.decision_ms_p50, ".0f"))
    row("drops / episode", lambda s: f"{s.drops:.2f}")
    row("hand contacts / episodes with a hand", lambda s: f"{s.contacts} / {s.hands}")
    row("STOP ok / with STOP", lambda s: f"{s.stops_ok} / {s.stops}")
    row("median wall s", lambda s: f"{s.wall_s:.1f}")
    heads = [f"{s.decider} + {s.planner}" for s in summaries]
    w0 = max(len(r[0]) for r in rows)
    ws = [max(len(h), *(len(r[1][i]) for r in rows)) for i, h in enumerate(heads)]
    line = f"{'':{w0}}  " + "  ".join(f"{h:>{w}}" for h, w in zip(heads, ws))
    out = [line, "-" * len(line)]
    for label, cells in rows:
        out.append(f"{label:{w0}}  " + "  ".join(f"{c:>{w}}" for c, w in zip(cells, ws)))
    return "\n".join(out)
