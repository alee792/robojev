"""K2's bar, and the seed of the conformance suite: N `pick_and_place` trials at random reachable
positions, judged from physics, recorded.

    uv run --frozen python experiments/skills_sim/trials.py --n 20 --seed 0

Each trial is a fresh scene: one 4 cm block at a random pose in x 0.26-0.38, y -0.15..0.15, random
yaw, and a random tray slot as the destination. Success is judged by `judge()` from the world state
alone, never from what the skill says about itself: the block rests inside the slot's radius, nothing
is in the gripper, and the arm has retreated to travel height. D1 runs the same loop against the real
backend; only `make_server` changes.

With --out, every policy tick of every trial goes to a JSONL (JsonlRecorder) with the
trial index on each line, plus a note per trial with the setup and the verdict.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the skills_sim package

from skills_sim import scene  # noqa: E402
from skills_sim.protocol import Manifest, RobotEvent, WorldState  # noqa: E402
from skills_sim.recorder import JsonlRecorder  # noqa: E402
from skills_sim.sim_server import ARM_ID, SimRobotServer  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "k2_trials.jsonl"
X_RANGE = (0.26, 0.38)
Y_RANGE = (-0.15, 0.15)
SKILL_TIMEOUT_S = 20.0      # sim seconds before a trial is abandoned (the skill's own timeout is 15 s)
SETTLE_S = 0.3              # physics after the skill reports done, so a block still falling is judged resting or not


@dataclass
class Trial:
    index: int
    block_xy: tuple[float, float]
    block_yaw: float
    place: str


@dataclass
class Verdict:
    ok: bool
    why: str                   # literal: what the world looked like
    state: str                 # the skill's own verdict, for comparison
    reason: str | None
    sim_s: float
    wall_s: float


def sample(rng: random.Random, index: int, n_slots: int) -> Trial:
    return Trial(index, (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)), rng.uniform(-math.pi, math.pi),
                 f"slot_{rng.randrange(n_slots)}")


def judge(world: WorldState, manifest: Manifest, oid: str, place: str, arm: str = ARM_ID) -> tuple[bool, str]:
    """Did the world end up as a successful pick and place leaves it? From the world state only."""
    o, a = world.objects.get(oid), world.arms[arm]
    travel_z = next(s.travel_z for s in manifest.arms if s.id == arm)
    if o is None:
        return False, f"{oid} is not in the world"
    problems = []
    if o.where != place:
        p = world.places[place]
        problems.append(f"{oid} is {o.where}, {math.hypot(o.x - p.x, o.y - p.y) * 100:.1f} cm from {place}")
    if a.holding:
        problems.append(f"the gripper still holds {a.holding}")
    if a.z < travel_z - 0.01:
        problems.append(f"the arm is at z {a.z:.3f}, not back at travel height {travel_z:.2f}")
    if problems:
        return False, "; ".join(problems)
    return True, f"{oid} resting in {place}, gripper empty, arm at travel height"


def make_server(trial: Trial, recorder: JsonlRecorder | None) -> SimRobotServer:
    spec = scene.SceneSpec(blocks=[scene.Block("block_1", trial.block_xy, yaw=trial.block_yaw)])
    return SimRobotServer(spec, realtime=False, recorder=recorder)


def run_trial(trial: Trial, recorder: JsonlRecorder | None = None) -> Verdict:
    t0 = time.perf_counter()
    srv = make_server(trial, recorder)
    events: list[RobotEvent] = []
    srv.subscribe(events.append)
    args = {"object": "block_1", "place": trial.place}
    why = srv.precondition(ARM_ID, "pick_and_place", args)
    if why:
        return Verdict(False, f"refused: {why}", "refused", why, 0.0, time.perf_counter() - t0)
    sid = srv.start(ARM_ID, "pick_and_place", args)
    srv.run_until(srv.finished(sid), SKILL_TIMEOUT_S)
    srv.run_for(SETTLE_S)
    st = srv.status(sid)
    ok, text = judge(srv.world(), srv.manifest(), "block_1", trial.place)
    v = Verdict(ok, text, st.state, st.reason, srv.t, time.perf_counter() - t0)
    if recorder is not None:
        recorder.note(kind="verdict", trial=asdict(trial), verdict=asdict(v),
                      events=[e.kind for e in events])
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL of every policy tick and event; default: none, so a check run can't overwrite "
                         f"the committed seed-0 log at {RESULTS}")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rec = None if a.out is None else JsonlRecorder(a.out)
    n_slots = scene.Tray().n
    verdicts = []
    for i in range(a.n):
        trial = sample(rng, i, n_slots)
        if rec is not None:
            rec.tag = {"trial": i}
        v = run_trial(trial, rec)
        verdicts.append(v)
        mark = "ok  " if v.ok else "FAIL"
        print(f"{mark} {i:2d}  block ({trial.block_xy[0]:.3f}, {trial.block_xy[1]:+.3f}) yaw {math.degrees(trial.block_yaw):+6.1f} deg "
              f"-> {trial.place}  {v.sim_s:5.1f} s sim {v.wall_s:4.1f} s wall  skill {v.state}"
              f"{'' if v.reason is None else ' (' + v.reason + ')'}  | {v.why}")
    wins = sum(v.ok for v in verdicts)
    print(f"\n{wins}/{a.n} succeeded (seed {a.seed}); mean {sum(v.sim_s for v in verdicts) / max(1, a.n):.1f} s sim per trial")
    for v in verdicts:
        if not v.ok:
            print(f"  failed: skill {v.state} {v.reason or ''} | {v.why}")
    if rec is not None:
        print("recorded:", rec.close())
    return 0 if wins >= math.ceil(0.95 * a.n) else 1


if __name__ == "__main__":
    sys.exit(main())
