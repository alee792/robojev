"""K2's and K4's bar, and the seed of the conformance suite: N trials of one standard skill, judged
from the world state, recorded. In sim the positions are random and the judge reads physics; on the
real arm (D1) the positions are typed in and the judge reads the server's own bookkeeping.

    uv run --frozen robojev-conformance --skill pick_and_place --n 20 --seed 0
    uv run --frozen robojev-conformance --skill stack_on
    uv run --frozen robojev-conformance --skill push
    uv run --frozen robojev-conformance --robot real --robot-ip 192.168.1.5 --blocks blocks.json \\
        --teleop-off-ack "your name" --skill pick_and_place --n 20
    uv run --frozen robojev-conformance --robot real --driver fake --blocks blocks.json --skill pick_and_place --n 5

Sim: each trial is a fresh scene. pick_and_place: one 4 cm block at a random pose in x 0.26-0.38,
y -0.15..0.15, random yaw, a random tray slot as the destination. stack_on: two such blocks at least
10 cm apart, the first stacked on the second. push: one block, a random catalog direction and a
distance of 3-12 cm whose path stays in the workspace. hand_over is not here: the sim has no hands,
so it is exercised by injecting one (tests/test_robots_widowx_sim.py).

Real: one `RealRobotServer` for the whole run (staging the arm once), the blocks where `--blocks`
says (robots/widowx_real/world.load_blocks), re-laid by a person between trials at the prompt; the
destination slot, push direction and distance are drawn as in sim. `--driver fake` runs the same
loop here against `FakeTrossenDriver`, with no prompt.

Success is judged by `judge_trial()` from the world state alone, never from what the skill says
about itself: the object rests where the skill was to leave it (in the slot's radius; "on:" the
base block; at least 80 % of the distance along the direction and still upright), nothing is in the
gripper, and the arm has retreated to travel height. In sim the world state is MuJoCo ground truth.
On the real arm it is `StaticWorld`'s bookkeeping (a held block follows the EE, a released one
settles where it was let go) plus the gripper snapshot: the arm is trusted to have done what it was
told, and the run's header says so. A block that fell out unnoticed, or a slot missed by more than
the gripper knows, needs a person's eyes or P1's camera.

With --out, every policy tick of every trial goes to a JSONL (JsonlRecorder) with the trial index
on each line, plus a note per trial with the setup and the verdict.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robojev.config import Workspace
from robojev.protocol import Manifest, RobotEvent, WorldState
from robojev.recording.recorder import JsonlRecorder
from robojev.robots.widowx_sim import scene, skills
from robojev.robots.widowx_sim.server import ARM_ID, LIMITS, SimRobotServer

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results" / "k2_trials.jsonl"
SKILLS = ("pick_and_place", "stack_on", "push")
X_RANGE = (0.26, 0.38)
Y_RANGE = (-0.15, 0.15)
PUSH_RANGE = (0.03, 0.12)   # m
MIN_APART = 0.10            # m between two blocks' centres at the start of a stack trial
SKILL_TIMEOUT_S = 20.0      # sim seconds before a trial is abandoned (the skill's own timeout is 15 s)
REAL_TIMEOUT_S = 30.0       # wall seconds on the real arm (its Params.timeout_s is 20 s active)
SETTLE_S = 0.3              # physics after the skill reports done, so a block still falling is judged resting or not
REAL_JUDGE_NOTE = ("judge: the real arm has no ground truth; verdicts come from the server's bookkeeping (where "
                   "the arm was told to put the block) and the gripper snapshot, not from observation")


@dataclass
class Trial:
    index: int
    skill: str
    blocks: list[scene.Block]
    args: dict = field(default_factory=dict)

    @property
    def block_xy(self) -> tuple[float, float]:
        return self.blocks[0].xy

    @property
    def block_yaw(self) -> float:
        return self.blocks[0].yaw

    @property
    def place(self) -> str:
        return self.args["place"]


@dataclass
class Verdict:
    ok: bool
    why: str                   # literal: what the world looked like
    state: str                 # the skill's own verdict, for comparison
    reason: str | None
    sim_s: float               # seconds on the server's clock: sim time in sim, wall time on the real arm
    wall_s: float


def _block(rng: random.Random, name: str) -> scene.Block:
    return scene.Block(name, (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)), yaw=rng.uniform(-math.pi, math.pi))


def _push_args(rng: random.Random, b: scene.Block, workspace: Workspace, table_z: float) -> dict:
    """A random catalog direction and distance whose whole path stays inside `workspace`."""
    p = skills.Params()
    while True:
        direction = rng.choice(sorted(skills.DIRECTIONS))
        distance = round(rng.uniform(*PUSH_RANGE), 3)
        pose = skills.ObjPose(b.xy[0], b.xy[1], 0.0, b.yaw, b.size)
        start, limit = skills.push_path(pose, direction, distance, p)
        z = table_z + p.tip_height
        if workspace.contains(start + (z,)) and workspace.contains(limit + (z,)):
            return {"object": b.name, "direction": direction, "distance": distance}


def sample(rng: random.Random, index: int, n_slots: int, skill: str = "pick_and_place") -> Trial:
    """A random sim trial of `skill`, drawn so its preconditions hold (blocks apart, a push path in reach)."""
    b1 = _block(rng, "block_1")
    if skill == "pick_and_place":
        return Trial(index, skill, [b1], {"object": "block_1", "place": f"slot_{rng.randrange(n_slots)}"})
    if skill == "stack_on":
        b2 = _block(rng, "block_2")
        while math.dist(b1.xy, b2.xy) < MIN_APART:
            b2 = _block(rng, "block_2")
        return Trial(index, skill, [b1, b2], {"object": "block_1", "onto": "block_2"})
    if skill == "push":
        return Trial(index, skill, [b1], _push_args(rng, b1, LIMITS.workspace, scene.TABLE_Z))
    raise ValueError(f"no trials for {skill}")


def sample_laid_out(rng: random.Random, index: int, blocks: list[scene.Block], n_slots: int, skill: str,
                    workspace: Workspace, table_z: float) -> Trial:
    """A trial of `skill` on blocks where a person laid them (the real arm): positions as given, the
    first block the object and the second the stack base, the slot or the push drawn as `sample` does."""
    if not blocks:
        raise ValueError("no blocks laid out")
    b1 = blocks[0]
    if skill == "pick_and_place":
        if n_slots == 0:
            raise ValueError("pick_and_place needs a tray")
        return Trial(index, skill, list(blocks), {"object": b1.name, "place": f"slot_{rng.randrange(n_slots)}"})
    if skill == "stack_on":
        if len(blocks) < 2:
            raise ValueError("stack_on needs two blocks laid out")
        return Trial(index, skill, list(blocks), {"object": b1.name, "onto": blocks[1].name})
    if skill == "push":
        return Trial(index, skill, list(blocks), _push_args(rng, b1, workspace, table_z))
    raise ValueError(f"no trials for {skill}")


def _arm_checks(world: WorldState, manifest: Manifest, arm: str) -> list[str]:
    a = world.arms[arm]
    travel_z = next(s.travel_z for s in manifest.arms if s.id == arm)
    problems = []
    if a.holding:
        problems.append(f"the gripper still holds {a.holding}")
    if a.z < travel_z - 0.01:
        problems.append(f"the arm is at z {a.z:.3f}, not back at travel height {travel_z:.2f}")
    return problems


def judge(world: WorldState, manifest: Manifest, oid: str, place: str, arm: str = ARM_ID) -> tuple[bool, str]:
    """Did the world end up as a successful pick and place leaves it? From the world state only."""
    o = world.objects.get(oid)
    if o is None:
        return False, f"{oid} is not in the world"
    problems = []
    if o.where != place:
        p = world.places[place]
        problems.append(f"{oid} is {o.where}, {math.hypot(o.x - p.x, o.y - p.y) * 100:.1f} cm from {place}")
    problems += _arm_checks(world, manifest, arm)
    if problems:
        return False, "; ".join(problems)
    return True, f"{oid} resting in {place}, gripper empty, arm at travel height"


def judge_stack(world: WorldState, manifest: Manifest, oid: str, onto: str, arm: str = ARM_ID) -> tuple[bool, str]:
    """As `judge`, for a stack: the object reads "on:<onto>" (resting on its top, centred within 1.5 cm)."""
    o, b = world.objects.get(oid), world.objects.get(onto)
    if o is None or b is None:
        return False, f"{oid if o is None else onto} is not in the world"
    problems = []
    if o.where != f"on:{onto}":
        problems.append(f"{oid} is {o.where}, {math.hypot(o.x - b.x, o.y - b.y) * 100:.1f} cm off {onto}'s centre, "
                        f"{(o.z - b.z) * 100:.1f} cm above it")
    problems += _arm_checks(world, manifest, arm)
    if problems:
        return False, "; ".join(problems)
    return True, f"{oid} resting on {onto}, gripper empty, arm at travel height"


def judge_push(world: WorldState, manifest: Manifest, oid: str, start: tuple[float, float], direction: str,
               distance: float, arm: str = ARM_ID, table_z: float = scene.TABLE_Z) -> tuple[bool, str]:
    """As `judge`, for a push: the object moved at least 80 % of `distance` along `direction` from
    `start` and rests upright on the table (or in a slot) rather than tipped or still in flight."""
    o = world.objects.get(oid)
    if o is None:
        return False, f"{oid} is not in the world"
    dx, dy = skills.DIRECTIONS[direction]
    along = (o.x - start[0]) * dx + (o.y - start[1]) * dy
    across = (o.x - start[0]) * dy - (o.y - start[1]) * dx
    problems = []
    if along < 0.8 * distance:
        problems.append(f"{oid} moved {along * 100:.1f} cm {direction}, under 80 % of {distance * 100:.1f}")
    if o.where.startswith(("on:", "gripper:")):
        problems.append(f"{oid} is {o.where}")
    size = o.size or 0.04
    if abs(o.z - (table_z + size / 2)) > 0.005:
        problems.append(f"{oid} is not flat on the table: centre {(o.z - table_z) * 100:.1f} cm up, tipped or in flight")
    problems += _arm_checks(world, manifest, arm)
    if problems:
        return False, "; ".join(problems)
    return True, f"{oid} moved {along * 100:.1f} cm {direction} ({abs(across) * 100:.1f} cm sideways), upright, gripper empty, arm at travel height"


def judge_trial(world: WorldState, manifest: Manifest, trial: Trial, table_z: float = scene.TABLE_Z) -> tuple[bool, str]:
    a = trial.args
    if trial.skill == "pick_and_place":
        return judge(world, manifest, a["object"], a["place"])
    if trial.skill == "stack_on":
        return judge_stack(world, manifest, a["object"], a["onto"])
    if trial.skill == "push":
        return judge_push(world, manifest, a["object"], trial.block_xy, a["direction"], a["distance"], table_z=table_z)
    raise ValueError(f"no judge for {trial.skill}")


def make_server(trial: Trial, recorder: JsonlRecorder | None) -> SimRobotServer:
    return SimRobotServer(scene.SceneSpec(blocks=list(trial.blocks)), realtime=False, recorder=recorder)


def run_trial(trial: Trial, recorder: JsonlRecorder | None = None, srv=None, world=None,
              table_z: float = scene.TABLE_Z, timeout_s: float = SKILL_TIMEOUT_S) -> Verdict:
    """One trial. Without `srv`, a fresh sim server for the trial's scene; with one (the real
    server, staged once for the run), `world` (a StaticWorld of the trial's blocks) is swapped in."""
    t0 = time.perf_counter()
    if srv is None:
        srv = make_server(trial, recorder)
    else:
        srv.set_world(world)
    t_start = srv.t
    events: list[RobotEvent] = []
    srv.subscribe(events.append)
    try:
        why = srv.precondition(ARM_ID, trial.skill, trial.args)
        if why:
            return Verdict(False, f"refused: {why}", "refused", why, 0.0, time.perf_counter() - t0)
        sid = srv.start(ARM_ID, trial.skill, trial.args)
        srv.run_until(srv.finished(sid), timeout_s)
        srv.run_for(SETTLE_S)
        st = srv.status(sid)
        ok, text = judge_trial(srv.world(), srv.manifest(), trial, table_z)
    finally:
        srv.unsubscribe(events.append)
    v = Verdict(ok, text, st.state, st.reason, srv.t - t_start, time.perf_counter() - t0)
    if recorder is not None:
        recorder.note(kind="verdict", trial=asdict(trial), verdict=asdict(v),
                      events=[e.kind for e in events])
    return v


def describe(trial: Trial) -> str:
    b = trial.blocks[0]
    s = f"block ({b.xy[0]:.3f}, {b.xy[1]:+.3f}) yaw {math.degrees(b.yaw):+6.1f} deg"
    if trial.skill == "pick_and_place":
        return f"{s} -> {trial.place}"
    if trial.skill == "stack_on":
        c = trial.blocks[1]
        return f"{s} -> onto ({c.xy[0]:.3f}, {c.xy[1]:+.3f})"
    return f"{s} push {trial.args['distance'] * 100:4.1f} cm {trial.args['direction']}"


def layout(trial: Trial) -> str:
    """What a person lays out for a real trial."""
    return "; ".join(f"{b.name} at ({b.xy[0]:.3f}, {b.xy[1]:+.3f}) yaw {math.degrees(b.yaw):+.0f} deg" for b in trial.blocks)


def _report(skill: str, n: int, seed: int, verdicts: list[Verdict], rec: JsonlRecorder | None) -> int:
    wins = sum(v.ok for v in verdicts)
    print(f"\n{skill}: {wins}/{n} succeeded (seed {seed}); mean {sum(v.sim_s for v in verdicts) / max(1, n):.1f} s per trial")
    for v in verdicts:
        if not v.ok:
            print(f"  failed: skill {v.state} {v.reason or ''} | {v.why}")
    if rec is not None:
        print("recorded:", rec.close())
    return 0 if wins >= math.ceil(0.95 * n) else 1


def _line(i: int, trial: Trial, v: Verdict, clock: str) -> str:
    mark = "ok  " if v.ok else "FAIL"
    return (f"{mark} {i:2d}  {describe(trial)}  {v.sim_s:5.1f} s {clock} {v.wall_s:4.1f} s wall  skill {v.state}"
            f"{'' if v.reason is None else ' (' + v.reason + ')'}  | {v.why}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skill", choices=SKILLS, default="pick_and_place")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL of every policy tick and event; default: none, so a check run can't overwrite "
                         f"the committed seed-0 log at {RESULTS}")
    ap.add_argument("--robot", choices=("sim", "real"), default="sim", help="MuJoCo (default) or the real follower")
    real = ap.add_argument_group("real", "the real arm (robots/widowx_real); see docs/day-one.md")
    real.add_argument("--robot-ip", default=None, help="the follower's address (default 192.168.1.5)")
    real.add_argument("--blocks", type=Path, default=None,
                      help="JSON of the typed-in block positions and tray (robots/widowx_real/world.load_blocks)")
    real.add_argument("--driver", choices=("trossen", "fake"), default="trossen",
                      help="the trossen_arm driver, or the fake to run the real server here with no arm")
    real.add_argument("--teleop-off-ack", metavar="NAME", default=None,
                      help="your name: you have checked that no teleop process is running on the leader")
    real.add_argument("--no-prompt", action="store_true", help="do not wait for Enter between trials")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rec = None if a.out is None else JsonlRecorder(a.out)
    if a.robot == "sim":
        n_slots = scene.Tray().n
        verdicts = []
        for i in range(a.n):
            trial = sample(rng, i, n_slots, a.skill)
            if rec is not None:
                rec.tag = {"trial": i}
            v = run_trial(trial, rec)
            verdicts.append(v)
            print(_line(i, trial, v, "sim"))
        return _report(a.skill, a.n, a.seed, verdicts, rec)
    return _main_real(a, ap, rng, rec)


def _main_real(a, ap: argparse.ArgumentParser, rng: random.Random, rec: JsonlRecorder | None) -> int:
    from robojev.arm.real import FOLLOWER_IP
    from robojev.robots.widowx_real.arm import day_one_config
    from robojev.robots.widowx_real.fake_driver import FakeDriverFactory
    from robojev.robots.widowx_real.server import RealRobotServer
    from robojev.robots.widowx_real.world import StaticWorld, load_blocks

    if a.blocks is None:
        ap.error("--robot real needs --blocks FILE with the typed-in block positions")
    factory = None
    if a.driver == "fake":
        factory = FakeDriverFactory(lambda: srv.world_src)       # follows set_world between trials
        ack = a.teleop_off_ack or "fake driver: no leader exists"
    else:
        ack = a.teleop_off_ack
        if not ack:
            ap.error("--robot real --driver trossen needs --teleop-off-ack NAME: check that no teleop process is running "
                     "on the leader (docs/day-one.md), then pass your name")
    blocks, tray, table_z = load_blocks(a.blocks)
    cfg = day_one_config(table_z)
    n_slots = tray.n if tray else 0
    try:
        trials = [sample_laid_out(rng, i, blocks, n_slots, a.skill, cfg.workspace, table_z) for i in range(a.n)]
    except ValueError as e:
        ap.error(str(e))
    srv = RealRobotServer(a.robot_ip or FOLLOWER_IP, StaticWorld(blocks, tray, table_z), cfg, recorder=rec, driver_factory=factory)
    srv.preflight(ack, teleop_off=True, notes=f"robojev-conformance --skill {a.skill} --n {a.n} --driver {a.driver}")
    print(REAL_JUDGE_NOTE)
    if rec is not None:
        rec.note(kind="judge", text=REAL_JUDGE_NOTE)
    verdicts = []
    with srv:
        for trial in trials:
            if a.driver != "fake" and not a.no_prompt:
                input(f"trial {trial.index}: lay out {layout(trial)}, clear the tray, then press Enter (Ctrl-C aborts and parks) ")
            if rec is not None:
                rec.tag = {"trial": trial.index}
            v = run_trial(trial, rec, srv=srv, world=StaticWorld(trial.blocks, tray, table_z), table_z=table_z,
                          timeout_s=REAL_TIMEOUT_S)
            verdicts.append(v)
            print(_line(trial.index, trial, v, "arm"))
    return _report(a.skill, a.n, a.seed, verdicts, rec)


if __name__ == "__main__":
    sys.exit(main())
