"""H6: turn the server's JSONL recording (recorder.JsonlRecorder) into a LeRobot-style dataset, and
replay it back frame for frame. The recording half of the data flywheel (docs/v2.md, "Data capture").

    uv run --frozen python -m robojev.recording.lerobot_export <recording.jsonl> <out_dir>

Layout (LeRobot v2.1's, with npz in place of parquet: neither pyarrow nor pandas is in the venv and
`lerobot` must not be added; every column keeps LeRobot's name, dtype and shape)::

    meta/info.json        fps, features, counts, path templates, catalog version, timing, source sha256
    meta/episodes.jsonl   one line per episode: index, tasks, length, trial, arm, skill_id, skill, args,
                          outcome (done | failed | unfinished), timing
    meta/tasks.jsonl      {"task_index", "task"} in order of first appearance
    meta/events.jsonl     every event and note line of the recording, verbatim, plus "episode_index"
                          (null when no episode owns it) and "timestamp" (s since that episode's start)
    data/chunk-000/episode_000000.npz   one per episode; arrays keyed by feature name (np.load, no pickle)

An episode is one skill run: the frames sharing (trial, skill_id). Skill ids restart at "sk1" on
every new server, so the trial tag the trials put on each line is part of the key; frames of two arms
interleaved in one tick split cleanly. Frames with no skill_id are skipped and counted.

Per-frame features (N = episode length; vectors float32, as LeRobot stores them):

    timestamp            (N,)    s since the episode's first frame (the server's clock, not resampled)
    frame_index          (N,)    0..N-1 within the episode
    episode_index        (N,)
    index                (N,)    0..total_frames-1 across the dataset
    task_index           (N,)    row of meta/tasks.jsonl
    observation.state    (N, 9)  ee_x ee_y ee_z ee_yaw gripper_width held target_x target_y target_z
    observation.setpoint (N, 3)  the motor loop's rate-limited xyz setpoint (motor-loop state, not
                                 the policy's output)
    observation.frozen   (N,)    bool: the mover ignored the action this tick (held, paused, tripped)
    phase                (N,)    str: the skill's machine phase name ("descend")
    action               (N, 6)  goal_x goal_y goal_z yaw gripper speed: what the policy commanded
                                 through ArmIO this tick (goto's goal, yaw goal and speed, grip width)

`held` is 1.0 when the gripper holds any object. `target_*` is the pose of the skill's `object`
argument and NaN when the skill has none or the object is missing from the world state. Units are
metres, radians and seconds in the arm's base frame. The recorder rounds to 1e-4, so float32 loses
nothing that was recorded.

Timing is validated, not resampled: the sim's clock ticks exactly 1/fps (the 1e-15 s jitter on
K2 is float error), and a resampled frame would hold an observation the policy never saw. Frame
periods off 1/fps are reported per episode and in info.json (max |dt - 1/fps| over periods that are
not gaps; gaps = periods over 1.5/fps). A clock that goes backwards, or a median period more than
10% off 1/fps (the wrong --fps), is a ValueError.

The task sentence comes from the skill's name and args. A recording can carry them as a note
`{"kind": "skill_start", "skill_id", "skill", "args"}` (what the writer should add at start();
trials.py can do it with JsonlRecorder.note today). K2-era recordings have no such line, so the name
is read from the "<name> finished: ..." text of the skill's skill_done event, `object` from that
event, and `place`/`onto` from the trial's note; episodes.jsonl says which ("task_source").

Writes are byte-deterministic: the same recording gives the same bytes (fixed zip timestamps, no
compression, no clocks in the metadata).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import numpy as np

from robojev import catalog

FORMAT = "robojev-lerobot-npz/1"       # LeRobot v2.1's layout and columns, npz instead of parquet
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
STATE_NAMES = ("ee_x", "ee_y", "ee_z", "ee_yaw", "gripper_width", "held", "target_x", "target_y", "target_z")
ACTION_NAMES = ("goal_x", "goal_y", "goal_z", "yaw", "gripper", "speed")
SETPOINT_NAMES = ("x", "y", "z")
FEATURES: dict[str, dict] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
    "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": list(STATE_NAMES)},
    "observation.setpoint": {"dtype": "float32", "shape": [3], "names": list(SETPOINT_NAMES)},
    "observation.frozen": {"dtype": "bool", "shape": [1], "names": None},
    "phase": {"dtype": "string", "shape": [1], "names": None},
    "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": list(ACTION_NAMES)},
}
GAP_FACTOR = 1.5           # a frame period over 1.5/fps is a gap (dropped ticks), not jitter
FPS_TOLERANCE = 0.10       # median period this far off 1/fps means the recording is at another rate
T_TOL_S = 1e-4             # LeRobot's default tolerance_s; also the recorder's 0.1 ms rounding
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

# One sentence per catalog skill; anything else falls back to "<name> k=v, ...".
TASK_TEMPLATES = {
    "pick_and_place": "pick up {object} and place it in {place}",
    "stack_on": "pick up {object} and stack it on {onto}",
    "push": "push {object} {direction} by {distance_cm} cm",
    "hand_over": "hand {object} to the person",
    "survey": "lift the gripper clear of the table",
    "hold": "stay still for {seconds} s",
}
_FINISHED = re.compile(r"^([\w.]+) finished: ")


@dataclass(frozen=True)
class Summary:
    """What export() wrote. `max_jitter_s` and `gaps` are over all episodes (see the module doc)."""
    out_dir: str
    episodes: int
    frames: int
    tasks: int
    fps: int
    max_jitter_s: float
    gaps: int
    skipped_frames: int          # frames with no skill_id
    unmatched_lines: int         # events/notes in meta/events.jsonl with episode_index null
    bytes: int


class ReplayFrame(NamedTuple):
    t: float                     # s since the episode's first frame
    observation: dict[str, Any]  # state (9,), setpoint (3,), frozen bool, phase str
    action: np.ndarray           # (6,) goal xyz, yaw, gripper, speed


@dataclass
class _Episode:
    trial: Any
    skill_id: str
    arm: str
    frames: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    skill: str | None = None
    args: dict = field(default_factory=dict)
    task_source: str = "none"    # skill_start | inferred | none


@dataclass
class _Recording:
    episodes: list[_Episode]
    sidecar: list[tuple[_Episode | None, dict]]   # every event/note line, with the episode owning it
    skipped_frames: int


# ------------------------------------------------------------------ reading the recording


def _trial_of(line: dict) -> Any:
    """The trial tag. A note's own `trial` field (a dict with "index") overrides the recorder's tag."""
    v = line.get("trial")
    return v.get("index") if isinstance(v, dict) else v


def _read(jsonl_path: Path) -> _Recording:
    episodes: dict[tuple, _Episode] = {}
    starts: dict[tuple, dict] = {}
    others: list[dict] = []
    skipped = 0
    with open(jsonl_path) as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            line = json.loads(raw)
            kind = line.get("type")
            if kind == "frame":
                if line.get("skill_id") is None:
                    skipped += 1
                    continue
                key = (_trial_of(line), line["skill_id"])
                ep = episodes.get(key)
                if ep is None:
                    ep = episodes[key] = _Episode(key[0], key[1], line["arm"])
                elif ep.arm != line["arm"]:
                    raise ValueError(f"{jsonl_path}:{n}: skill {key[1]} of trial {key[0]} "
                                     f"on both {ep.arm} and {line['arm']}")
                ep.frames.append(line)
            elif kind in ("event", "note"):
                others.append(line)
                if kind == "note" and line.get("kind") == "skill_start":
                    starts[(_trial_of(line), line.get("skill_id"))] = line
            else:
                raise ValueError(f"{jsonl_path}:{n}: unknown line type {kind!r}")

    by_trial: dict[Any, list[_Episode]] = {}
    for ep in episodes.values():
        by_trial.setdefault(ep.trial, []).append(ep)
    sidecar: list[tuple[_Episode | None, dict]] = []
    for line in others:
        trial = _trial_of(line)
        if line.get("skill_id") is not None:
            owner = episodes.get((trial, line["skill_id"]))
        else:                                    # a trial note: owned only when the trial ran one skill
            in_trial = by_trial.get(trial, [])
            owner = in_trial[0] if len(in_trial) == 1 else None
        if owner is not None and line["type"] == "event":
            owner.events.append(line)
        sidecar.append((owner, line))

    for key, ep in episodes.items():
        start = starts.get(key)
        if start is not None:
            ep.skill, ep.args, ep.task_source = start["skill"], dict(start.get("args") or {}), "skill_start"
        else:
            notes = [ln for o, ln in sidecar if o is ep and ln["type"] == "note"]
            _infer_k2_era(ep, notes)
    return _Recording(list(episodes.values()), sidecar, skipped)


def _infer_k2_era(ep: _Episode, notes: list[dict]) -> None:
    """Name from the skill_done text "<name> finished: ...", `object` from that event, `place` /
    `onto` from the trial note's setup dict. Leaves task_source "none" when there is no skill_done."""
    for ev in ep.events:
        m = _FINISHED.match(ev.get("text") or "") if ev.get("kind") == "skill_done" else None
        if m is None:
            continue
        ep.skill, ep.task_source = m.group(1), "inferred"
        if ev.get("object"):
            ep.args["object"] = ev["object"]
        for note in notes:
            setup = note.get("trial")
            if isinstance(setup, dict):
                ep.args.update({k: setup[k] for k in ("place", "onto") if isinstance(setup.get(k), str)})
        return


def task_text(skill: str | None, args: dict) -> str:
    """The task sentence for a skill and its args: "pick up block_1 and place it in slot_3"."""
    if skill is None:
        return "unknown skill"
    fields = {k: (v.replace("_", " ") if k == "direction" and isinstance(v, str) else v) for k, v in args.items()}
    if isinstance(args.get("distance"), (int, float)):
        fields["distance_cm"] = f"{args['distance'] * 100:g}"
    try:
        return TASK_TEMPLATES[skill].format(**fields)
    except KeyError:
        return " ".join([skill.replace("_", " "), ", ".join(f"{k}={args[k]}" for k in sorted(args))]).strip()


# ------------------------------------------------------------------ frames to arrays


def _state(frame: dict, target: str | None) -> list[float]:
    o = frame["observation"]
    pose = (o.get("objects") or {}).get(target) if target else None
    xyz = list(pose[:3]) if pose else [math.nan] * 3
    return [*o["ee"], o["yaw"], o["width"], float(o["holding"] is not None), *xyz]


def _action(frame: dict) -> list[float]:
    a = frame["action"]
    return [*a["goal"], a["yaw"], a["grip"], a["speed"]]


def _timing(t: np.ndarray, fps: int, where: str) -> dict:
    """Periods against 1/fps. Raises on a clock going backwards or a recording at another rate."""
    period = 1.0 / fps
    if len(t) < 2:
        return {"max_abs_jitter_s": 0.0, "gaps": 0, "longest_period_s": None}
    dt = np.diff(t)
    bad = np.flatnonzero(dt <= 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(f"{where}: t goes from {t[i]} to {t[i + 1]} at frame {i + 1}: not monotonic")
    median = float(np.median(dt))
    if abs(median - period) > FPS_TOLERANCE * period:
        raise ValueError(f"{where}: median frame period {median * 1e3:.2f} ms is not 1/{fps} s; wrong fps?")
    gap = dt > GAP_FACTOR * period
    jitter = np.abs(dt[~gap] - period)
    return {"max_abs_jitter_s": round(float(jitter.max()) if jitter.size else 0.0, 9),
            "gaps": int(gap.sum()), "longest_period_s": round(float(dt.max()), 6)}


def _arrays(ep: _Episode, episode_index: int, first_index: int, task_index: int) -> dict[str, np.ndarray]:
    n = len(ep.frames)
    t = np.array([f["t"] for f in ep.frames], dtype=np.float64)
    target = ep.args.get("object")
    return {
        "timestamp": (t - t[0]).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": np.arange(first_index, first_index + n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
        "observation.state": np.array([_state(f, target) for f in ep.frames], dtype=np.float32),
        "observation.setpoint": np.array([f["observation"]["setpoint"] for f in ep.frames], dtype=np.float32),
        "observation.frozen": np.array([bool(f["action"].get("frozen")) for f in ep.frames], dtype=np.bool_),
        "phase": np.array([f["phase"] for f in ep.frames], dtype=np.str_),
        "action": np.array([_action(f) for f in ep.frames], dtype=np.float32),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez, minus its wall-clock zip timestamps: same arrays, same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_EPOCH), buf.getvalue())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def _prepare(out: Path) -> None:
    """Replace a previous export in place; refuse to write into any other non-empty directory."""
    if out.exists() and any(out.iterdir()):
        if not (out / "meta" / "info.json").is_file():
            raise FileExistsError(f"{out} is not empty and holds no dataset (meta/info.json); not writing into it")
        for sub in ("data", "meta"):
            shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ export


def export(jsonl_path, out_dir, fps: int = 50, robot_type: str = "unknown",
           catalog_version: str | None = None) -> Summary:
    """Write the dataset for one recording. `robot_type` and `catalog_version` go into info.json
    because the recording does not hold them; catalog_version defaults to the current catalog's,
    marked "assumed" in info.json."""
    src, out = Path(jsonl_path), Path(out_dir)
    rec = _read(src)
    if not rec.episodes:
        raise ValueError(f"{src}: no frames with a skill_id")
    timings = [_timing(np.array([f["t"] for f in ep.frames]), fps, f"episode {i} (trial {ep.trial}, {ep.skill_id})")
               for i, ep in enumerate(rec.episodes)]         # all checked before anything is written
    _prepare(out)

    tasks: dict[str, int] = {}
    owner_index: dict[int, int] = {}
    episode_rows: list[dict] = []
    total, max_jitter, gaps = 0, 0.0, 0
    for i, (ep, timing) in enumerate(zip(rec.episodes, timings)):
        task = task_text(ep.skill, ep.args)
        task_index = tasks.setdefault(task, len(tasks))
        arrays = _arrays(ep, i, total, task_index)
        _write_npz(out / DATA_PATH.format(episode_chunk=i // CHUNKS_SIZE, episode_index=i), arrays)
        ends = [e for e in ep.events if e.get("kind") in ("skill_done", "skill_failed")]
        episode_rows.append({
            "episode_index": i, "tasks": [task], "length": len(ep.frames), "trial": ep.trial, "arm": ep.arm,
            "skill_id": ep.skill_id, "skill": ep.skill, "args": ep.args, "task_source": ep.task_source,
            "outcome": {"skill_done": "done", "skill_failed": "failed"}[ends[-1]["kind"]] if ends else "unfinished",
            "reason": ends[-1]["text"] if ends and ends[-1]["kind"] == "skill_failed" else None,
            "t0": ep.frames[0]["t"], "timing": timing})
        owner_index[id(ep)] = i
        total += len(ep.frames)
        max_jitter, gaps = max(max_jitter, timing["max_abs_jitter_s"]), gaps + timing["gaps"]

    sidecar_rows, unmatched = [], 0
    for owner, line in rec.sidecar:
        idx = owner_index[id(owner)] if owner is not None else None
        unmatched += idx is None
        ts = round(line["t"] - owner.frames[0]["t"], 6) if owner is not None and "t" in line else None
        sidecar_rows.append({"episode_index": idx, "timestamp": ts, **line})

    n = len(rec.episodes)
    info = {
        "codebase_version": FORMAT,
        "lerobot_layout": "v2.1",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": n,
        "total_frames": total,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (n - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{n}"},
        "data_path": DATA_PATH,
        "video_path": None,
        "features": FEATURES,
        "catalog_version": catalog_version or catalog.CATALOG_VERSION,
        "catalog_version_source": "argument" if catalog_version else "assumed: catalog.py at export time",
        "units": "m, rad, s; positions in the arm's base frame",
        "timing": {"period_s": 1.0 / fps, "max_abs_jitter_s": max_jitter, "gaps": gaps,
                   "gap_factor": GAP_FACTOR, "resampled": False},
        "skipped_frames": rec.skipped_frames,
        "source": {"file": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
    }
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    _write_jsonl(meta / "episodes.jsonl", episode_rows)
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    _write_jsonl(meta / "events.jsonl", sidecar_rows)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return Summary(str(out), n, total, len(tasks), fps, max_jitter, gaps, rec.skipped_frames, unmatched, size)


# ------------------------------------------------------------------ replay


def load_info(out_dir) -> dict:
    return json.loads((Path(out_dir) / "meta" / "info.json").read_text())


def load_episode(out_dir, episode_index: int) -> dict[str, np.ndarray]:
    """Every feature array of one episode, keyed by feature name."""
    info = load_info(out_dir)
    if not 0 <= episode_index < info["total_episodes"]:
        raise IndexError(f"episode {episode_index} not in 0..{info['total_episodes'] - 1}")
    path = Path(out_dir) / info["data_path"].format(episode_chunk=episode_index // info["chunks_size"],
                                                    episode_index=episode_index)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if set(arrays) != set(info["features"]):
        raise ValueError(f"{path}: arrays {sorted(arrays)} != features {sorted(info['features'])}")
    if not (arrays["episode_index"] == episode_index).all():
        raise ValueError(f"{path}: episode_index column is not {episode_index}")
    return arrays


def replay(out_dir, episode_index: int) -> Iterator[ReplayFrame]:
    """The episode's frames in order, as the policy saw them (observation) and what it commanded (action)."""
    a = load_episode(out_dir, episode_index)
    for i in range(len(a["frame_index"])):
        obs = {"state": a["observation.state"][i], "setpoint": a["observation.setpoint"][i],
               "frozen": bool(a["observation.frozen"][i]), "phase": str(a["phase"][i])}
        yield ReplayFrame(float(a["timestamp"][i]), obs, a["action"][i])


def check_replay(jsonl_path, out_dir, atol: float = 1e-6) -> int:
    """Replay every episode and compare it with the recording frame for frame: the action (goal xyz,
    yaw goal, grip, speed) and the state within `atol`, the timestamp within T_TOL_S, the phase exactly.
    Raises AssertionError naming the first differing frame; returns the number of frames checked."""
    rec = _read(Path(jsonl_path))
    info = load_info(out_dir)
    if info["total_episodes"] != len(rec.episodes):
        raise AssertionError(f"{info['total_episodes']} episodes exported, {len(rec.episodes)} recorded")
    checked = 0
    for i, ep in enumerate(rec.episodes):
        frames = list(replay(out_dir, i))
        if len(frames) != len(ep.frames):
            raise AssertionError(f"episode {i}: {len(frames)} frames replayed, {len(ep.frames)} recorded")
        t0 = ep.frames[0]["t"]
        for j, (got, raw) in enumerate(zip(frames, ep.frames)):
            a = raw["action"]
            want = np.array([*a["goal"], a["yaw"], a["grip"], a["speed"]])
            where = f"episode {i} (trial {ep.trial}, {ep.skill_id}) frame {j}"
            if not np.allclose(got.action, want, rtol=0, atol=atol):
                raise AssertionError(f"{where}: replayed action {got.action.tolist()} != recorded {want.tolist()}")
            if not np.allclose(got.observation["state"], _state(raw, ep.args.get("object")), rtol=0, atol=atol,
                               equal_nan=True):
                raise AssertionError(f"{where}: replayed state {got.observation['state'].tolist()} != recorded")
            if abs(got.t - (raw["t"] - t0)) > T_TOL_S or got.observation["phase"] != raw["phase"]:
                raise AssertionError(f"{where}: t/phase {got.t}/{got.observation['phase']} != "
                                     f"{raw['t'] - t0}/{raw['phase']}")
            checked += 1
    return checked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl", type=Path, help="a JsonlRecorder recording")
    ap.add_argument("out_dir", type=Path, help="dataset directory (a previous export there is replaced)")
    ap.add_argument("--fps", type=int, default=50, help="the policy rate the recording was made at")
    ap.add_argument("--robot-type", default="unknown", help='info.json robot_type, e.g. "widowx-sim"')
    ap.add_argument("--catalog-version", default=None, help="default: the current catalog.CATALOG_VERSION")
    a = ap.parse_args(argv)
    s = export(a.jsonl, a.out_dir, a.fps, a.robot_type, a.catalog_version)
    print(json.dumps(asdict(s), indent=2))
    print(f"replay check: {check_replay(a.jsonl, a.out_dir)} frames match the recording")
    return 0


if __name__ == "__main__":
    sys.exit(main())
