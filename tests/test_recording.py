"""H6: the JSONL recording to a LeRobot-style dataset, and replay. No MuJoCo needed: a synthetic
recording written through the real JsonlRecorder, and the committed K2 recording."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from robojev.protocol import RobotEvent
from robojev.recording import lerobot_export as lx
from robojev.recording.recorder import JsonlRecorder

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
K2 = RESULTS / "k2_trials.jsonl"
K2_DATASET = RESULTS / "k2_trials_lerobot"
DT = 0.02


def _frame(rec, t, arm, sid, phase, ee, goal, holding=None, objects=None, frozen=False):
    rec.frame(t, arm, sid, phase,
              {"ee": list(ee), "yaw": 0.5, "width": 0.04, "holding": holding, "setpoint": list(ee), "lag": 0.0,
               "mode": "running", "objects": objects or {}},
              {"goal": list(goal), "speed": 0.15, "yaw": 0.25, "grip": 0.07, "frozen": frozen})


@pytest.fixture
def synthetic(tmp_path) -> Path:
    """Trial 0: a pick_and_place on arm_0 announced by a skill_start note, done, with a heartbeat_lost
    in the middle. Trial 1: stack_on (arm_0) and push (arm_1) interleaved tick by tick, both "sk1"/"sk2"
    ids restarting as on a new server; the push fails. One frame with no skill, and a trial verdict
    that no single episode owns."""
    p = tmp_path / "rec.jsonl"
    rec = JsonlRecorder(p)
    rec.tag = {"trial": 0}
    rec.note(kind="skill_start", skill_id="sk1", arm="arm_0", skill="pick_and_place",
             args={"object": "block_1", "place": "slot_3"})
    for i in range(5):
        _frame(rec, i * DT, "arm_0", "sk1", "approach" if i < 3 else "descend", (0.2 + i * 0.01, 0.0, 0.1),
               (0.3, 0.05, 0.1), holding="block_1" if i == 4 else None, objects={"block_1": [0.3, 0.05, 0.0, 0.1]})
        if i == 2:
            rec.event(RobotEvent(i * DT, "heartbeat_lost", "no heartbeat for 1.0 s"))
    rec.event(RobotEvent(4 * DT, "skill_done", "pick_and_place finished: block_1 placed at slot_3",
                         skill_id="sk1", arm="arm_0", object="block_1"))
    rec.note(kind="verdict", verdict={"ok": True})
    rec.tag = {"trial": 1}
    rec.note(kind="skill_start", skill_id="sk1", arm="arm_0", skill="stack_on", args={"object": "b2", "onto": "b3"})
    rec.note(kind="skill_start", skill_id="sk2", arm="arm_1", skill="push",
             args={"object": "b4", "direction": "toward_robot", "distance": 0.05})
    for i in range(3):
        _frame(rec, 1.0 + i * DT, "arm_0", "sk1", "approach", (0.25, 0.0, 0.1), (0.3, 0.1, 0.1))
        _frame(rec, 1.0 + i * DT, "arm_1", "sk2", "push", (0.25, 0.0, 0.1), (0.3, 0.2, 0.02), frozen=i == 2)
    rec.event(RobotEvent(1.04, "skill_failed", "blocked: b4 did not move", skill_id="sk2", arm="arm_1", object="b4"))
    _frame(rec, 1.06, "arm_0", None, "", (0.25, 0.0, 0.1), (0.25, 0.0, 0.1))
    rec.note(kind="verdict", verdict={"ok": False})
    rec.close()
    return p


def _meta(out: Path, name: str) -> list[dict]:
    return [json.loads(ln) for ln in (out / "meta" / name).read_text().splitlines()]


def _tree(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------ synthetic recording


def test_synthetic_schema_and_episodes(synthetic, tmp_path):
    out = tmp_path / "ds"
    s = lx.export(synthetic, out, robot_type="test")
    assert (s.episodes, s.frames, s.skipped_frames, s.tasks) == (3, 11, 1, 3)
    info = lx.load_info(out)
    assert info["fps"] == 50 and info["total_episodes"] == 3 and info["total_frames"] == 11
    assert info["catalog_version"] == "0.1" and info["robot_type"] == "test"
    assert info["features"]["observation.state"]["names"] == list(lx.STATE_NAMES)
    assert info["features"]["action"]["shape"] == [6]

    eps = _meta(out, "episodes.jsonl")
    assert [(e["trial"], e["skill_id"], e["arm"], e["length"]) for e in eps] == \
        [(0, "sk1", "arm_0", 5), (1, "sk1", "arm_0", 3), (1, "sk2", "arm_1", 3)]
    assert [e["tasks"][0] for e in eps] == ["pick up block_1 and place it in slot_3", "pick up b2 and stack it on b3",
                                           "push b4 toward robot by 5 cm"]
    assert [e["outcome"] for e in eps] == ["done", "unfinished", "failed"]
    assert eps[2]["reason"] == "blocked: b4 did not move"
    assert {e["task_source"] for e in eps} == {"skill_start"}

    a = lx.load_episode(out, 0)
    for name, spec in info["features"].items():
        assert a[name].shape[0] == 5
        assert a[name].shape[1:] == (() if spec["shape"] == [1] else tuple(spec["shape"])), name
    assert a["index"].tolist() == list(range(5)) and lx.load_episode(out, 1)["index"].tolist() == [5, 6, 7]
    assert a["phase"].tolist() == ["approach"] * 3 + ["descend"] * 2
    assert a["observation.state"][:, 5].tolist() == [0, 0, 0, 0, 1]           # held flag
    np.testing.assert_allclose(a["observation.state"][0, 6:], [0.3, 0.05, 0.0], atol=1e-7)
    assert np.isnan(lx.load_episode(out, 1)["observation.state"][:, 6:]).all()   # b2 not in the world state
    assert lx.load_episode(out, 2)["observation.frozen"].tolist() == [False, False, True]


def test_synthetic_events_sidecar(synthetic, tmp_path):
    out = tmp_path / "ds"
    s = lx.export(synthetic, out)
    ev = _meta(out, "events.jsonl")
    by_kind = {(e["type"], e["kind"], _trial(e)): e for e in ev}
    assert by_kind[("event", "skill_done", 0)]["episode_index"] == 0
    assert by_kind[("event", "skill_done", 0)]["timestamp"] == pytest.approx(0.08)
    assert by_kind[("event", "skill_failed", 1)]["episode_index"] == 2
    assert by_kind[("event", "heartbeat_lost", 0)]["episode_index"] == 0      # trial 0 ran one skill
    assert by_kind[("note", "verdict", 0)]["episode_index"] == 0
    starts = [e["episode_index"] for e in ev if e.get("kind") == "skill_start"]
    assert starts == [0, 1, 2]                                               # keyed by their skill_id
    assert by_kind[("note", "verdict", 1)]["episode_index"] is None          # trial 1 ran two skills: ambiguous
    assert s.unmatched_lines == sum(e["episode_index"] is None for e in ev) == 1


def _trial(line):
    return line["trial"]["index"] if isinstance(line["trial"], dict) else line["trial"]


def test_synthetic_replay_is_frame_accurate(synthetic, tmp_path):
    out = tmp_path / "ds"
    lx.export(synthetic, out)
    assert lx.check_replay(synthetic, out) == 11
    frames = list(lx.replay(out, 2))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.02, 0.04])
    np.testing.assert_allclose(frames[0].action, [0.3, 0.2, 0.02, 0.25, 0.07, 0.15], atol=1e-7)
    assert frames[2].observation["frozen"] and frames[0].observation["phase"] == "push"
    with pytest.raises(IndexError):
        next(lx.replay(out, 3))


def test_check_replay_catches_a_changed_action(synthetic, tmp_path):
    out = tmp_path / "ds"
    lx.export(synthetic, out)
    path = out / lx.DATA_PATH.format(episode_chunk=0, episode_index=1)
    a = lx.load_episode(out, 1)
    a["action"][2, 0] += 1e-3
    lx._write_npz(path, a)
    with pytest.raises(AssertionError, match=r"episode 1 \(trial 1, sk1\) frame 2"):
        lx.check_replay(synthetic, out)


def test_timing_is_validated_and_gaps_reported(tmp_path):
    def rec_with(ts, name):
        p = tmp_path / name
        rec = JsonlRecorder(p)
        for t in ts:
            _frame(rec, t, "arm_0", "sk1", "approach", (0.2, 0, 0.1), (0.3, 0, 0.1))
        rec.close()
        return p

    s = lx.export(rec_with([0, 0.02, 0.0401, 0.06, 0.12, 0.14], "gap.jsonl"), tmp_path / "gap")
    assert s.gaps == 1 and s.max_jitter_s == pytest.approx(1e-4, abs=1e-9)
    with pytest.raises(ValueError, match="not monotonic"):
        lx.export(rec_with([0, 0.02, 0.02, 0.04], "dup.jsonl"), tmp_path / "dup")
    with pytest.raises(ValueError, match="wrong fps"):
        lx.export(rec_with([0, 0.05, 0.1, 0.15], "20hz.jsonl"), tmp_path / "20hz")
    assert not (tmp_path / "dup").exists()                # nothing written for a rejected recording


def test_refuses_to_write_into_a_foreign_directory(synthetic, tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        lx.export(synthetic, tmp_path / "other")
    assert (tmp_path / "other" / "keep.txt").read_text() == "mine"


def test_task_text():
    assert lx.task_text("pick_and_place", {"object": "block_1", "place": "slot_3"}) == \
        "pick up block_1 and place it in slot_3"
    assert lx.task_text("hold", {"seconds": 2}) == "stay still for 2 s"
    assert lx.task_text("widowx.wiggle_free", {"object": "b1", "amp": 0.01}) == "widowx.wiggle free amp=0.01, object=b1"
    assert lx.task_text("pick_and_place", {"object": "b1"}) == "pick and place object=b1"
    assert lx.task_text(None, {}) == "unknown skill"


# ------------------------------------------------------------------ the committed K2 recording


@pytest.fixture(scope="module")
def k2(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("k2") / "ds"
    s = lx.export(K2, out, robot_type="widowx-sim")
    assert (s.episodes, s.frames, s.skipped_frames, s.unmatched_lines) == (20, 7014, 0, 0)
    return out


def test_k2_episodes_and_tasks(k2):
    eps = _meta(k2, "episodes.jsonl")
    assert len(eps) == 20 and [e["trial"] for e in eps] == list(range(20))
    assert sum(e["length"] for e in eps) == 7014
    assert all(e["skill"] == "pick_and_place" and e["outcome"] == "done" and e["task_source"] == "inferred"
               for e in eps)
    notes = {e["episode_index"]: e for e in _meta(k2, "events.jsonl") if e["type"] == "note"}
    for e in eps:                                   # the task's place is the trial's sampled destination
        assert e["tasks"] == [f"pick up block_1 and place it in {notes[e['episode_index']]['trial']['place']}"]
    assert eps[0]["tasks"] == ["pick up block_1 and place it in slot_2"]
    tasks = _meta(k2, "tasks.jsonl")
    assert [t["task_index"] for t in tasks] == list(range(len(tasks)))
    assert {t["task"] for t in tasks} == {e["tasks"][0] for e in eps}


def test_k2_timestamps_monotonic_at_50hz(k2):
    for i in range(20):
        ts = lx.load_episode(k2, i)["timestamp"].astype(np.float64)
        assert ts[0] == 0.0
        dt = np.diff(ts)
        assert (dt > 0).all()
        assert np.abs(dt - 1 / 50).max() < lx.T_TOL_S
        assert ts[-1] == pytest.approx((len(ts) - 1) / 50, abs=lx.T_TOL_S)
    assert lx.load_info(k2)["timing"]["gaps"] == 0


def test_k2_replay_is_frame_accurate(k2):
    assert lx.check_replay(K2, k2) == 7014
    # independent of check_replay: the first frame of trial 0, straight from the JSONL
    raw = json.loads(K2.open().readline())
    f0 = next(lx.replay(k2, 0))
    a = raw["action"]
    np.testing.assert_allclose(f0.action, [*a["goal"], a["yaw"], a["grip"], a["speed"]], atol=1e-6)
    o = raw["observation"]
    np.testing.assert_allclose(f0.observation["state"], [*o["ee"], o["yaw"], o["width"], 0.0,
                                                         *o["objects"]["block_1"][:3]], atol=1e-6)


def test_k2_export_is_deterministic(k2, tmp_path):
    again = tmp_path / "again"
    lx.export(K2, again, robot_type="widowx-sim")
    assert _tree(again) == _tree(k2)


def test_committed_k2_dataset_matches_a_fresh_export(k2):
    """experiments/results/k2_trials_lerobot is this exporter's output for the committed recording."""
    assert _tree(K2_DATASET) == _tree(k2)
    assert math.fsum(len(b) for b in _tree(K2_DATASET).values()) < 5e6
