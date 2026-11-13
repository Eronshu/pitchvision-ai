import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pitchvision import atd, config, pipeline
from starter import baseline
from tests.synthetic import write_fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
N = 600
BLOCKS = [(1, 50, 299), (2, 350, 549)]
# Scripted truth: who dribbles the ball, frame span [start, stop).
SCRIPT = [("H", 50, 175), ("A", 175, 300), ("H", 350, 550)]


def scripted_frames():
    """Home slots 0 and 2, away slots 1 and 3. The controlling player runs with the ball; the field
    keeps moving at 3 m/s so nothing reads as a stoppage."""
    length, width = PITCH
    owner = {}
    for team, start, stop in SCRIPT:
        for t in range(start, stop):
            owner[t] = (team, t - start)
    frames = []
    for t in range(N):
        team, step = owner.get(t, ("H", t % 100))
        bx, by = 20.0 + 0.2 * step, 34.0
        holder = (bx - 0.5, by)
        far = [(bx + 15.0, 10.0), (bx - 15.0, 58.0), (bx + 25.0, 60.0)]
        if team == "H":
            players = [holder, far[0], far[1], far[2]]
        else:
            players = [far[0], holder, far[1], far[2]]
        frames.append({"players": [(x / length, y / width, 3.0) for x, y in players],
                       "ball": (bx / length, by / width)})
    return frames


def scripted_gt():
    gt = np.full(N, "X", dtype="<U1")
    for team, start, stop in SCRIPT:
        gt[start:stop] = team
    return gt


@pytest.fixture
def fixture_files(tmp_path):
    return write_fixture(tmp_path, scripted_frames())


@pytest.fixture
def labels(fixture_files):
    tr = atd.load_arrays(*fixture_files, N, PITCH)
    return pipeline.run(tr, config.defaults(), FPS, PITCH, BLOCKS)


def test_shape_and_x_outside(labels):
    assert labels.shape == (N,)
    assert set(labels[:50]) == {"X"}
    assert set(labels[300:350]) == {"X"}
    assert set(labels[550:]) == {"X"}
    assert set(labels[50:300]) <= {"H", "A", "D"}


def test_no_leak_across_halves(labels):
    assert labels[299] == "A"
    assert labels[350] == "H"


def test_metric_sanity(labels):
    result = pm.evaluate(pm.label_codes(scripted_gt()), pm.label_codes(labels))
    assert result["spell_f1_mean"] >= 0.9


def test_valid_submission(labels, tmp_path):
    path = baseline.write_submission(str(tmp_path / "sub.csv"), labels)
    assert pm.validate_submission(pm.read_label_csv(path), N) is None


def _cli(fixture_files, out_dir, *extra):
    meta, raw = fixture_files
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"),
           "--config", os.path.join(ROOT, "configs", "v1.json"), "--out", str(out_dir),
           "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
           "--blocks", json.dumps(BLOCKS)] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_cli_manifest(fixture_files, tmp_path):
    result = _cli(fixture_files, tmp_path / "run")
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "run" / "manifest.json") as handle:
        manifest = json.load(handle)
    assert {"config", "config_sha256", "git_commit", "git_dirty", "inputs", "output_sha256",
            "n_rows", "label_counts", "mode"} <= set(manifest)
    assert manifest["output_sha256"] == _sha(tmp_path / "run" / "submission.csv")
    assert manifest["n_rows"] == N and manifest["mode"] == "v1"
    assert manifest["config_sha256"] == config.sha256(config.defaults())
    assert set(manifest["inputs"]) == {"atd.xml", "atd.txt"}


def test_deterministic(fixture_files, tmp_path):
    for name in ("a", "b"):
        assert _cli(fixture_files, tmp_path / name).returncode == 0
    assert _sha(tmp_path / "a" / "submission.csv") == _sha(tmp_path / "b" / "submission.csv")


def test_cli_fallback_is_official_smooth50(fixture_files, tmp_path):
    result = _cli(fixture_files, tmp_path / "fb", "--fallback-baseline")
    assert result.returncode == 0, result.stderr
    meta, raw = fixture_files
    codes = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta), N, PITCH)
    expected = baseline.write_submission(str(tmp_path / "expected.csv"), baseline.smooth(codes, 50))
    assert _sha(tmp_path / "fb" / "baseline_smooth50.csv") == _sha(expected)
    with open(tmp_path / "fb" / "manifest.json") as handle:
        assert json.load(handle)["mode"] == "fallback_baseline_smooth50"


def test_cli_rejects_missing_out(fixture_files):
    meta, raw = fixture_files
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"),
                             "--config", os.path.join(ROOT, "configs", "v1.json")],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "--out" in result.stderr


# -- review fixes (PR #1): end-to-end stoppages, and Smart Tagging kept off until wired ---------

def stoppage_frames():
    """Half 1: H dribbles near the touchline, the ball runs out (seen beyond the margin for 2
    frames), is lost for 4 s, A restarts and dribbles. Half 2: pre-kickoff, ball still on the spot with a slow field and the kicker
    stepping up, then H kicks off and dribbles."""
    length, width = PITCH
    frames = []
    for t in range(N):
        speed, kicker = 3.0, None
        if 50 <= t < 144:
            team, ball = "H", (20.0 + 0.2 * (t - 50), 4.0)
        elif 144 <= t < 152:
            team, ball = "H", (38.8, 4.0 - 0.8 * (t - 143))   # 20 m/s over the touchline
            speed = 2.0 if t >= 150 else 3.0
        elif 152 <= t < 250:
            team, ball, speed = "H", None, 2.0
        elif 250 <= t < 300:
            team, ball = "A", (40.0 + 0.2 * (t - 250), 0.5)
        elif 350 <= t < 450:
            team, ball, speed = "H", (52.65, 34.0), 0.2
            kicker = (52.05, 34.0) if t >= 430 else None
        elif 450 <= t < 550:
            team, ball = "H", (52.65 + 0.2 * (t - 450), 34.0)
        else:
            team, ball = "H", (20.0 + 0.2 * (t % 100), 34.0)

        anchor = ball if ball is not None else (40.0, 10.0)
        holder = kicker or ((anchor[0] - 0.5, anchor[1]) if 350 > t or t >= 450
                            else (anchor[0] - 12.0, 20.0))
        far = [(anchor[0] + 15.0, 50.0), (anchor[0] - 15.0, 58.0), (anchor[0] + 25.0, 60.0)]
        players = [holder, far[0], far[1], far[2]] if team == "H" else \
            [far[0], holder, far[1], far[2]]
        frames.append({"players": [(x / length, y / width, speed) for x, y in players],
                       "ball": None if ball is None else (ball[0] / length, ball[1] / width)})
    return frames


def stoppage_gt():
    gt = np.full(N, "X", dtype="<U1")
    for label, start, stop in [("H", 50, 150), ("D", 150, 250), ("A", 250, 300),
                               ("D", 350, 450), ("H", 450, 550)]:
        gt[start:stop] = label
    return gt


@pytest.fixture
def stoppage_labels(tmp_path):
    meta, raw = write_fixture(tmp_path, stoppage_frames())
    return pipeline.run(atd.load_arrays(meta, raw, N, PITCH), config.defaults(), FPS, PITCH,
                        BLOCKS)


def test_home_dead_away_end_to_end(stoppage_labels):
    assert set(stoppage_labels[60:145]) == {"H"}
    assert set(stoppage_labels[150:250]) == {"D"}
    assert set(stoppage_labels[255:300]) == {"A"}


def test_pre_kickoff_dead_end_to_end(stoppage_labels):
    assert set(stoppage_labels[360:450]) == {"D"}
    assert set(stoppage_labels[455:550]) == {"H"}


def test_stoppage_metric_sanity(stoppage_labels):
    result = pm.evaluate(pm.label_codes(stoppage_gt()), pm.label_codes(stoppage_labels))
    assert result["spell_f1_mean"] >= 0.9
    assert result["per_class"]["D"]["f1"] >= 0.9


def test_cli_tagging_needs_the_tagging_file(fixture_files, tmp_path):
    path = tmp_path / "tagging.json"
    path.write_text(json.dumps({"tagging": {"use_setpiece_team": True}}))
    meta, raw = fixture_files
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"), "--config", str(path),
         "--out", str(tmp_path / "run"), "--atd-meta", meta, "--atd-raw", raw,
         "--n-frames", str(N), "--blocks", json.dumps(BLOCKS),
         "--data-dir", str(tmp_path / "empty")],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "smart_tagging" in result.stderr
    assert not (tmp_path / "run" / "submission.csv").exists()


def test_cli_tagging_records_its_input(fixture_files, tmp_path):
    from tests.synthetic import write_smart_tagging
    write_smart_tagging(tmp_path / "bundle" / "tactical" / "BAR-ATH_tactical_smart_tagging.xml",
                        [("SET PIECES", 2.0, 4.0, {"Team": "Home FC"})])
    path = tmp_path / "tagging.json"
    path.write_text(json.dumps({"tagging": {"use_setpiece_team": True,
                                            "use_setpiece_dead": True}}))
    meta, raw = fixture_files
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"), "--config", str(path),
         "--out", str(tmp_path / "run"), "--atd-meta", meta, "--atd-raw", raw,
         "--n-frames", str(N), "--blocks", json.dumps(BLOCKS),
         "--data-dir", str(tmp_path / "bundle")],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "run" / "manifest.json") as handle:
        manifest = json.load(handle)
    assert "BAR-ATH_tactical_smart_tagging.xml" in manifest["inputs"]
    assert manifest["config"]["tagging"] == {"use_setpiece_dead": True, "use_setpiece_team": True}
