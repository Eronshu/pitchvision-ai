"""End-to-end submission generation with scripts/run_v2.py on synthetic ATD (no match data)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pitchvision import atd
from pvext.v2 import features
from tests.ext.v2 import release_fixture as rf
from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, PITCH, scripted_frames

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(ROOT, "scripts", "run_v2.py")
TEAM_FILE = "PitchVision AI.csv"


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return rf.build_release(tmp_path_factory.mktemp("release"))[0]


@pytest.fixture(scope="module")
def atd_files(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("atd"), scripted_frames())


def run(release, atd_files, out, *extra):
    meta, raw = atd_files
    cmd = [sys.executable, RUN, "--config", release, "--out", str(out), "--atd-meta", meta,
           "--atd-raw", raw, "--n-frames", str(N), "--blocks", json.dumps(BLOCKS)] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def run_dir(release, atd_files, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    result = run(release, atd_files, out)
    assert result.returncode == 0, result.stderr
    return out


def read_csv(path):
    with open(path) as handle:
        lines = handle.read().split("\n")
    return lines[0], [line.split(",") for line in lines[1:] if line]


def test_submission_format(run_dir):
    header, rows = read_csv(run_dir / TEAM_FILE)
    assert header == "frame,label"
    assert len(rows) == N
    assert [int(f) for f, _ in rows] == list(range(N))
    labels = np.array([lab for _, lab in rows])
    assert set(labels) <= {"H", "A", "D", "X"}


def test_graded_blocks_and_x_outside(run_dir):
    labels = np.array([lab for _, lab in read_csv(run_dir / TEAM_FILE)[1]])
    graded = np.zeros(N, dtype=bool)
    for _, first, last in BLOCKS:
        graded[first:last + 1] = True
        assert labels[first - 1] == "X" and labels[first] != "X"       # exact boundaries
        assert labels[last] != "X" and labels[last + 1] == "X"
    assert set(labels[~graded]) == {"X"} and "X" not in set(labels[graded])


def test_frame_id_minus_one_alignment(tmp_path):
    frames = scripted_frames()
    frames[100] = dict(frames[100], ball=None)        # frame_id 101 -> video frame 100
    meta, raw = write_fixture(tmp_path, frames)
    tr = atd.load_arrays(meta, raw, N, PITCH)
    assert not tr.ball_present[100] and tr.ball_present[99] and tr.ball_present[101]
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    col = X[:, names.index("ball_present")]
    assert col[100] == 0 and col[99] == 1 and col[101] == 1


def test_official_validator_passes(run_dir):
    result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                             str(run_dir / TEAM_FILE), "--rows", str(N)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_and_verifier(run_dir, release):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "v2_m1" and manifest["output"] == TEAM_FILE and manifest["valid"]
    assert manifest["n_rows"] == N and manifest["output_sha256"] == rf.sha(str(run_dir / TEAM_FILE))
    with open(release) as handle:
        rel = json.load(handle)
    assert manifest["model_sha256"] == rel["model_sha256"]
    assert manifest["threshold"] == rel["threshold"]
    assert {"config_sha256", "git_commit", "git_dirty", "source_sha256", "decision_margin_min",
            "dependencies", "feature_names_sha256"} <= set(manifest)
    verify = os.path.join(ROOT, "scripts", "verify_submission.py")
    ok = subprocess.run([sys.executable, verify, str(run_dir / "manifest.json"), "--rows", str(N)],
                        capture_output=True, text=True, cwd=ROOT)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_verifier_catches_a_tampered_output(release, atd_files, tmp_path):
    out = tmp_path / "run"
    assert run(release, atd_files, out).returncode == 0
    path = out / TEAM_FILE
    path.write_text(path.read_text().replace("\n599,X", "\n599,H"))
    verify = os.path.join(ROOT, "scripts", "verify_submission.py")
    bad = subprocess.run([sys.executable, verify, str(out / "manifest.json"), "--rows", str(N)],
                         capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode != 0


def test_two_runs_are_byte_identical(run_dir, release, atd_files, tmp_path):
    assert run(release, atd_files, tmp_path / "again").returncode == 0
    assert (tmp_path / "again" / TEAM_FILE).read_bytes() == (run_dir / TEAM_FILE).read_bytes()


def test_invalid_model_is_rejected_before_data_access(release, tmp_path):
    bad = rf.edit_release(release, tmp_path / "bad.json", model_sha256="0" * 64)
    meta, raw = str(tmp_path / "missing.xml"), str(tmp_path / "missing.txt")   # never read
    result = run(bad, (meta, raw), tmp_path / "out")
    assert result.returncode == 3 and "release check failed" in result.stderr
    assert not (tmp_path / "out").exists()


def test_output_inside_repository_and_bad_name_are_refused(release, atd_files, tmp_path):
    assert run(release, atd_files, os.path.join(ROOT, "run_v2_out")).returncode == 2
    assert not os.path.exists(os.path.join(ROOT, "run_v2_out"))
    assert run(release, atd_files, tmp_path / "o", "--output-name", "../x.csv").returncode == 2
