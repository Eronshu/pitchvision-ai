"""End-to-end HA3 submission generation with scripts/run_ha3.py on synthetic ATD (no match data)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.ext.ha.test_deploy import build_ha3_release
from tests.ext.v2 import release_fixture as rf
from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, scripted_frames

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUN = os.path.join(ROOT, "scripts", "run_ha3.py")
RUN_V2 = os.path.join(ROOT, "scripts", "run_v2.py")
TEAM_FILE = "PitchVision AI.csv"


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return build_ha3_release(tmp_path_factory.mktemp("ha3"))


@pytest.fixture(scope="module")
def atd_files(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("atd"), scripted_frames())


def run(release, atd_files, out, *extra, script=RUN):
    meta, raw = atd_files
    cmd = [sys.executable, script, "--config", release, "--out", str(out), "--atd-meta", meta,
           "--atd-raw", raw, "--n-frames", str(N), "--blocks", json.dumps(BLOCKS)] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def run_dir(release, atd_files, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    result = run(release, atd_files, out)
    assert result.returncode == 0, result.stderr
    return out


def labels_of(path):
    with open(path) as handle:
        lines = handle.read().split("\n")
    assert lines[0] == "frame,label"
    rows = [line.split(",") for line in lines[1:] if line]
    assert [int(f) for f, _ in rows] == list(range(len(rows)))
    return np.array([lab for _, lab in rows])


def test_submission_format_and_blocks(run_dir):
    labels = labels_of(run_dir / TEAM_FILE)
    assert len(labels) == N and set(labels) <= {"H", "A", "D", "X"}
    graded = np.zeros(N, dtype=bool)
    for _, first, last in BLOCKS:
        graded[first:last + 1] = True
    assert set(labels[~graded]) == {"X"} and "X" not in set(labels[graded])


def test_official_validator_passes(run_dir):
    result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                             str(run_dir / TEAM_FILE), "--rows", str(N)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dead_and_x_frames_equal_m1_with_the_same_dead_model(run_dir, release, atd_files, tmp_path):
    # the synthetic dead-ball model is the synthetic M1 final's regression, so run_v2's D/X match;
    # with the real league model the HA3 D frames generally differ from the M1 final fallback
    with open(release) as handle:
        m1_config = json.load(handle)["m1_release_config"]
    assert run(m1_config, atd_files, tmp_path / "m1", script=RUN_V2).returncode == 0
    ha3 = labels_of(run_dir / TEAM_FILE)
    m1 = labels_of(tmp_path / "m1" / TEAM_FILE)
    assert np.array_equal(ha3 == "D", m1 == "D") and np.array_equal(ha3 == "X", m1 == "X")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["m1_dead_model_label_counts"] == {k: int(np.sum(m1 == k)) for k in "HADX"}


def test_manifest_and_verifier(run_dir, release):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "v2_m1_ha3" and manifest["valid"] and manifest["n_rows"] == N
    assert manifest["output_sha256"] == rf.sha(str(run_dir / TEAM_FILE))
    with open(release) as handle:
        rel = json.load(handle)
    assert manifest["ha3_model_sha256"] == rel["model_sha256"] and manifest["ha3_lam"] == rel["lam"]
    assert manifest["dead_model_sha256"] == rel["dead_model"]["model_sha256"]
    assert manifest["dead_model_kind"] == "league"
    verify = os.path.join(ROOT, "scripts", "verify_submission.py")
    ok = subprocess.run([sys.executable, verify, str(run_dir / "manifest.json"), "--rows", str(N)],
                        capture_output=True, text=True, cwd=ROOT)
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_two_runs_are_byte_identical(run_dir, release, atd_files, tmp_path):
    assert run(release, atd_files, tmp_path / "again").returncode == 0
    assert (tmp_path / "again" / TEAM_FILE).read_bytes() == (run_dir / TEAM_FILE).read_bytes()


def test_invalid_ha3_model_is_rejected_before_data_access(release, tmp_path):
    bad = rf.edit_release(release, tmp_path / "bad.json", model_sha256="0" * 64)
    meta, raw = str(tmp_path / "missing.xml"), str(tmp_path / "missing.txt")   # never read
    result = run(bad, (meta, raw), tmp_path / "out")
    assert result.returncode == 3 and "release check failed" in result.stderr
    assert not (tmp_path / "out").exists()


def test_output_inside_repository_and_bad_name_are_refused(release, atd_files, tmp_path):
    assert run(release, atd_files, os.path.join(ROOT, "run_ha3_out")).returncode == 2
    assert not os.path.exists(os.path.join(ROOT, "run_ha3_out"))
    assert run(release, atd_files, tmp_path / "o", "--output-name", "../x.csv").returncode == 2
