import json
import os
import subprocess
import sys

import pytest

from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, scripted_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(ROOT, "scripts", "verify_submission.py")


@pytest.fixture
def run_dir(tmp_path):
    meta, raw = write_fixture(tmp_path, scripted_frames())
    out = tmp_path / "run"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"),
                    "--config", os.path.join(ROOT, "configs", "v1.json"), "--out", str(out),
                    "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
                    "--blocks", json.dumps(BLOCKS)], check=True, cwd=ROOT,
                   capture_output=True)
    return out


def verify(run_dir, *extra):
    return subprocess.run([sys.executable, VERIFY, str(run_dir / "manifest.json"),
                           "--rows", str(N)] + list(extra),
                          capture_output=True, text=True, cwd=ROOT)


def edit_manifest(run_dir, **changes):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def test_fresh_run_verifies(run_dir):
    result = verify(run_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_tampered_output_fails(run_dir):
    path = run_dir / "submission.csv"
    path.write_text(path.read_text()[:-2] + "D\n")   # last label X -> D: still a valid file
    result = verify(run_dir)
    assert result.returncode == 1 and "output_sha256" in result.stdout


def test_tampered_config_fails(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"]["alive"]["min_dead_s"] = 9.0
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    result = verify(run_dir)
    assert result.returncode == 1 and "config_sha256" in result.stdout


def test_wrong_row_count_fails(run_dir):
    result = subprocess.run([sys.executable, VERIFY, str(run_dir / "manifest.json")],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 1 and "199319" in result.stdout


@pytest.mark.parametrize("changes, fragment", [
    ({"git_dirty": True}, "git_dirty"),
    ({"git_commit": "0" * 40}, "git_commit"),
])
def test_strict_requires_clean_provenance(run_dir, changes, fragment):
    edit_manifest(run_dir, git_dirty=False)
    edit_manifest(run_dir, **changes)
    assert verify(run_dir).returncode == 0              # provenance is a warning by default
    result = verify(run_dir, "--strict")
    assert result.returncode == 1 and fragment in result.stdout


def test_invalid_flag_in_manifest_fails(run_dir):
    edit_manifest(run_dir, valid=False)
    result = verify(run_dir)
    assert result.returncode == 1 and "valid" in result.stdout


def test_unknown_provenance_is_not_clean(tmp_path):
    """Outside a git checkout the run cannot know its commit or whether the tree was clean: the
    manifest must say unknown, and --strict must refuse it rather than match None to None."""
    meta, raw = write_fixture(tmp_path, scripted_frames())
    out = tmp_path / "run"
    env = dict(os.environ, GIT_DIR=str(tmp_path / "no-such-repo"))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"),
                    "--config", os.path.join(ROOT, "configs", "v1.json"), "--out", str(out),
                    "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
                    "--blocks", json.dumps(BLOCKS)], check=True, cwd=ROOT, env=env,
                   capture_output=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["git_commit"] is None and manifest["git_dirty"] is None
    result = subprocess.run([sys.executable, VERIFY, str(out / "manifest.json"),
                             "--rows", str(N), "--strict"],
                            capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.returncode == 1
    assert "FAIL git_dirty" in result.stdout and "FAIL git_commit" in result.stdout
