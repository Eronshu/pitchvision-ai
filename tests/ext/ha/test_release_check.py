"""HA3 release integrity (scripts/verify_ha3_release.py) in a throwaway git checkout holding the
code, tracked configs/v2.json + models/m1_final.joblib and configs/ha3.json +
models/ha3_final.joblib. scripts/run_ha3.py is run FROM that checkout. Synthetic data only."""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from pvext.ha import release_check
from pvext.v2 import release_check as m1_check
from tests.ext.ha.test_deploy import build_ha3_release
from tests.ext.v2.test_release_check import CODE_DIRS, SKIP, git
from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, scripted_frames

ROOT = release_check.ROOT
CODE_FILES = ("games.py", "configs/v1.json", "scripts/run_v2.py", "scripts/run_ha3.py",
              "evaluation/possession_metrics.py", "evaluation/validate_submission.py")


def test_frozen_constants_are_the_current_prediction_code():
    for rel, frozen in release_check.FROZEN_HA3.items():
        with open(os.path.join(ROOT, rel), "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == frozen, rel


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return build_ha3_release(tmp_path_factory.mktemp("ha3rel"))


@pytest.fixture(scope="module")
def atd_files(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("atd"), scripted_frames())


def make_checkout(repo, ha3_config_path):
    for d in CODE_DIRS:
        shutil.copytree(os.path.join(ROOT, d), repo / d, ignore=SKIP)
    for f in CODE_FILES:
        os.makedirs(repo / os.path.dirname(f), exist_ok=True)
        shutil.copy(os.path.join(ROOT, f), repo / f)
    with open(ha3_config_path) as handle:
        ha3 = json.load(handle)
    with open(ha3["m1_release_config"]) as handle:
        m1 = json.load(handle)
    os.makedirs(repo / "models")
    shutil.copy(m1["model"], repo / "models" / "m1_final.joblib")
    shutil.copy(ha3["model"], repo / "models" / "ha3_final.joblib")
    shutil.copy(ha3["dead_model"]["model"], repo / "models" / "ha3_dead_m1_league.joblib")
    m1["model"] = "models/m1_final.joblib"
    ha3["model"] = "models/ha3_final.joblib"
    ha3["dead_model"]["model"] = "models/ha3_dead_m1_league.joblib"
    ha3["m1_release_config"] = "configs/v2.json"
    (repo / "configs" / "v2.json").write_text(json.dumps(m1, indent=2))
    (repo / "configs" / "ha3.json").write_text(json.dumps(ha3, indent=2))
    # like the real repository: running the scripts writes bytecode, which must not dirty the tree
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "release")


def generate(repo, atd_files, out):
    meta, raw = atd_files
    result = subprocess.run([sys.executable, str(repo / "scripts" / "run_ha3.py"),
                             "--config", str(repo / "configs" / "ha3.json"), "--out", str(out),
                             "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
                             "--blocks", json.dumps(BLOCKS)],
                            capture_output=True, text=True, cwd=repo)
    assert result.returncode == 0, result.stderr
    return str(out / "manifest.json")


@pytest.fixture
def released(synthetic, atd_files, tmp_path):
    repo = tmp_path / "repo"
    make_checkout(repo, synthetic)
    return repo, generate(repo, atd_files, tmp_path / "run")


def failed(manifest, repo):
    return {name for name, ok, _ in release_check.check(manifest, str(repo), N) if not ok}


def test_untouched_release_verifies(released):
    repo, manifest = released
    names = {name for name, _, _ in release_check.check(manifest, str(repo), N)}
    assert {"git_clean", "model", "sources", "output", "ha3_release_config", "ha3_model",
            "ha3_release_files", "ha3_dead_model", "ha3_frozen_code"} <= names
    assert failed(manifest, repo) == set()
    cli = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "verify_ha3_release.py"),
                          manifest, "--root", str(repo), "--rows", str(N)],
                         capture_output=True, text=True, cwd=ROOT)
    assert cli.returncode == 0, cli.stdout + cli.stderr


def test_verification_never_writes_into_the_output_directory(released):
    repo, manifest = released
    before = sorted(os.listdir(os.path.dirname(manifest)))
    release_check.check(manifest, str(repo), N)
    assert sorted(os.listdir(os.path.dirname(manifest))) == before


def test_tampered_ha3_model_fails(released):
    repo, manifest = released
    with open(repo / "models" / "ha3_final.joblib", "ab") as handle:
        handle.write(b"x")
    assert "ha3_model" in failed(manifest, repo)


def test_tampered_dead_model_fails(released):
    repo, manifest = released
    with open(repo / "models" / "ha3_dead_m1_league.joblib", "ab") as handle:
        handle.write(b"x")
    assert "ha3_dead_model" in failed(manifest, repo)


def test_untracked_dead_model_fails(released):
    repo, manifest = released
    git(repo, "rm", "-q", "--cached", "models/ha3_dead_m1_league.joblib")
    git(repo, "commit", "-qm", "untrack")
    assert "ha3_dead_model" in failed(manifest, repo)


def test_tampered_ha3_config_fails(released):
    repo, manifest = released
    cfg = json.loads((repo / "configs" / "ha3.json").read_text())
    cfg["lam"] = 0.25
    (repo / "configs" / "ha3.json").write_text(json.dumps(cfg, indent=2))
    assert "ha3_release_config" in failed(manifest, repo)


def test_changed_prediction_code_fails_even_if_committed(released):
    repo, manifest = released
    path = repo / "pvext" / "ha" / "features.py"
    path.write_text(path.read_text() + "\n# changed\n")
    git(repo, "commit", "-qam", "change")
    out = failed(manifest, repo)
    assert {"ha3_frozen_code", "git_head", "sources"} <= out


def test_untracked_ha3_model_fails(released):
    repo, manifest = released
    git(repo, "rm", "-q", "--cached", "models/ha3_final.joblib")
    git(repo, "commit", "-qm", "untrack")
    assert "ha3_release_files" in failed(manifest, repo)


def test_tampered_csv_fails(released):
    repo, manifest = released
    csv = os.path.join(os.path.dirname(manifest), "PitchVision AI.csv")
    with open(csv) as handle:
        text = handle.read()
    with open(csv, "w") as handle:
        handle.write(text.replace("\n599,X", "\n599,H"))
    assert "output" in failed(manifest, repo)


def test_modes_are_not_interchangeable(released, tmp_path):
    repo, manifest = released
    assert m1_check.check(manifest, str(repo), N)[0][:2] == ("mode", False)
    with open(manifest) as handle:
        data = json.load(handle)
    data["mode"] = "v2_m1"
    other = tmp_path / "m.json"
    other.write_text(json.dumps(data))
    assert release_check.check(str(other), str(repo), N)[0][:2] == ("mode", False)
