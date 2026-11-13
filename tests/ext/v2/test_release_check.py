"""V2 release integrity (scripts/verify_v2_release.py): tampering after generation must fail, the
release artifacts must be git-tracked inside the repository, and the official row count applies.

A throwaway git repository stands in for a clean clone of the release: the code needed to run
scripts/run_v2.py, configs/v1.json, a tracked configs/v2.json whose model entry is the relative
models/m1_final.joblib, and that tracked model. run_v2.py is run FROM that checkout, so it records
the checkout's own commit and clean state. Synthetic data only.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from pvext.v2 import release_check
from tests.ext.v2 import release_fixture as rf
from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, scripted_frames

ROOT = release_check.ROOT
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
       "-c", "commit.gpgsign=false"]
CODE_DIRS = ("pitchvision", "pvext", "starter")
CODE_FILES = ("games.py", "configs/v1.json", "scripts/run_v2.py", "evaluation/possession_metrics.py",
              "evaluation/validate_submission.py")
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb")


def git(repo, *args):
    return subprocess.run(GIT + list(args), cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture(scope="module")
def synthetic_release(tmp_path_factory):
    return rf.build_release(tmp_path_factory.mktemp("release"))[0]


@pytest.fixture(scope="module")
def atd_files(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("atd"), scripted_frames())


def make_checkout(repo, synthetic_release, model_entry="models/m1_final.joblib"):
    for d in CODE_DIRS:
        shutil.copytree(os.path.join(ROOT, d), repo / d, ignore=SKIP)
    for f in CODE_FILES:
        os.makedirs(repo / os.path.dirname(f), exist_ok=True)
        shutil.copy(os.path.join(ROOT, f), repo / f)
    with open(synthetic_release) as handle:
        rel = json.load(handle)
    os.makedirs(repo / "models")
    shutil.copy(rel["model"], repo / "models" / "m1_final.joblib")
    rel["model"] = model_entry
    (repo / "configs" / "v2.json").write_text(json.dumps(rel, indent=2))
    # like the real repository: running the scripts writes bytecode, which must not dirty the tree
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "release")
    return rel


def generate(repo, atd_files, out, config="configs/v2.json"):
    meta, raw = atd_files
    result = subprocess.run([sys.executable, str(repo / "scripts" / "run_v2.py"),
                             "--config", str(repo / config) if not os.path.isabs(config) else config,
                             "--out", str(out), "--atd-meta", meta, "--atd-raw", raw,
                             "--n-frames", str(N), "--blocks", json.dumps(BLOCKS)],
                            capture_output=True, text=True, cwd=repo)
    assert result.returncode == 0, result.stderr
    return str(out / "manifest.json")


@pytest.fixture
def released(synthetic_release, atd_files, tmp_path):
    """(repo, manifest_path): a clean, committed release checkout and a run generated from it."""
    repo = tmp_path / "repo"
    make_checkout(repo, synthetic_release)
    return repo, generate(repo, atd_files, tmp_path / "run")


def failed(manifest_path, repo, rows=N):
    return {name for name, ok, _ in release_check.check(manifest_path, str(repo), rows) if not ok}


def verify_cli(manifest, repo, *extra):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "verify_v2_release.py"),
                           manifest, "--root", str(repo)] + list(extra),
                          capture_output=True, text=True, cwd=ROOT)


# -- untouched release, and tampering after generation ---------------------------------------------

def test_untouched_release_verifies(released):
    repo, manifest = released
    assert failed(manifest, repo) == set()
    cli = verify_cli(manifest, repo, "--rows", str(N))
    assert cli.returncode == 0, cli.stdout


def test_tampered_model_fails(released):
    repo, manifest = released
    with open(repo / "models" / "m1_final.joblib", "ab") as handle:
        handle.write(b"\0")
    assert "model" in failed(manifest, repo)


def test_tampered_release_config_fails(released):
    repo, manifest = released
    path = repo / "configs" / "v2.json"
    rel = json.loads(path.read_text())
    rel["threshold"] = 0.7 if rel["threshold"] != 0.7 else 0.3
    path.write_text(json.dumps(rel, indent=2))
    assert "release_config" in failed(manifest, repo)


def test_reformatted_release_config_fails(released):
    repo, manifest = released
    path = repo / "configs" / "v2.json"
    path.write_text(json.dumps(json.loads(path.read_text())))      # same content, other bytes
    assert "release_config" in failed(manifest, repo)


def test_tampered_source_fails(released):
    repo, manifest = released
    with open(repo / "pvext" / "v2" / "deploy.py", "a") as handle:
        handle.write("\n# edited after the run\n")
    assert "sources" in failed(manifest, repo)


def test_dirty_working_tree_fails(released):
    repo, manifest = released
    (repo / "untracked.txt").write_text("x")
    assert failed(manifest, repo) == {"git_clean"}


def test_moved_head_fails(released):
    repo, manifest = released
    git(repo, "commit", "-q", "--allow-empty", "-m", "later")
    assert failed(manifest, repo) == {"git_head"}


def test_run_recorded_as_dirty_fails(released):
    repo, manifest = released
    with open(manifest) as handle:
        data = json.load(handle)
    data["git_dirty"] = True
    with open(manifest, "w") as handle:
        json.dump(data, handle)
    assert "git_head" in failed(manifest, repo)


def test_tampered_csv_fails(released):
    repo, manifest = released
    csv = os.path.join(os.path.dirname(manifest), "PitchVision AI.csv")
    with open(csv) as handle:
        text = handle.read()
    with open(csv, "w") as handle:
        handle.write(text.replace("\n599,X", "\n599,H"))
    assert failed(manifest, repo) == {"output"}


def test_cli_exits_non_zero_on_failure(released):
    repo, manifest = released
    (repo / "untracked.txt").write_text("x")
    cli = verify_cli(manifest, repo, "--rows", str(N))
    assert cli.returncode == 1 and "FAIL git_clean" in cli.stdout


# -- release artifacts must be tracked inside the repository (finding 1) ---------------------------

def test_external_absolute_model_path_fails(synthetic_release, atd_files, tmp_path):
    repo = tmp_path / "repo"
    with open(synthetic_release) as handle:
        external = json.load(handle)["model"]                     # outside the checkout
    make_checkout(repo, synthetic_release, model_entry=external)
    manifest = generate(repo, atd_files, tmp_path / "run")       # development loading still works
    assert "release_files" in failed(manifest, repo)


def test_absolute_path_into_the_repository_is_not_a_release_layout(synthetic_release, atd_files,
                                                                   tmp_path):
    repo = tmp_path / "repo"
    make_checkout(repo, synthetic_release, model_entry=str(tmp_path / "repo" / "models" /
                                                           "m1_final.joblib"))
    manifest = generate(repo, atd_files, tmp_path / "run")
    assert "release_files" in failed(manifest, repo)


@pytest.mark.parametrize("target", ["models/m1_final.joblib", "configs/v2.json"])
def test_untracked_release_artifact_fails(released, target):
    repo, manifest = released
    git(repo, "rm", "-q", "--cached", target)
    git(repo, "commit", "-q", "-m", "untrack " + target)
    results = {n: ok for n, ok, _ in release_check.check(manifest, str(repo), N)}
    assert results["release_files"] is False


def test_model_symlink_escaping_the_repository_fails(released, tmp_path):
    repo, manifest = released
    outside = tmp_path / "outside.joblib"
    shutil.copy(repo / "models" / "m1_final.joblib", outside)    # same bytes, same sha256
    os.remove(repo / "models" / "m1_final.joblib")
    os.symlink(outside, repo / "models" / "m1_final.joblib")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "symlinked model")
    results = {n: ok for n, ok, _ in release_check.check(manifest, str(repo), N)}
    assert results["model"] is True                               # the hash alone would pass
    assert results["release_files"] is False


def test_config_symlink_escaping_the_repository_fails(released, tmp_path):
    repo, manifest = released
    outside = tmp_path / "outside_v2.json"
    shutil.copy(repo / "configs" / "v2.json", outside)
    os.remove(repo / "configs" / "v2.json")
    os.symlink(outside, repo / "configs" / "v2.json")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "symlinked config")
    assert "release_files" in failed(manifest, repo)


def test_release_config_outside_the_repository_fails(synthetic_release, atd_files, tmp_path):
    repo = tmp_path / "repo"
    rel = make_checkout(repo, synthetic_release)
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps(dict(rel, model=str(repo / "models" / "m1_final.joblib"))))
    manifest = generate(repo, atd_files, tmp_path / "run", config=str(outside))
    assert "release_files" in failed(manifest, repo)


# -- official row count (finding 2) ------------------------------------------------------------------

def test_default_row_count_is_the_official_tactical_frame_count():
    assert release_check.official_rows() == 199319


def test_without_an_override_the_official_count_is_enforced(released):
    repo, manifest = released
    assert "output" in failed(manifest, repo, rows=None)          # a 600-row synthetic file fails
    cli = verify_cli(manifest, repo)                              # the normal command
    assert cli.returncode == 1 and "FAIL output" in cli.stdout and "199319 rows" in cli.stdout


def test_manifest_row_count_cannot_lower_the_requirement(released):
    repo, manifest = released
    with open(manifest) as handle:
        data = json.load(handle)
    assert data["n_rows"] == N
    assert "output" in failed(manifest, repo, rows=None)          # manifest's 600 is not used


def test_explicit_override_must_match_the_manifest(released):
    repo, manifest = released
    assert "output" in failed(manifest, repo, rows=N + 1)
