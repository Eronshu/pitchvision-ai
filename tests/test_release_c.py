"""Release C checks on synthetic ATD only (no match data, no ground truth)."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tests.synthetic import write_fixture
from tests.test_pipeline import BLOCKS, N, scripted_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "scripts", "run_astra5_combo.py")
CONFIG = os.path.join(ROOT, "candidate", "combo.json")
CSV = "PitchVision AI.csv"
MODELS = ("combo_dead_hgb.joblib", "combo_ha5a.joblib")
FROZEN_MODELS = {"models/combo_dead_hgb.joblib": "5acbfa493de2443f83af0713d815f3927513370ff5ada6aaeb06c4e125c75139",
                 "models/combo_ha5a.joblib": "956ed0081b1bfabd54e20b604125ac87f1b86f4c408ce995ad84cd2c1f1332e8"}
PINNED_SOURCE = "astra5/s2_joint/decode.py"


def sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(out, atd, config=CONFIG, script=RUN, cwd=ROOT, env=None, hidden=True):
    cmd = [sys.executable, script, "--config", config, "--out", str(out)]
    if hidden:
        cmd += ["--atd-meta", atd[0], "--atd-raw", atd[1], "--n-frames", str(N), "--blocks", json.dumps(BLOCKS)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


@pytest.fixture(scope="module")
def atd(tmp_path_factory):
    d = tmp_path_factory.mktemp("atd")
    return write_fixture(d, scripted_frames())


@pytest.fixture(scope="module")
def first(atd, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    r = run(out, atd)
    assert r.returncode == 0, r.stderr
    return out


def test_end_to_end_format_and_official_validator(first):
    lines = open(first / CSV).read().split("\n")
    assert lines[0] == "frame,label"
    labels = np.array([x.split(",")[1] for x in lines[1:] if x])
    graded = np.zeros(N, bool)
    for _, a, b in BLOCKS:
        graded[a:b + 1] = True
    assert len(labels) == N and set(labels[~graded]) == {"X"} and set(labels[graded]) <= {"H", "A", "D"}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), str(first / CSV), "--rows", str(N)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.load(open(first / "manifest.json"))
    assert m["valid"] and "release C" in m["status"] and m["output_sha256"] == sha(first / CSV)
    assert m["lock_sha256"] == sha(os.path.join(ROOT, "candidate", "RELEASE.lock")) and "label_counts" not in m
    assert sorted(os.listdir(first)) == [CSV, "manifest.json"]


def test_repeated_runs_are_byte_identical(first, atd, tmp_path):
    for i in range(2):
        assert run(tmp_path / str(i), atd).returncode == 0
        assert sha(tmp_path / str(i) / CSV) == sha(first / CSV)


def test_equivalent_to_the_frozen_candidate_implementation(first):
    expected = open(os.path.join(ROOT, "tests", "expected_synthetic_sha256.txt")).read().split()[0]
    assert sha(first / CSV) == expected          # recorded from the research candidate implementation (astra5-combo-v1) before the release changes


def test_models_are_the_frozen_research_candidate_models_and_match_the_lock():
    lock = dict(line.split(None, 1)[::-1] for line in open(os.path.join(ROOT, "candidate", "RELEASE.lock")).read().splitlines())
    lock = {k.strip(): v for k, v in lock.items()}
    assert set(lock) == set(FROZEN_MODELS) | {"combo.json"}
    for rel, digest in lock.items():
        assert sha(os.path.join(ROOT, "candidate", rel)) == digest
        assert FROZEN_MODELS.get(rel, digest) == digest
    joint = json.load(open(CONFIG))["joint"]
    assert (joint["lam_ha"], joint["lam_on"], joint["lam_off"]) == (0.5, 0.25, 4.0)


def _package(tmp_path):
    pkg = tmp_path / "candidate"
    shutil.copytree(os.path.join(ROOT, "candidate"), pkg)
    return pkg


NO_DATA = ("/nonexistent/a.xml", "/nonexistent/a.txt")


@pytest.mark.parametrize("name", MODELS)
def test_each_modified_model_is_refused_before_any_data(tmp_path, name):
    pkg = _package(tmp_path)
    with open(pkg / "models" / name, "ab") as handle:
        handle.write(b"x")
    r = run(tmp_path / "o", NO_DATA, config=str(pkg / "combo.json"))
    assert r.returncode == 3 and "sha256 differs from RELEASE.lock" in r.stderr and not (tmp_path / "o").exists()


@pytest.mark.parametrize("key,value", [("lam_off", 2.0), ("lam_on", 0.5), ("lam_ha", 1.0)])
def test_edited_decoding_cost_is_refused_before_any_data(tmp_path, key, value):
    pkg = _package(tmp_path)
    cfg = json.load(open(pkg / "combo.json"))
    cfg["joint"][key] = value
    json.dump(cfg, open(pkg / "combo.json", "w"), indent=2, sort_keys=True)
    r = run(tmp_path / "o", NO_DATA, config=str(pkg / "combo.json"))
    assert r.returncode == 3 and "combo.json sha256 differs from RELEASE.lock" in r.stderr and not (tmp_path / "o").exists()


def test_missing_lock_is_refused(tmp_path):
    pkg = _package(tmp_path)
    os.remove(pkg / "RELEASE.lock")
    r = run(tmp_path / "o", NO_DATA, config=str(pkg / "combo.json"))
    assert r.returncode == 3 and "RELEASE.lock" in r.stderr


def _relock(pkg, cfg):
    """Rewrite config AND lock consistently, to reach the checks behind the lock."""
    open(pkg / "combo.json", "w").write(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    lines = open(pkg / "RELEASE.lock").read().splitlines()
    open(pkg / "RELEASE.lock", "w").write("\n".join(
        "{}  combo.json".format(sha(pkg / "combo.json")) if l.endswith("combo.json") else l for l in lines) + "\n")


@pytest.mark.parametrize("field,value,message", [
    (("dependencies", "pandas"), "0.0.1", "dependency pandas"),
    (("dependencies", "numpy"), "0.0.1", "dependency numpy"),
    (("python",), "3.12", "Python is"),
    (("fps",), 30, "25 fps"),
    (("sources", "games.py"), "0" * 64, "prediction code differs"),
    (("sources", "scripts/run_astra5_combo.py"), "0" * 64, "prediction code differs"),
])
def test_version_and_source_pins_are_enforced_even_with_a_consistent_lock(tmp_path, field, value, message):
    pkg = _package(tmp_path)
    cfg = json.load(open(pkg / "combo.json"))
    node = cfg
    for k in field[:-1]:
        node = node[k]
    node[field[-1]] = value
    _relock(pkg, cfg)
    r = run(tmp_path / "o", NO_DATA, config=str(pkg / "combo.json"))
    assert r.returncode == 3 and message in r.stderr, r.stderr
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("source", ["astra5/s2_joint/decode.py", "games.py", "scripts/run_astra5_combo.py",
                                    "pvext/ha5/features.py", "astra5/release/deploy.py"])
def test_modified_source_file_is_refused(tmp_path, atd, source):
    tree = tmp_path / "tree"
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git", ".venv", "gt", "results", "data"))
    with open(tree / source, "a") as handle:
        handle.write("\n# changed\n")
    r = run(tmp_path / "o", atd, config=str(tree / "candidate" / "combo.json"), script=str(tree / "scripts" / "run_astra5_combo.py"), cwd=str(tree))
    assert r.returncode == 3 and "prediction code differs" in r.stderr and source in r.stderr and not (tmp_path / "o").exists()


def test_the_entry_loads_no_other_model_and_reads_no_ground_truth():
    for rel in ("scripts/run_astra5_combo.py", "astra5/release/deploy.py"):
        text = open(os.path.join(ROOT, rel)).read()
        assert "ha3_dead_m1_league" not in text and "possession_gt" not in text and "evaluation/gt" not in text


def test_output_inside_the_draft_is_refused(atd):
    r = run(os.path.join(ROOT, "out_tmp"), atd)
    assert r.returncode == 2 and not os.path.exists(os.path.join(ROOT, "out_tmp"))


def test_without_match_data_the_real_entry_fails_cleanly(tmp_path):
    env = dict(os.environ, MTAH_DATA_DIR=str(tmp_path / "no_data"))
    r = run(tmp_path / "o", None, env=env, hidden=False)
    assert r.returncode != 0 and "match data not found" in r.stderr and not (tmp_path / "o").exists()
