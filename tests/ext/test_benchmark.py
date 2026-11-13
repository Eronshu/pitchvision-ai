import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pitchvision import atd
from pvext import benchmark
from starter import baseline
from tests.ext import dfl_fixture as fx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "run_benchmark.py")


def labels_of(text):
    return np.array(list(text), dtype="<U1")


# -- boundary scoring -----------------------------------------------------------------------------

def test_boundary_scores_exact_and_shifted():
    target = labels_of("H" * 100 + "D" * 50 + "A" * 100)
    blocks = [(1, 0, 249)]
    exact = benchmark.boundary_scores(target, target, blocks, tolerances=(12,))
    assert exact["dead_start"][12]["f1"] == 1.0 and exact["dead_end"][12]["f1"] == 1.0
    late = labels_of("H" * 110 + "D" * 50 + "A" * 90)          # both boundaries 10 frames late
    shifted = benchmark.boundary_scores(target, late, blocks, tolerances=(5, 12))
    assert shifted["dead_start"][5]["f1"] == 0.0 and shifted["dead_start"][12]["f1"] == 1.0
    assert shifted["dead_start"][12]["median_abs_offset_s"] == pytest.approx(0.4)


def test_boundary_scores_missed_dead_run():
    target = labels_of("H" * 100 + "D" * 50 + "A" * 100)
    nothing = labels_of("H" * 250)
    result = benchmark.boundary_scores(target, nothing, [(1, 0, 249)], tolerances=(25,))
    assert result["dead_start"][25] == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                        "n_true": 1, "n_pred": 0, "median_abs_offset_s": None}


def test_score_reports_primary_and_secondary_metrics():
    target = labels_of("H" * 100 + "D" * 50 + "A" * 100)
    s = benchmark.score(target, target, [(1, 0, 249)])
    assert s["alive_dead_f1"] == 1.0 and s["d_f1"] == 1.0
    assert s["d_precision"] == 1.0 and s["d_recall"] == 1.0
    assert s["spell_f1_mean"] == 1.0 and "boundary" in s


# -- the benchmark run on a synthetic DFL match -----------------------------------------------------

@pytest.fixture
def match(tmp_path):
    return fx.standard_match(tmp_path / "src")


def test_run_rows_per_method_preset_seed(match, tmp_path):
    rows = benchmark.run(*match, presets=["clean", "doc"], seeds=[0, 1],
                         work_dir=str(tmp_path / "work"))
    keys = {(r["method"], r["preset"], r["seed"]) for r in rows}
    assert keys == {(m, "clean", 0) for m in ("v1", "fallback")} | {
        (m, "doc", s) for m in ("v1", "fallback") for s in (0, 1)}
    for r in rows:
        assert r["n_graded"] == 200 and len(r["input_sha256"]) == 64


def test_identical_inputs_for_both_methods(match, tmp_path, monkeypatch):
    seen = {"v1": [], "fallback": []}
    real_load, real_nearest = atd.load_arrays, baseline.nearest_player_labels

    def sha(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def spy_load(meta_path, raw_path, *args):
        seen["v1"].append((raw_path, sha(raw_path), sha(meta_path)))
        return real_load(meta_path, raw_path, *args)

    def spy_nearest(raw_path, *args):
        seen["fallback"].append((raw_path, sha(raw_path)))
        return real_nearest(raw_path, *args)
    monkeypatch.setattr(benchmark.atd, "load_arrays", spy_load)
    monkeypatch.setattr(benchmark.baseline, "nearest_player_labels", spy_nearest)
    rows = benchmark.run(*match, presets=["doc"], seeds=[0], work_dir=str(tmp_path / "w"))
    (v1_path, v1_sha, _), = seen["v1"]
    (fb_path, fb_sha), = seen["fallback"]
    assert v1_path == fb_path and v1_sha == fb_sha
    assert {r["input_sha256"] for r in rows} == {v1_sha}


@pytest.mark.parametrize("transform", [
    lambda t: np.roll(t, 37),                                          # shifted in time
    lambda t: np.where(t == "H", "A", np.where(t == "A", "H", t)),     # flipped sides
    lambda t: np.where(t == "X", "X", "D"),                            # everything dead
])
def test_predictions_do_not_depend_on_labels(match, tmp_path, transform):
    """Label leakage guard: changing the targets changes scores, never predictions."""
    plain = benchmark.run(*match, presets=["clean", "doc"], seeds=[0],
                          work_dir=str(tmp_path / "a"))
    altered = benchmark.run(*match, presets=["clean", "doc"], seeds=[0],
                            work_dir=str(tmp_path / "b"), target_transform=transform)
    for a, b in zip(plain, altered):
        assert (a["method"], a["preset"]) == (b["method"], b["preset"])
        assert a["prediction_sha256"] == b["prediction_sha256"]
        assert a["input_sha256"] == b["input_sha256"]
    def scores(row):
        flat = benchmark.flatten(row)
        return {k: v for k, v in flat.items() if not k.endswith("sha256")}
    assert any(scores(a) != scores(b) for a, b in zip(plain, altered))   # the targets did change


def test_work_files_are_removed(match, tmp_path):
    work = tmp_path / "work"
    benchmark.run(*match, presets=["doc"], seeds=[0], work_dir=str(work))
    assert not work.exists() or os.listdir(work) == []


def test_summary_groups_by_preset_and_method(match, tmp_path):
    rows = benchmark.run(*match, presets=["clean", "doc"], seeds=[0, 1],
                         work_dir=str(tmp_path / "w"))
    summary = benchmark.summarize(rows)
    assert set(summary) == {"clean", "doc"}
    doc_v1 = summary["doc"]["v1"]
    assert doc_v1["n_seeds"] == 2 and set(doc_v1["d_f1"]) == {"mean", "min", "max"}


def test_cli_writes_reports_with_attribution(match, tmp_path):
    meta, positions = match
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, SCRIPT, "--meta", meta, "--positions", positions,
                             "--presets", "clean,mild", "--seeds", "0", "--work",
                             str(tmp_path / "w"), "--out", str(out)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert "CC BY 4.0" in summary["attribution"] and summary["exploratory"] is True
    assert (out / "per_run.csv").exists()
