import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pitchvision import atd, diagnostics
from tests.synthetic import make_tracking
from tests.test_pipeline import BLOCKS, N, PITCH, scripted_frames
from tests.synthetic import write_fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25


def labels_of(text):
    return np.array(list(text), dtype="<U1")


HDA = labels_of("H" * 50 + "D" * 25 + "A" * 25)


def test_summarize_counts_and_shares():
    s = diagnostics.summarize(HDA, [(1, 0, 99)], FPS)["total"]
    assert s["n_spells"] == 2
    assert s["d_share_graded"] == pytest.approx(0.25)
    assert s["home_share"] == pytest.approx(50 / 75)
    assert s["share_spells_lt_1s"] == 0.0
    assert s["duration_p10_p50_p90_s"][1] == pytest.approx(1.5)
    assert s["transitions_per_min"] == pytest.approx(30.0)
    assert s["d_run_p50_s"] == pytest.approx(1.0)


def test_summarize_ignores_frames_outside_blocks():
    padded = np.concatenate([labels_of("A" * 30), HDA, labels_of("H" * 10)])
    s = diagnostics.summarize(padded, [(1, 30, 129)], FPS)["total"]
    assert s == diagnostics.summarize(HDA, [(1, 0, 99)], FPS)["total"]


def test_summarize_per_block_and_total():
    two = np.concatenate([HDA, labels_of("X" * 20), HDA])
    s = diagnostics.summarize(two, [(1, 0, 99), (2, 120, 219)], FPS)
    assert set(s) == {"half_1", "half_2", "total"}
    assert s["half_1"]["n_spells"] == 2 and s["total"]["n_spells"] == 4


def test_summarize_fills_abstentions_like_the_scorer():
    s = diagnostics.summarize(labels_of("H" * 40 + "X" * 20 + "H" * 40), [(1, 0, 99)], FPS)
    assert s["total"]["n_spells"] == 1


def test_summarize_no_spells():
    s = diagnostics.summarize(labels_of("D" * 100), [(1, 0, 99)], FPS)["total"]
    assert s["n_spells"] == 0 and s["duration_p10_p50_p90_s"] is None
    assert s["home_share"] is None


def test_agreement():
    same = diagnostics.agreement(HDA, HDA, [(1, 0, 99)])
    assert same == {"frame_agreement": 1.0, "spell_f1_mean": 1.0}
    flipped = labels_of("H" * 50 + "D" * 25 + "H" * 25)
    other = diagnostics.agreement(flipped, HDA, [(1, 0, 99)])
    assert other["frame_agreement"] == pytest.approx(0.75)
    assert other["spell_f1_mean"] < 1.0


def test_perturb_ball_deterministic_and_counted():
    ball = np.tile([50.0, 30.0], (200, 1))
    ball[::4] = np.nan                     # 150 present
    tr = make_tracking(np.full((200, 2, 2), 5.0), ball, [0, 1])
    a = diagnostics.perturb_ball(tr, 0.1, seed=0)
    b = diagnostics.perturb_ball(tr, 0.1, seed=0)
    assert np.array_equal(a.ball_present, b.ball_present)
    assert tr.ball_present.sum() - a.ball_present.sum() == 15
    assert np.isnan(a.ball_xy[~a.ball_present]).all()
    assert tr.ball_present.sum() == 150     # input untouched


def test_diagnose_cli_writes_aggregates_only(tmp_path):
    meta, raw = write_fixture(tmp_path, scripted_frames())
    run = [sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"),
           "--config", os.path.join(ROOT, "configs", "v1.json"),
           "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
           "--blocks", json.dumps(BLOCKS)]
    assert subprocess.run(run + ["--out", str(tmp_path / "v1")], cwd=ROOT).returncode == 0
    assert subprocess.run(run + ["--out", str(tmp_path / "fb"), "--fallback-baseline"],
                          cwd=ROOT).returncode == 0
    out = tmp_path / "diag.json"
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "diagnose.py"),
         "--config", os.path.join(ROOT, "configs", "v1.json"),
         "--submission", str(tmp_path / "v1" / "submission.csv"),
         "--baseline", str(tmp_path / "fb" / "baseline_smooth50.csv"), "--out", str(out),
         "--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
         "--blocks", json.dumps(BLOCKS)],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert {"submission", "baseline", "agreement_with_baseline", "stability",
            "config_sha256", "perturbation", "l2_setpieces"} <= set(report)
    assert report["l2_setpieces"] is None        # no tagging file next to the fixture
    assert report["stability"]["spell_f1_mean"] > 0.5

    def longest_list(node):
        if isinstance(node, dict):
            return max([longest_list(v) for v in node.values()] + [0])
        if isinstance(node, list):
            return max([len(node)] + [longest_list(v) for v in node])
        return 0
    assert longest_list(report) <= 3      # aggregates only: no per-frame arrays leave the run


def test_diagnose_cli_adds_l2_when_tagging_present(tmp_path):
    from tests.synthetic import write_smart_tagging
    meta, raw = write_fixture(tmp_path, scripted_frames())
    bundle = tmp_path / "bundle"
    write_smart_tagging(bundle / "tactical" / "BAR-ATH_tactical_smart_tagging.xml",
                        [("SET PIECES", 2.0, 4.0, {"Team": "Home FC"}),
                         ("SET PIECES", 15.0, 16.0, {"Team": "N/A"})])
    common = ["--atd-meta", meta, "--atd-raw", raw, "--n-frames", str(N),
              "--blocks", json.dumps(BLOCKS), "--data-dir", str(bundle)]
    cfg = tmp_path / "tagged.json"
    cfg.write_text(json.dumps({"tagging": {"use_setpiece_team": True}}))
    run = [sys.executable, os.path.join(ROOT, "scripts", "run_v1.py"), "--config", str(cfg)]
    assert subprocess.run(run + ["--out", str(tmp_path / "v1")] + common, cwd=ROOT).returncode == 0
    assert subprocess.run(run + ["--out", str(tmp_path / "fb"), "--fallback-baseline"] + common,
                          cwd=ROOT).returncode == 0
    out = tmp_path / "diag.json"
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "diagnose.py"), "--config", str(cfg),
         "--submission", str(tmp_path / "v1" / "submission.csv"),
         "--baseline", str(tmp_path / "fb" / "baseline_smooth50.csv"), "--out", str(out)]
        + common, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    l2 = json.loads(out.read_text())["l2_setpieces"]
    assert l2["submission"]["restart_team"]["independent"] is False
    assert l2["submission"]["restart_timing"]["independent"] is False
    assert l2["baseline"]["restart_team"]["independent"] is True
    assert l2["submission"]["n_setpiece_tags"] == 2
