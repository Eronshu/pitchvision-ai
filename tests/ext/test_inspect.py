import json
import os
import subprocess
import sys

import pytest

from tests.ext import dfl_fixture as fx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSPECT = os.path.join(ROOT, "scripts", "inspect_idsse.py")


@pytest.fixture
def report(tmp_path):
    meta, positions = fx.standard_match(tmp_path)
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, INSPECT, "--meta", meta, "--positions", positions,
                             "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text())


def test_report_sections_and_timeline(report):
    first = report["sections"]["firstHalf"]
    assert first["frames"] == 100 and first["n_range"] == [10000, 10099]
    assert first["ball_gaps"] == 0 and first["ball_present_share"] == 1.0
    assert report["sections"]["secondHalf"]["n_range"] == [100000, 100099]


def test_report_frame_rate(report):
    # The fixture's T advances 40 ms per frame.
    assert report["sections"]["firstHalf"]["fps_from_timestamps"] == pytest.approx(25.0, rel=1e-3)
    assert report["sections"]["firstHalf"]["total_time_ms_meta"] == 4000


def test_report_labels(report):
    labels = report["sections"]["firstHalf"]["labels"]
    assert labels["ball_status"] == {"0": 10, "1": 90, "-1": 0}
    assert labels["possession"] == {"1": 60, "2": 40, "-1": 0}
    assert labels["dead_runs"]["count"] == 1
    assert labels["dead_runs"]["duration_s_p10_p50_p90"] == pytest.approx([0.4, 0.4, 0.4])
    # alive frames 0-39 (home), 50-59 (home), 60-99 (away): spells split by the dead run
    assert labels["alive_spells"]["count"] == 3
    assert labels["possession_during_dead"] == {"1": 10, "2": 0, "-1": 0}


def test_report_coordinates(report):
    coords = report["sections"]["firstHalf"]["coordinates"]
    assert coords["ball_x_min_max"] == pytest.approx([-20.0, 19.6], abs=1e-4)
    assert coords["kickoff_ball_xy"] == pytest.approx([-20.0, 5.0])
    assert coords["share_outside_centred_pitch"] == 0.0
    assert coords["share_outside_corner_pitch"] > 0.0      # negative x/y: not corner-origin


def test_report_people(report):
    people = report["sections"]["firstHalf"]["players_per_frame"]
    assert people["min_mean_max"] == [4, 4.0, 4]
    assert report["persons"] == {"home": 2, "away": 2}
    assert report["skipped_framesets"] == {"referee": 2}


def test_report_is_aggregates_only(report):
    def longest_list(node):
        if isinstance(node, dict):
            return max([longest_list(v) for v in node.values()] + [0])
        if isinstance(node, list):
            return max([len(node)] + [longest_list(v) for v in node])
        return 0
    assert longest_list(report) <= 3
    assert "attribution" in report and "CC BY 4.0" in report["attribution"]


# -- PR #2 review 3, fix 4: frame rate from the ball's own N endpoints, not the union timeline --

def test_frame_rate_uses_ball_endpoints_when_players_extend_beyond(tmp_path):
    """Ball: N 1010..1109 at 40 ms. Players: N 1000..1119 (beyond the ball on both sides).
    The union timeline has 120 frames; the ball spans 99 intervals over 3.96 s = 25 fps."""
    meta = fx.write_meta(str(tmp_path / "m.xml"), ["H1"], ["A1"])
    positions = fx.write_positions(str(tmp_path / "p.xml"), [
        ("firstHalf", fx.BALL, "B", fx.frames(1010, 100, 0.0, 0.0, Z=0, BallPossession=1,
                                                BallStatus=1)),
        ("firstHalf", fx.HOME, "H1", fx.frames(1000, 120, 1.0, 1.0)),
        ("firstHalf", fx.AWAY, "A1", fx.frames(1000, 120, 5.0, 5.0))])
    out = tmp_path / "r.json"
    result = subprocess.run([sys.executable, INSPECT, "--meta", meta, "--positions", positions,
                             "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    section = json.loads(out.read_text())["sections"]["firstHalf"]
    assert section["frames"] == 120
    assert section["ball_n_range"] == [1010, 1109]
    assert section["fps_from_timestamps"] == pytest.approx(25.0, rel=1e-6)
