import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.ext import dfl_fixture as fx

pytest.importorskip("kloppy")
from pvext import crosscheck, dfl_parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "crosscheck_kloppy_idsse.py")


def test_agreement_on_fixture(tmp_path):
    result = crosscheck.compare(*fx.standard_match(tmp_path))
    assert result["frames"] == {"kloppy": 200, "ours": 200, "kloppy_missing_in_ours": 0}
    # float32 storage vs kloppy's float64: tiny representation error only; the verdict uses the
    # same 1e-4 m tolerance. On the real J03WPY the maximum is 2e-6 m, for the same reason.
    assert result["ball_max_abs_diff_m"] < 1e-4
    assert result["status_mismatches"] == 0 and result["possession_mismatches"] == 0
    assert result["player_observations"]["kloppy"] == result["player_observations"]["ours"]
    assert result["player_max_abs_diff_m"] < 1e-4
    assert result["kloppy_only_alive_default_frames"] == 180     # the default drops dead frames
    assert result["unknown_labels"] == 0 and result["non_finite_coordinates"] == 0
    assert result["agrees"] is True and type(result["agrees"]) is bool


@pytest.mark.parametrize("corrupt, key", [
    (lambda t, l: t.data[0].ball_xy.__setitem__((7, 0), t.data[0].ball_xy[7, 0] + 0.5),
     "ball_max_abs_diff_m"),
    (lambda t, l: l.data[0].ball_status.__setitem__(3, 1 - l.data[0].ball_status[3]),
     "status_mismatches"),
    (lambda t, l: l.data[1].possession.__setitem__(5, 3 - l.data[1].possession[5]),
     "possession_mismatches"),
    (lambda t, l: t.data[0].players_xy.__setitem__((9, 1, 1), t.data[0].players_xy[9, 1, 1] + 2),
     "player_max_abs_diff_m"),
])
def test_disagreement_is_detected(tmp_path, monkeypatch, corrupt, key):
    """The check must be able to fail: corrupt one value of our parse and it has to show."""
    real_parse = dfl_parse.parse

    def corrupted(meta, positions):
        track, labels, meta_out = real_parse(meta, positions)
        corrupt(track, labels)
        return track, labels, meta_out
    monkeypatch.setattr(crosscheck.dfl_parse, "parse", corrupted)
    result = crosscheck.compare(*fx.standard_match(tmp_path))
    assert result[key] > 0 and result["agrees"] is False


# -- PR #2 review 3, fix 3: unknown labels and non-finite values can never read as agreement ---

def _nan_ball(t, l):
    t.data[0].ball_xy[12] = np.nan


def _nan_player(t, l):
    t.data[1].players_xy[30, 2] = np.nan


@pytest.mark.parametrize("corrupt, key", [
    # kloppy maps a missing BallStatus to dead; ours says unknown (-1). Not an agreement.
    (lambda t, l: l.data[0].ball_status.__setitem__(45, -1), "unknown_labels"),
    # kloppy maps a missing BallPossession to away; ours says unknown (-1). Not an agreement.
    (lambda t, l: l.data[0].possession.__setitem__(70, -1), "unknown_labels"),
    (_nan_ball, "non_finite_coordinates"),
    (_nan_player, "player_observations"),
])
def test_unknown_or_non_finite_never_agree(tmp_path, monkeypatch, corrupt, key):
    real_parse = dfl_parse.parse

    def corrupted(meta, positions):
        track, labels, meta_out = real_parse(meta, positions)
        corrupt(track, labels)
        return track, labels, meta_out
    monkeypatch.setattr(crosscheck.dfl_parse, "parse", corrupted)
    result = crosscheck.compare(*fx.standard_match(tmp_path))
    value = result[key]
    assert (value["kloppy_missing_in_ours"] if isinstance(value, dict) else value) > 0
    assert result["agrees"] is False


def test_non_finite_kloppy_values_never_agree(tmp_path, monkeypatch):
    """A NaN on kloppy's side must be counted too, not silently dropped by max()."""
    from kloppy import sportec
    real_load = sportec.load_tracking

    def nan_ball(*args, **kwargs):
        dataset = real_load(*args, **kwargs)
        if kwargs.get("only_alive") is False:
            record = dataset.records[5]
            record.ball_coordinates = dataclasses.replace(record.ball_coordinates, x=float("nan"))
        return dataset
    monkeypatch.setattr(sportec, "load_tracking", nan_ball)
    result = crosscheck.compare(*fx.standard_match(tmp_path))
    assert result["non_finite_coordinates"] == 1 and result["agrees"] is False


def test_cli_writes_aggregates(tmp_path):
    meta, positions = fx.standard_match(tmp_path)
    out = tmp_path / "x.json"
    proc = subprocess.run([sys.executable, SCRIPT, "--meta", meta, "--positions", positions,
                           "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["agrees"] is True and "CC BY 4.0" in report["attribution"]
