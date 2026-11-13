import numpy as np
import pytest

from pitchvision import atd
from starter import baseline
from tests.synthetic import raw_line, write_fixture, write_meta

PITCH = (105.3, 68.0)


def _frames(balls, n_slots=2):
    return [{"players": [(0.2 + 0.1 * s, 0.5, 1.0) for s in range(n_slots)], "ball": ball}
            for ball in balls]


def test_alignment(tmp_path):
    meta, raw = write_fixture(tmp_path, _frames([(.5, .5), None, (.1, .1)]))
    tr = atd.load_arrays(meta, raw, 3, PITCH)
    assert tr.ball_present.tolist() == [True, False, True]
    np.testing.assert_allclose(tr.ball_xy[0], (52.65, 34.0), rtol=1e-6)
    np.testing.assert_allclose(tr.ball_xy[2], (10.53, 6.8), rtol=1e-6)
    assert np.isnan(tr.ball_xy[1]).all()


def test_metres_and_shapes(tmp_path):
    meta, raw = write_fixture(tmp_path, _frames([(.5, .5)], n_slots=4))
    tr = atd.load_arrays(meta, raw, 1, PITCH)
    assert tr.players_xy.shape == (1, 4, 2) and tr.players_xy.dtype == np.float32
    assert tr.players_speed.shape == (1, 4)
    assert tr.team_of_slot.tolist() == [0, 1, 0, 1]
    np.testing.assert_allclose(tr.players_xy[0, 0], (0.2 * 105.3, 34.0), rtol=1e-6)


def test_incomplete_file_rejected(tmp_path):
    meta, raw = write_fixture(tmp_path, _frames([(.5, .5)]))
    with pytest.raises(ValueError, match="1 records, expected 4"):
        atd.load_arrays(meta, raw, 4, PITCH)


def _write_ids(tmp_path, frame_ids):
    meta = write_meta(str(tmp_path / "atd.xml"), slot_teams=["T1", "T2"])
    raw = str(tmp_path / "atd.txt")
    with open(raw, "w") as handle:
        for frame_id in frame_ids:
            handle.write(raw_line(frame_id, [(.5, .5, 0)] * 2, (.5, .5)))
    return meta, raw


@pytest.mark.parametrize("frame_ids, n_frames, fragment", [
    ([1, 2, 2, 3], 4, "duplicate or out-of-order frame_id 2 at line 3"),
    ([1, 3, 2, 4], 4, "missing frame_id 2"),
    ([1, 2, 5, 6], 6, "missing frame_id 3..4"),
    ([2, 3, 4], 3, "missing frame_id 1"),
    ([1, 2, 3], 4, "3 records, expected 4"),
    ([1, 2, 3, 4, 5], 4, "frame_id 5 outside 1..4"),
])
def test_frame_sequence_integrity(tmp_path, frame_ids, n_frames, fragment):
    meta, raw = _write_ids(tmp_path, frame_ids)
    with pytest.raises(ValueError, match=fragment):
        atd.load_arrays(meta, raw, n_frames, PITCH)


def test_complete_sequence_accepted(tmp_path):
    meta, raw = _write_ids(tmp_path, [1, 2, 3, 4])
    assert atd.load_arrays(meta, raw, 4, PITCH).ball_present.all()


def test_equivalence_with_baseline(tmp_path):
    rng = np.random.RandomState(0)
    frames = []
    for _ in range(50):
        players = [None if rng.rand() < 0.3 else (rng.rand(), rng.rand(), rng.rand() * 5)
                   for _ in range(6)]
        ball = None if rng.rand() < 0.3 else (rng.rand(), rng.rand())
        frames.append({"players": players, "ball": ball})
    meta, raw = write_fixture(tmp_path, frames)

    tr = atd.load_arrays(meta, raw, 50, PITCH)
    expected = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta), 50, PITCH)
    assert atd.nearest_team(tr).tolist() == expected.tolist()


def test_slot_count_mismatch(tmp_path):
    meta = write_meta(str(tmp_path / "atd.xml"), slot_teams=["T1", "T2"] * 3)
    raw = str(tmp_path / "atd.txt")
    with open(raw, "w") as handle:
        handle.write(raw_line(1, [(.5, .5, 0)] * 5, (.5, .5)))
    with pytest.raises(ValueError, match="atd.txt"):
        atd.load_arrays(meta, raw, 1, PITCH)


def test_third_team_id(tmp_path):
    meta, raw = write_fixture(tmp_path, _frames([(.5, .5)], n_slots=3),
                              slot_teams=["T1", "T2", "REF"])
    with pytest.raises(ValueError, match="REF"):
        atd.load_arrays(meta, raw, 1, PITCH)


@pytest.mark.parametrize("bad_line, fragment", [
    ("1:0.5,0.5,0;;0.5,0.5,0:0.5,0.5,0\n", "line 1"),
    ("1:0.5,0.5,0;0.5,0.5,0\n", "line 1"),
])
def test_malformed_line(tmp_path, bad_line, fragment):
    meta = write_meta(str(tmp_path / "atd.xml"), slot_teams=["T1", "T2"])
    raw = str(tmp_path / "atd.txt")
    with open(raw, "w") as handle:
        handle.write(bad_line)
    with pytest.raises(ValueError, match=fragment):
        atd.load_arrays(meta, raw, 1, PITCH)
