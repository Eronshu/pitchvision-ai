import inspect

import numpy as np
import pytest

from pitchvision import atd
from pvext import convert, dfl_parse
from starter import baseline
from tests.ext import dfl_fixture as fx

PITCH = (105.0, 68.0)


@pytest.fixture
def parsed(tmp_path):
    return dfl_parse.parse(*fx.standard_match(tmp_path / "src"))


@pytest.fixture
def laid_out(parsed):
    track, labels, meta = parsed
    frames, target, blocks = convert.layout(track, labels, meta, gap_frames=50)
    return frames, target, blocks, meta


def test_layout_axis_blocks_and_gap(laid_out):
    frames, target, blocks, _ = laid_out
    assert len(frames.ball_xy) == 100 + 50 + 100 == len(target)
    assert blocks == [(1, 0, 99), (2, 150, 249)]
    assert set(target[100:150]) == {"X"}
    assert np.isnan(frames.ball_xy[100:150]).all() and np.isnan(frames.players_xy[100:150]).all()


def test_corner_origin_metres(laid_out):
    frames, _, _, _ = laid_out
    # fixture ball at x = -20.0 + 0.4 i, y = 5.0 (centre origin) -> + (52.5, 34)
    np.testing.assert_allclose(frames.ball_xy[0], [32.5, 39.0])
    np.testing.assert_allclose(frames.ball_xy[10], [36.5, 39.0])
    np.testing.assert_allclose(frames.ball_xy[150], [32.5, 39.0])     # second half, same script


def test_speed_converted_to_mps(laid_out):
    frames, _, _, _ = laid_out
    assert frames.speed_unit == "m/s"
    # fixture players have S = 1.0 km/h
    np.testing.assert_allclose(frames.players_speed_mps[0, :4], 1.0 / 3.6)


@pytest.mark.parametrize("status, possession, expected", [
    (0, 1, "D"), (0, 2, "D"), (0, -1, "D"), (1, 1, "H"), (1, 2, "A"), (1, -1, "X"),
    (-1, 1, "X"), (-1, -1, "X"),
])
def test_target_mapping(status, possession, expected):
    assert convert.target_label(status, possession) == expected


def test_target_from_fixture(laid_out):
    _, target, _, _ = laid_out
    assert set(target[:40]) == {"H"} and set(target[40:50]) == {"D"}
    assert set(target[50:60]) == {"H"} and set(target[60:100]) == {"A"}


def test_writer_takes_no_labels():
    params = inspect.signature(convert.write_atd).parameters
    assert not any(name in params for name in ("labels", "target", "status", "possession"))


def test_round_trip_through_frozen_v1_loader(laid_out, tmp_path):
    frames, _, _, meta = laid_out
    meta_path, raw_path = convert.write_atd(frames, meta, str(tmp_path / "atd"))
    tr = atd.load_arrays(meta_path, raw_path, len(frames.ball_xy), meta.pitch)
    np.testing.assert_allclose(tr.ball_xy, frames.ball_xy, atol=1e-3, equal_nan=True)
    np.testing.assert_allclose(tr.players_xy, frames.players_xy, atol=1e-3, equal_nan=True)
    np.testing.assert_allclose(tr.players_speed, frames.players_speed_mps, atol=1e-4,
                               equal_nan=True)
    assert tr.team_of_slot.tolist() == frames.team_of_slot.tolist()
    assert tr.ball_present.tolist() == (~np.isnan(frames.ball_xy[:, 0])).tolist()


def test_team_mapping_home_first(laid_out, tmp_path):
    frames, _, _, meta = laid_out
    meta_path, _ = convert.write_atd(frames, meta, str(tmp_path / "atd"))
    assert baseline.team_names(meta_path) == meta.names           # home first, as V1 assumes
    assert frames.team_of_slot.tolist() == [0, 0, 1, 1]


def test_fallback_and_v1_parsers_agree_on_written_file(laid_out, tmp_path):
    frames, _, _, meta = laid_out
    meta_path, raw_path = convert.write_atd(frames, meta, str(tmp_path / "atd"))
    tr = atd.load_arrays(meta_path, raw_path, len(frames.ball_xy), meta.pitch)
    expected = baseline.nearest_player_labels(raw_path, baseline.team_of_each_track(meta_path),
                                              len(frames.ball_xy), meta.pitch)
    assert atd.nearest_team(tr).tolist() == expected.tolist()


def test_written_files_carry_no_labels(parsed, tmp_path):
    track, labels, meta = parsed
    frames, _, _ = convert.layout(track, labels, meta)
    meta_path, raw_path = convert.write_atd(frames, meta, str(tmp_path / "a"))
    for path in (meta_path, raw_path):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "BallStatus" not in text and "BallPossession" not in text


def test_written_bytes_independent_of_labels(parsed, tmp_path):
    track, labels, meta = parsed
    flipped = dfl_parse.Labels(
        data=[dfl_parse.SectionLabels(ball_status=1 - l.ball_status,
                                      possession=3 - l.possession) for l in labels.data],
        counts=labels.counts)
    outputs = []
    for name, lab in (("a", labels), ("b", flipped)):
        frames, target, _ = convert.layout(track, lab, meta)
        _, raw_path = convert.write_atd(frames, meta, str(tmp_path / name))
        with open(raw_path, "rb") as handle:
            outputs.append(handle.read())
    assert outputs[0] == outputs[1]
