import inspect
import warnings

import numpy as np
import pytest

from pitchvision import atd
from pvext import convert
from pvext.v2 import features
from tests.synthetic import make_tracking

FPS = 25
PITCH = (105.0, 68.0)


def tracking(T=1500, seed=0, ball_missing=None, slowdown_at=None, kickoff=None):
    """22 players (11 home slots, 11 away), random-walk positions, ball near player 0."""
    rng = np.random.default_rng(seed)
    base = rng.uniform([10, 8], [95, 60], size=(22, 2))
    speed = np.full(T, 3.0)
    if slowdown_at is not None:
        speed[slowdown_at:] = 0.3
    steps = rng.normal(0, 1, size=(T, 22, 2))
    steps /= np.linalg.norm(steps, axis=2, keepdims=True)
    players = base + np.cumsum(steps * (speed / FPS)[:, None, None], axis=0)
    if kickoff is not None:
        a, b = kickoff
        players[a:b, :11, 0] = np.linspace(10, 50, 11)[None, :]     # home in their own half
        players[a:b, 11:, 0] = np.linspace(55, 95, 11)[None, :]     # away in theirs
        players[a:b, :, 1] = 34.0
    players = np.clip(players, 1, [104, 67])
    ball = players[:, 0, :] + 0.6
    if ball_missing is not None:
        ball[ball_missing[0]:ball_missing[1]] = np.nan
    slot_speed = np.repeat(speed[:, None], 22, axis=1)
    return make_tracking(players, ball, [0] * 11 + [1] * 11, slot_speed)


BLOCKS = [(1, 0, 1499)]


@pytest.fixture(scope="module")
def plain():
    tr = tracking()
    return tr, features.compute(tr, PITCH, BLOCKS)


def test_signature_has_no_labels():
    params = inspect.signature(features.compute).parameters
    assert list(params) == ["tr", "pitch", "blocks", "fps"]


def test_shape_finite_and_named(plain):
    _, (names, X) = plain
    assert X.dtype == np.float32 and X.shape == (1500, len(names)) and len(set(names)) == len(names)
    assert np.isfinite(X).all()


@pytest.mark.parametrize("forbidden", ["_z", "height", "ball_speed_field", "status",
                                       "possession", "_S_", "dfl"])
def test_no_forbidden_inputs_in_names(plain, forbidden):
    _, (names, _) = plain
    assert not any(forbidden.lower() in n.lower() for n in names)


def test_slot_permutation_within_team_is_invariant(plain):
    tr, (_, X) = plain
    rng = np.random.default_rng(3)
    order = np.concatenate([rng.permutation(11), 11 + rng.permutation(11)])
    shuffled = tr._replace(players_xy=tr.players_xy[:, order], players_speed=tr.players_speed[:, order],
                           team_of_slot=tr.team_of_slot[order])
    _, Y = features.compute(shuffled, PITCH, BLOCKS)
    np.testing.assert_allclose(X, Y, rtol=1e-5, atol=1e-5)


def test_deterministic(plain):
    tr, (_, X) = plain
    assert np.array_equal(features.compute(tr, PITCH, BLOCKS)[1], X)


def test_long_missing_ball_is_finite_without_warnings():
    tr = tracking(ball_missing=(200, 1300))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, X = features.compute(tr, PITCH, BLOCKS)
    assert np.isfinite(X).all()


def test_deceleration_onset_before_slowdown():
    names, X = features.compute(tracking(slowdown_at=800), PITCH, BLOCKS)
    decel = X[:, names.index("decel_3s")]
    assert decel[780:800].mean() > 1.0          # just before the slowdown: past fast, future slow
    assert abs(decel[300:400].mean()) < 0.5     # steady play


def test_kickoff_formation_detected():
    names, X = features.compute(tracking(kickoff=(600, 700)), PITCH, BLOCKS)
    own_half = X[:, names.index("own_half_split")]
    assert own_half[620:680].min() == pytest.approx(1.0)
    assert own_half[100:500].mean() < 0.9


def test_ball_third_field_never_reaches_features(tmp_path):
    """The ATD ball group's third value (constant 0 from convert.write_atd) must be unreadable."""
    tr = tracking(T=300)
    frames = convert.FramesTrack(players_xy=tr.players_xy, players_speed_mps=tr.players_speed,
                                 team_of_slot=tr.team_of_slot, ball_xy=tr.ball_xy,
                                 ball_z=np.zeros(300, np.float32), pitch=PITCH, speed_unit="m/s")

    class Meta:
        home_id, away_id, names = "H", "A", ("Home", "Away")
    meta_path, raw_path = convert.write_atd(frames, Meta, str(tmp_path / "a"))
    altered = str(tmp_path / "altered.txt")
    with open(raw_path) as src, open(altered, "w") as dst:
        for line in src:
            head, ball = line.rstrip("\n").rsplit(":", 1)
            parts = ball.split(",")
            if parts[0] != "NaN":
                parts[2] = "99.0"
            dst.write(head + ":" + ",".join(parts) + "\n")
    blocks = [(1, 0, 299)]
    a = features.compute(atd.load_arrays(meta_path, raw_path, 300, PITCH), PITCH, blocks)[1]
    b = features.compute(atd.load_arrays(meta_path, altered, 300, PITCH), PITCH, blocks)[1]
    assert np.array_equal(a, b)


def test_frames_outside_blocks_are_zero(plain):
    tr, _ = plain
    names, X = features.compute(tr, PITCH, [(1, 100, 1399)])
    assert (X[:100] == 0).all() and (X[1400:] == 0).all()
