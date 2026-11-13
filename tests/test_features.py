import warnings

import numpy as np
import pytest

from pitchvision import config, features
from tests.synthetic import make_tracking

FPS = 25
PITCH = (105.3, 68.0)
CFG = config.defaults()["features"]
NAN = np.nan


def _static(players, ball, n=1):
    """n identical frames: players [(x, y) | None] in metres, ball (x, y) | None."""
    p = np.array([[(NAN, NAN) if q is None else q for q in players]] * n, dtype=float)
    b = np.array([(NAN, NAN) if ball is None else ball] * n, dtype=float)
    return p, b


def test_distances():
    p, b = _static([(10, 10), (20, 10)], (11, 10))
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert f.dist_h[0] == pytest.approx(1.0)
    assert f.dist_a[0] == pytest.approx(9.0)


def test_no_players_is_inf_without_warning():
    p, b = _static([None, None], (11, 10), n=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert np.isinf(f.dist_h).all() and np.isinf(f.dist_a).all()
    assert np.isnan(f.players_motion).all()


def test_no_ball_is_inf():
    p, b = _static([(10, 10), (20, 10)], None, n=2)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert np.isinf(f.dist_h).all() and np.isnan(f.ball_speed).all()


def test_ball_speed():
    t = np.arange(30)
    b = np.stack([10 + 0.4 * t, np.full(30, 30.0)], axis=1)
    p = np.full((30, 2, 2), NAN)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    np.testing.assert_allclose(f.ball_speed[5:25], 10.0, rtol=1e-4)


def test_rel_speed_dribble():
    t = np.arange(30)
    b = np.stack([10 + 0.2 * t, np.full(30, 30.0)], axis=1)
    p = np.stack([np.stack([10 + 0.2 * t - 0.5, np.full(30, 30.0)], axis=1),
                  np.full((30, 2), [60.0, 30.0])], axis=1)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    np.testing.assert_allclose(f.ball_speed[5:25], 5.0, rtol=1e-4)
    np.testing.assert_allclose(f.rel_speed_h[5:25], 0.0, atol=1e-3)
    np.testing.assert_allclose(f.rel_speed_a[5:25], 5.0, rtol=1e-4)


L, W = PITCH


@pytest.mark.parametrize("ball, expected", [
    ((-0.5, 30), False), ((-1.5, 30), True),
    ((L + 0.5, 30), False), ((L + 1.5, 30), True),
    ((50, -0.5), False), ((50, -1.5), True),
    ((50, W + 0.5), False), ((50, W + 1.5), True),
    ((50, 30), False),
])
def test_ball_out_margin(ball, expected):
    p, b = _static([(10, 10), (20, 10)], ball)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert bool(f.ball_out[0]) is expected


def test_ball_out_false_when_missing():
    p, b = _static([(10, 10), (20, 10)], None)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert not f.ball_out[0]


def test_gap_len():
    b = np.array([(1, 1), (NAN, NAN), (NAN, NAN), (NAN, NAN), (1, 1)], dtype=float)
    p = np.full((5, 2, 2), 5.0)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert f.gap_len.tolist() == [0, 3, 3, 3, 0]


def test_players_motion_median():
    p = np.full((1, 3, 2), 5.0)
    speed = np.array([[1.0, 3.0, NAN]])
    f = features.compute(make_tracking(p, np.array([(1.0, 1.0)]), [0, 1, 0], speed),
                         CFG, FPS, PITCH)
    assert f.players_motion[0] == pytest.approx(2.0)


@pytest.mark.parametrize("before, outside, expected", [
    ((50.0, 0.2), (50.0, -1.3), True),     # 1.5 m (37.5 m/s) from 0.2 m inside: a real exit
    ((50.0, 34.0), (50.0, -3.0), False),   # 37 m in one frame from mid-pitch: a noise spike
    ((50.0, 5.0), (50.0, -1.5), False),    # 6.5 m in a frame (160 m/s): implausible
    ((50.0, 8.0), (50.0, -1.3), False),    # last seen 8 m inside, 9.3 m in a frame
    (None, (50.0, -1.3), False),           # no in-play sighting just before: unconfirmed
])
def test_exit_plausible(before, outside, expected):
    b = np.full((4, 2), NAN)
    if before is not None:
        b[1] = before
    b[2] = outside
    p = np.full((4, 2, 2), 5.0)
    f = features.compute(make_tracking(p, b, [0, 1]), CFG, FPS, PITCH)
    assert bool(f.exit_plausible[2]) is expected
    assert not f.exit_plausible[[0, 1, 3]].any()
