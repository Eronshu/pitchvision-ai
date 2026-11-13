"""HA1 features: antisymmetry under a team swap, ATD-only inputs, NaN safety. Synthetic data."""

import inspect
import warnings

import numpy as np
import pytest

from pitchvision import atd
from pvext.ha import experiment as hx
from pvext.ha import features
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking


def swapped(tr):
    return tr._replace(team_of_slot=(1 - tr.team_of_slot).astype(np.int8))


def test_signature_takes_tracking_only():
    assert list(inspect.signature(features.compute).parameters) == [
        "tr", "pitch", "blocks", "fps", "cfg_features", "feature_set"]


def test_shape_finite_and_zero_outside_blocks():
    tr = tracking()
    blocks = [(1, 100, 1399)]
    names, X = features.compute(tr, PITCH, blocks, 25)
    assert X.shape == (1500, len(names)) and X.dtype == np.float32 and np.isfinite(X).all()
    assert not X[:100].any() and not X[1400:].any()
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("feature_set", ["ha1", "ha3"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_team_swap_negates_every_feature(seed, feature_set):
    tr = tracking(seed=seed, ball_missing=(300, 360))
    names, X = features.compute(tr, PITCH, BLOCKS, 25, feature_set=feature_set)
    names2, X2 = features.compute(swapped(tr), PITCH, BLOCKS, 25, feature_set=feature_set)
    assert names == names2
    assert np.allclose(X2, -X, atol=1e-5)
    assert np.abs(X).max() > 0


def test_model_is_exactly_team_symmetric():
    tr = tracking(seed=5)
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    y = (np.arange(len(X)) // 200) % 2 == 0
    model = hx.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    p_swap = model.predict_proba(features.compute(swapped(tr), PITCH, BLOCKS, 25)[1])[:, 1]
    assert np.allclose(p_swap, 1 - p, atol=1e-6)


def test_closer_team_pushes_the_distance_feature():
    tr = tracking(seed=1)
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    # the synthetic ball sits next to player 0, a home player: home is closer almost everywhere
    col = X[:, names.index("nearest_dist_diff")]
    assert np.mean(col[50:1450] > 0) > 0.95


def test_missing_ball_is_carried_one_second_then_neutral():
    tr = tracking(seed=2, ball_missing=(400, 600))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        names, X = features.compute(tr, PITCH, BLOCKS, 25)
    col = X[:, names.index("nearest_dist_diff")]
    assert col[410] != 0                      # carried (<= 1 s after the last sighting)
    assert col[450] == 0 and col[590] == 0    # beyond 1 s: no evidence


def test_attack_sign_follows_the_teams():
    tr = tracking(seed=3, kickoff=(0, 1500))  # home in the low-x half throughout
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    sign = features._attack_sign(tr, BLOCKS)
    assert set(np.unique(sign)) == {1.0}
    assert set(np.unique(features._attack_sign(swapped(tr), BLOCKS))) == {-1.0}


def test_sticky_memory_holds_and_ages():
    close = np.array([0, 1, 0, 0, -1, 0], dtype=float)
    sticky, age = features._sticky(close, [(1, 0, 5)])
    assert sticky.tolist() == [0, 1, 1, 1, -1, -1] and age.tolist() == [0, 0, 1, 2, 0, 1]


def test_side_means():
    v = np.arange(10, dtype=float)
    assert features._side_mean(v, 2, past=True)[5] == 3.5
    assert features._side_mean(v, 2, past=False)[5] == 6.5
    assert np.isnan(features._side_mean(v, 2, past=True)[0])


TOUCH_NAMES = ["last_touch_team", "last_touch_decayed", "next_touch_team", "next_touch_decayed",
               "touch_balance_3s"]


def test_ha3_appends_exactly_the_touch_features():
    tr = tracking(seed=0)
    n1, X1 = features.compute(tr, PITCH, BLOCKS, 25)
    n3, X3 = features.compute(tr, PITCH, BLOCKS, 25, feature_set="ha3")
    assert n3 == n1 + TOUCH_NAMES and np.array_equal(X3[:, :len(n1)], X1)
    with pytest.raises(ValueError):
        features.compute(tr, PITCH, BLOCKS, 25, feature_set="nope")


def touch_play(missing=None):
    """Ball runs in +x at 10 m/s onto a home player (frame 100), leaves at 2.5 m/s towards an away
    player (frame 300), and comes back at 2.5 m/s. All other players far away."""
    tr = tracking(seed=0)
    T = len(tr.ball_xy)
    t = np.arange(T, dtype=float)
    x = np.where(t <= 100, 50 - (100 - t) * 0.4, np.where(t <= 300, 50 + (t - 100) * 0.1,
                                                             70 - (t - 300) * 0.1))
    ball = np.column_stack([x, np.full(T, 34.0)])
    if missing:
        ball[missing[0]:missing[1]] = np.nan
    players = np.empty((T, 22, 2))
    players[:] = np.column_stack([np.linspace(5, 100, 22), np.full(22, 64.0)])[None]
    players[:, 0] = (50.0, 35.0)
    players[:, 11] = (70.0, 35.0)
    return tr._replace(players_xy=players.astype(tr.players_xy.dtype),
                       ball_xy=ball.astype(tr.ball_xy.dtype))


def _cols(tr):
    names, X = features.compute(tr, PITCH, BLOCKS, 25, feature_set="ha3")
    return {n: X[:, names.index(n)] for n in TOUCH_NAMES}


def test_touches_are_found_and_attributed():
    c = _cols(touch_play())
    assert c["last_touch_team"][150] == 1 and c["last_touch_team"][400] == -1
    assert c["last_touch_team"][50] == 0                     # nothing touched yet
    assert c["next_touch_team"][50] == 1 and c["next_touch_team"][200] == -1
    assert c["next_touch_team"][400] == 0                    # no later touch
    assert 0 < c["last_touch_decayed"][120] < 1 and c["last_touch_decayed"][120] > c["last_touch_decayed"][180]
    assert c["touch_balance_3s"][100] > 0 and c["touch_balance_3s"][300] < 0
    assert c["touch_balance_3s"][200] == 0


def test_missing_ball_makes_no_touch():
    c = _cols(touch_play(missing=(280, 320)))
    assert c["last_touch_team"][400] == 1                    # the away touch was not seen
    assert c["next_touch_team"][200] == 0
