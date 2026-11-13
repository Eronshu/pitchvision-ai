import numpy as np
import pytest

from pitchvision import alive, config, features
from tests.synthetic import make_tracking

FPS = 25
PITCH = (105.3, 68.0)
NAN = np.nan


def scenario(ball_xy, player_speed, holder=None, holder_offset_m=0.3):
    """Four players far from the ball, all moving at `player_speed` (scalar or (T,) array).

    `holder` is a slice of frames during which slot 0 stands `holder_offset_m` from the ball
    (a goalkeeper with the ball in hand, or a kicker beside it).
    """
    ball_xy = np.asarray(ball_xy, dtype=float)
    n = len(ball_xy)
    players = np.tile(np.array([[20.0, 20.0], [30.0, 20.0], [40.0, 50.0], [60.0, 50.0]]),
                      (n, 1, 1))
    if holder is not None:
        players[holder, 0] = ball_xy[holder] + [holder_offset_m, 0.0]
    speed = np.broadcast_to(np.asarray(player_speed, dtype=float).reshape(-1, 1)
                            if np.ndim(player_speed) else player_speed, (n, 4)).copy()
    tr = make_tracking(players, ball_xy, [0, 1, 0, 1], speed)
    return features.compute(tr, config.defaults()["features"], FPS, PITCH)


def rolling_ball(n, speed_mps=5.0, y=34.0):
    x = 30.0 + (speed_mps / FPS) * (np.arange(n) % 250)
    return np.stack([x, np.full(n, y)], axis=1)


def kick_from(spot, n, speed_mps=5.0):
    """n frames of a ball rolling away from `spot` (a restart from where it rested)."""
    x = spot[0] + (speed_mps / FPS) * np.arange(1, n + 1)
    return np.stack([x, np.full(n, spot[1])], axis=1)


def cfg(**overrides):
    return config.from_dict({"alive": overrides})["alive"]


def test_missing_ball_active_play_not_dead():
    ball = rolling_ball(250)
    ball[100:150] = NAN  # 2 s
    assert not alive.dead_mask(scenario(ball, 3.0), cfg(), FPS).any()


def test_single_soft_cue_not_dead():
    feat = scenario(rolling_ball(250), 0.3)  # 10 s of a slow field, ball rolling
    assert not alive.dead_mask(feat, cfg(), FPS).any()


def test_static_ball_alone_not_dead():
    ball = np.tile([50.0, 34.0], (250, 1))  # held ball, field still running
    assert not alive.dead_mask(scenario(ball, 3.0), cfg(), FPS).any()


def test_long_gap_alone_not_dead():
    ball = rolling_ball(500)
    ball[100:400] = NAN  # 12 s missing, field running
    assert not alive.dead_mask(scenario(ball, 3.0), cfg(), FPS).any()


def test_static_and_low_motion_dead():
    ball = rolling_ball(500)
    ball[150:275] = [50.0, 34.0]  # 5 s at rest
    speed = np.full(500, 3.0)
    speed[150:275] = 0.2
    mask = alive.dead_mask(scenario(ball, speed), cfg(), FPS)
    assert mask[175:250].all()
    assert not mask[:100].any() and not mask[350:].any()


def test_missing_ball_and_low_motion_dead():
    ball = rolling_ball(500)
    ball[150:300] = NAN  # 6 s missing during a stoppage
    speed = np.full(500, 3.0)
    speed[150:300] = 0.2
    mask = alive.dead_mask(scenario(ball, speed), cfg(), FPS)
    assert mask[200:290].all() and not mask[:100].any()


@pytest.mark.parametrize("metres_outside, expected", [(3.0, True), (0.5, False)])
def test_out_of_bounds_dead(metres_outside, expected):
    ball = rolling_ball(400)
    ball[150:250] = [50.0, -metres_outside]  # held outside for 4 s
    mask = alive.dead_mask(scenario(ball, 3.0), cfg(), FPS)
    assert bool(mask[175:225].all()) is expected
    assert not mask[:100].any()


def test_short_dead_dropped():
    ball = rolling_ball(400)
    ball[150:175] = [50.0, -3.0]  # 1 s blip outside
    assert not alive.dead_mask(scenario(ball, 3.0), cfg(), FPS).any()


def test_short_alive_between_dead_absorbed():
    ball = rolling_ball(600)
    ball[100:250] = [50.0, -3.0]
    ball[274:400] = [50.0, -3.0]  # ~1 s back in play between two out-of-bounds stretches
    mask = alive.dead_mask(scenario(ball, 3.0), cfg(), FPS)
    assert mask[150:350].all()


def test_offsets():
    ball = rolling_ball(500)
    ball[150:275] = [50.0, 34.0]
    speed = np.full(500, 3.0)
    speed[150:275] = 0.2
    feat = scenario(ball, speed)
    base = alive.dead_mask(feat, cfg(), FPS)
    shifted = alive.dead_mask(feat, cfg(dead_start_offset_frames=5, dead_end_offset_frames=-3),
                              FPS)
    first, last = np.flatnonzero(base)[[0, -1]]
    assert np.flatnonzero(shifted)[0] == first + 5
    assert np.flatnonzero(shifted)[-1] == last - 3


def test_tag_evidence_adds_weight():
    feat = scenario(np.tile([50.0, 34.0], (250, 1)), 3.0)  # static ball only: 0.35
    tag = np.ones(250, dtype=np.float32)
    assert not alive.dead_mask(feat, cfg(), FPS).any()
    assert alive.dead_mask(feat, cfg(w_tag=0.3), FPS, tag_dead=tag)[50:200].all()


def test_score_bounded():
    ball = rolling_ball(300)
    ball[100:200] = [50.0, -3.0]
    score = alive.dead_score(scenario(ball, 0.0), cfg(), FPS, tag_dead=np.ones(300))
    assert score.min() >= 0.0 and score.max() <= 1.0


# -- review fixes (PR #1): held ball, event-triggered persistence, restart evidence -------------

def test_goalkeeper_holding_ball_not_dead():
    """Keeper holds the ball for 5 s while the field walks: live play, not a stoppage."""
    ball = rolling_ball(400)
    ball[100:225] = [10.0, 34.0]
    speed = np.full(400, 3.0)
    speed[100:225] = 0.3
    feat = scenario(ball, speed, holder=slice(100, 225), holder_offset_m=0.3)
    assert not alive.dead_mask(feat, cfg(), FPS).any()


def toward_touchline(ball, t_out, n_seen_out):
    """The ball runs to the touchline at 10 m/s and is seen beyond the 1 m margin for
    `n_seen_out` frames from t_out (a genuine exit); later frames are left to the caller."""
    t = np.arange(t_out - 60, t_out + n_seen_out)
    ball[t] = np.stack([np.full(len(t), 50.0), -1.3 + 0.4 * (t_out - t)], axis=1)
    return ball


def test_out_then_missing_then_restart_dead():
    """Ball seen outside for 2 frames, then lost for 5 s while players move, then restarted."""
    ball = toward_touchline(rolling_ball(400), 150, 2)
    ball[152:276] = NAN
    mask = alive.dead_mask(scenario(ball, 2.0), cfg(), FPS)
    assert mask[150:276].all()
    assert not mask[:140].any() and not mask[280:].any()


def test_dead_persists_until_restart():
    """After the ball goes out it is placed in-bounds and held still: DEAD until it moves."""
    ball = toward_touchline(rolling_ball(400), 100, 3)
    ball[103:250] = [50.0, 0.5]
    ball[250:] = kick_from((50.0, 0.5), 150)   # restart from where the ball rests
    mask = alive.dead_mask(scenario(ball, 2.0), cfg(), FPS)
    assert mask[100:250].all() and not mask[255:].any()


def test_dead_capped_without_restart():
    ball = toward_touchline(rolling_ball(400), 100, 2)
    ball[102:] = NAN
    mask = alive.dead_mask(scenario(ball, 2.0), cfg(max_dead_s=4.0), FPS)
    assert mask[100:200].all() and not mask[205:].any()


# -- second review (PR #1, fdca8d4) ------------------------------------------------------------

def test_noisy_single_out_then_missing_not_dead():
    """One spurious detection 37 m away, outside the pitch, then the ball is lost for 5 s while
    play goes on: a noise spike, not a stoppage."""
    ball = rolling_ball(400)
    ball[150] = [50.0, -3.0]
    ball[151:276] = NAN
    assert not alive.dead_mask(scenario(ball, 2.0), cfg(), FPS).any()


def test_sustained_out_without_approach_still_dead():
    """Seen outside for several frames in a row: out of play even with no tracked approach."""
    ball = rolling_ball(400)
    ball[150:156] = [50.0, -3.0]
    ball[156:276] = NAN
    mask = alive.dead_mask(scenario(ball, 2.0), cfg(), FPS)
    assert mask[150:276].all()


def test_timeout_does_not_reenter_while_soft_trigger_stays_on():
    """A soft stoppage outlives max_dead_s with no restart observed. The timeout must end DEAD
    deterministically: no immediate re-entry, and the ALIVE stretch is not absorbed back."""
    ball = rolling_ball(500)
    ball[50:350] = [50.0, 34.0]          # 12 s at rest, nobody near
    speed = np.full(500, 3.0)
    speed[50:350] = 0.2
    mask = alive.dead_mask(scenario(ball, speed), cfg(max_dead_s=4.0), FPS)
    first = int(np.flatnonzero(mask)[0])
    assert mask[first:first + 100].all()
    assert not mask[first + 100:350].any()


def test_soft_trigger_rearms_after_timeout():
    """After a timeout, a new stoppage that starts once the old evidence has cleared is caught."""
    ball = rolling_ball(800)
    ball[50:350] = [50.0, 34.0]
    ball[350:450] = kick_from((50.0, 34.0), 100)
    ball[450:650] = [70.0, 34.0]
    speed = np.full(800, 3.0)
    speed[50:350] = 0.2
    speed[450:650] = 0.2
    mask = alive.dead_mask(scenario(ball, speed), cfg(max_dead_s=4.0), FPS)
    assert mask[470:555].all()      # caught again (and timed out again after 4 s)


@pytest.mark.parametrize("held_s, expected_dead", [(5.0, False), (10.0, True)])
def test_holder_from_start_of_stoppage(held_s, expected_dead):
    """A player within r_hold_m from the very first frame of a still ball: a keeper's hold is
    short (six-second rule); a set-piece wait with the taker beside the ball lasts longer."""
    n_held = int(held_s * FPS)
    ball = rolling_ball(100 + n_held + 150)
    ball[100:100 + n_held] = [30.0, 34.0]
    ball[100 + n_held:] = kick_from((30.0, 34.0), 150)
    speed = np.full(len(ball), 3.0)
    speed[100:100 + n_held] = 0.3
    feat = scenario(ball, speed, holder=slice(100, 100 + n_held), holder_offset_m=0.6)
    mask = alive.dead_mask(feat, cfg(), FPS)
    if expected_dead:
        assert mask[125:100 + n_held].all()
    else:
        assert not mask.any()


def test_kicker_beside_ball_keeps_pre_kickoff_dead():
    """Pre-kickoff: ball on the spot, field still; the kicker steps up 2 s before the kick."""
    ball = rolling_ball(400)
    ball[:250] = [52.65, 34.0]
    ball[250:] = kick_from((52.65, 34.0), 150)
    speed = np.full(400, 3.0)
    speed[:250] = 0.2
    feat = scenario(ball, speed, holder=slice(200, 250), holder_offset_m=0.6)
    mask = alive.dead_mask(feat, cfg(), FPS)
    assert mask[15:250].all() and not mask[260:].any()
