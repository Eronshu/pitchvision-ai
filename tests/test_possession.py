import numpy as np
import pytest

from pitchvision import config, possession
from pitchvision.features import Features

FPS = 25
INF = np.inf


class Script:
    """A Features timeline written segment by segment: distances and relative speeds per team."""

    def __init__(self, n):
        self.dist_h = np.full(n, INF, dtype=np.float32)
        self.dist_a = np.full(n, INF, dtype=np.float32)
        self.rel_h = np.zeros(n, dtype=np.float32)
        self.rel_a = np.zeros(n, dtype=np.float32)

    def set(self, start, stop, dist_h, dist_a, rel_h=0.0, rel_a=0.0):
        self.dist_h[start:stop], self.dist_a[start:stop] = dist_h, dist_a
        self.rel_h[start:stop], self.rel_a[start:stop] = rel_h, rel_a
        return self

    def features(self):
        n = len(self.dist_h)
        missing = np.isinf(self.dist_h) & np.isinf(self.dist_a)
        zeros = np.zeros(n, dtype=np.float32)
        return Features(dist_h=self.dist_h, dist_a=self.dist_a,
                        rel_speed_h=self.rel_h, rel_speed_a=self.rel_a,
                        ball_speed=np.where(missing, np.nan, zeros),
                        ball_out=np.zeros(n, dtype=bool),
                        exit_plausible=np.zeros(n, dtype=bool), players_motion=zeros + 3.0,
                        gap_len=missing.astype(np.int32))


def cfg(**overrides):
    return config.from_dict({"possession": overrides})["possession"]


def label(script, dead=None, nearest=None, restart_team=None, **overrides):
    feat = script.features()
    n = len(feat.dist_h)
    dead = np.zeros(n, dtype=bool) if dead is None else dead
    nearest = np.full(n, -1, dtype=np.int8) if nearest is None else nearest
    return possession.label_block(feat, dead, nearest, cfg(**overrides), FPS, restart_team)


def h_control(script, start, stop):
    return script.set(start, stop, dist_h=0.5, dist_a=8.0)


def test_evidence_bounds_and_missing_ball():
    s = Script(4).set(0, 1, 0.5, 8.0).set(1, 2, 0.5, 8.0, rel_h=1000.0).set(2, 3, 1.0, 1.2)
    e_h, e_a = possession.evidence(s.features(), cfg())
    assert e_h[0] == pytest.approx(1.0)
    assert 0.3 <= e_h[1] < 0.32          # speed never vetoes: floor 0.3
    assert e_h[2] == pytest.approx(0.2)  # contested: 0.2 m margin of 1 m
    assert e_h[3] == 0 and e_a[3] == 0   # no ball
    assert ((e_h >= 0) & (e_h <= 1)).all() and ((e_a >= 0) & (e_a <= 1)).all()


def test_pass_between_teammates_no_switch():
    s = h_control(Script(150), 0, 50)
    s.set(50, 58, dist_h=6.0, dist_a=2.0, rel_a=20.0)   # ball flies past an away player
    h_control(s, 58, 150)
    assert set(label(s)) == {"H"}


@pytest.mark.parametrize("boundary", ["evidence_start", "confirmed"])
def test_fast_interception_switches(boundary):
    s = h_control(Script(150), 0, 50)
    s.set(50, 55, dist_h=5.0, dist_a=0.4, rel_a=20.0)   # interception of a 20 m/s ball
    s.set(55, 150, dist_h=5.0, dist_a=0.4)               # and kept
    out = label(s, boundary=boundary)
    first_a = int(np.flatnonzero(out == "A")[0])
    assert (out[:50] == "H").all() and (out[first_a:] == "A").all()
    if boundary == "evidence_start":
        assert first_a == 50
    else:
        assert first_a > 50


def test_deflection_then_regain_no_switch():
    s = h_control(Script(150), 0, 50)
    s.set(50, 53, dist_h=4.0, dist_a=0.8, rel_a=20.0)
    h_control(s, 53, 150)
    assert set(label(s)) == {"H"}


def test_loose_ball_holds():
    s = h_control(Script(150), 0, 50)
    s.set(50, 100, dist_h=8.0, dist_a=8.0)
    assert set(label(s)) == {"H"}


def test_missing_ball_holds():
    s = h_control(Script(150), 0, 50)   # frames 50-99 keep the default: no ball
    h_control(s, 100, 150)
    assert set(label(s)) == {"H"}


def test_contested_ball_holds():
    s = h_control(Script(150), 0, 50)
    s.set(50, 100, dist_h=1.0, dist_a=0.9)   # both teams on the ball
    assert set(label(s)) == {"H"}


def test_unknown_backfill():
    s = Script(100).set(0, 25, dist_h=6.0, dist_a=6.0).set(25, 100, dist_h=7.0, dist_a=0.5)
    assert set(label(s)) == {"A"}


def test_unknown_unresolved_uses_nearest_or_x():
    s = Script(50).set(0, 50, dist_h=6.0, dist_a=6.0)
    nearest = np.full(50, -1, dtype=np.int8)
    assert set(label(s, nearest=nearest)) == {"X"}
    nearest[10:20] = 1
    nearest[20:25] = 0
    assert set(label(s, nearest=nearest)) == {"A"}


def test_dead_frames_are_d_and_restart_is_unknown():
    s = h_control(Script(120), 0, 40)
    s.set(40, 120, dist_h=7.0, dist_a=0.5)
    dead = np.zeros(120, dtype=bool)
    dead[10:40] = True
    out = label(s, dead=dead)
    assert (out[10:40] == "D").all()
    assert (out[:10] == "H").all()
    assert (out[40:] == "A").all()   # restart starts UNKNOWN: no H carried over the stoppage


def test_restart_team_seeds_state():
    s = Script(60).set(0, 60, dist_h=6.0, dist_a=6.0)   # nobody near after the restart
    restart = np.full(60, -1, dtype=np.int8)
    restart[0] = 1
    assert set(label(s, restart_team=restart)) == {"A"}
