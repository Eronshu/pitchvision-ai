"""HA1 switch-penalty Viterbi decoding. Synthetic probabilities only."""

import numpy as np
import pytest

from pvext.ha import decode


def test_confident_evidence_is_followed():
    p = np.r_[np.full(100, 0.95), np.full(100, 0.05)]
    path = decode.viterbi(p, lam=1.0, fps=25)
    assert path[:95].all() and not path[105:].any()


def test_large_penalty_suppresses_short_flicker():
    p = np.r_[np.full(100, 0.9), np.full(10, 0.1), np.full(100, 0.9)]   # 0.4 s blip
    assert decode.viterbi(p, lam=4.0, fps=25).all()
    assert not decode.viterbi(p, lam=0.01, fps=25)[100:110].any()


def test_zero_penalty_is_framewise_argmax():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 300)
    assert np.array_equal(decode.viterbi(p, lam=0.0, fps=25), p >= 0.5)


def test_team_swap_symmetry():
    rng = np.random.default_rng(1)
    p = np.clip(np.r_[np.full(60, 0.8), np.full(80, 0.3), np.full(60, 0.7)]
                + rng.normal(0, 0.1, 200), 0.01, 0.99)
    for lam in (0.5, 2.0, 8.0):
        assert np.array_equal(decode.viterbi(1 - p, lam, 25), ~decode.viterbi(p, lam, 25))


def test_penalty_is_rate_invariant():
    p25 = np.r_[np.full(250, 0.8), np.full(250, 0.2)]
    p5 = p25[::5]
    for lam in (0.5, 2.0, 8.0):
        a = decode.viterbi(p25, lam, 25)
        b = decode.viterbi(p5, lam, 5)
        assert a.sum() // 5 == pytest.approx(b.sum(), abs=1)


def test_decode_keeps_dead_and_x_and_splits_segments():
    p = np.full(40, 0.9)
    p[20:] = 0.1
    dead = np.zeros(40, bool)
    dead[10:15] = True
    out = decode.decode(p, dead, [(1, 2, 37)], lam=100.0, fps=25)
    assert (out[:2] == "X").all() and (out[38:] == "X").all()
    assert (out[10:15] == "D").all()
    assert (out[2:10] == "H").all()
    # after the stoppage a new segment starts from a uniform prior: with a huge penalty it takes
    # one team for the whole segment, the one with more evidence (18 frames of A vs 5 of H)
    assert (out[15:38] == "A").all()


def test_segments():
    dead = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], bool)
    assert decode.segments(dead, [(1, 0, 9)]) == [(0, 2), (4, 7), (8, 10)]
    assert decode.segments(dead, [(1, 0, 4), (2, 5, 9)]) == [(0, 2), (4, 5), (5, 7), (8, 10)]
