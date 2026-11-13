import numpy as np
import pytest

from pitchvision import atd, config, pipeline
from pvext.v2 import m1
from tests.ext import dfl_fixture as fx

FPS = 25
CFG = config.load(m1.V1_CONFIG)


# -- folds and seeds --------------------------------------------------------------------------------

def test_lomo_folds_are_disjoint_and_cover_all():
    folds = m1.folds("lomo")
    assert len(folds) == 7
    assert sorted(test for _, test in folds) == sorted([[m] for m in m1.MATCHES])
    for train, test in folds:
        assert not set(train) & set(test) and len(train) == 6


def test_league_folds():
    folds = m1.folds("league")
    assert sorted(map(sorted, (test for _, test in folds))) == sorted(
        [sorted(m1.BUNDESLIGA), sorted(m1.SECOND_DIVISION)])
    for train, test in folds:
        assert not set(train) & set(test)


def test_training_seeds_disjoint_from_evaluation_seeds():
    assert not set(m1.TRAIN_SEEDS) & set(m1.EVAL_SEEDS)
    assert m1.EVAL_SEEDS == (0, 1, 2) and m1.EVAL_PRESETS == ("mild", "doc", "severe")


# -- persistence ------------------------------------------------------------------------------------

def test_persistence_drops_short_dead_and_absorbs_short_alive():
    prob = np.zeros(400)
    prob[50:70] = 0.9                 # 0.8 s spike: dropped (< 2 s)
    prob[100:200] = 0.9               # 4 s
    prob[210:300] = 0.9               # after a 0.4 s live gap: absorbed (< 1 s)
    mask = m1.persist(prob, [(1, 0, 399)], CFG["alive"], FPS, threshold=0.5)
    assert not mask[40:80].any()
    assert mask[110:290].all()


def test_persistence_runs_per_block():
    prob = np.full(400, 0.9)
    mask = m1.persist(prob, [(1, 0, 149), (2, 250, 399)], CFG["alive"], FPS, threshold=0.5)
    assert not mask[150:250].any() and mask[10:140].all() and mask[260:390].all()


def test_persistence_threshold_is_respected():
    prob = np.full(300, 0.6)
    assert m1.persist(prob, [(1, 0, 299)], CFG["alive"], FPS, threshold=0.5).all()
    assert not m1.persist(prob, [(1, 0, 299)], CFG["alive"], FPS, threshold=0.7).any()


# -- injection into the frozen possession machine ---------------------------------------------------

def test_injecting_v1_mask_reproduces_frozen_v1(tmp_path):
    from tests.test_pipeline import BLOCKS, N, PITCH, stoppage_frames
    from tests.synthetic import write_fixture
    from pitchvision import alive, features
    meta, raw = write_fixture(tmp_path, stoppage_frames())
    tr = atd.load_arrays(meta, raw, N, PITCH)
    frozen = pipeline.run(tr, CFG, FPS, PITCH, BLOCKS)
    feat = features.compute(tr, CFG["features"], FPS, PITCH)
    dead = np.zeros(N, dtype=bool)
    for _, first, last in BLOCKS:
        span = slice(first, last + 1)
        dead[span] = alive.dead_mask(features.Features(*(a[span] for a in feat)), CFG["alive"], FPS)
    assert np.array_equal(m1.run_with_dead(tr, CFG, FPS, PITCH, BLOCKS, dead), frozen)


# -- model fit, determinism and training-only threshold selection -----------------------------------

def _toy(n=3000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) // 300) % 2 == 1
    X = rng.normal(0, 1, size=(n, 4)) + np.where(y[:, None], 1.5 + shift, 0.0)
    return X.astype(np.float32), y


def test_fit_is_deterministic():
    X, y = _toy()
    a = m1.fit(X, y).predict_proba(X)[:, 1]
    b = m1.fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)


def test_threshold_selection_uses_only_given_training_matches(monkeypatch):
    rows = {name: dict(zip(("X", "y"), _toy(seed=i))) | {"blocks": [(1, 0, 2999)]}
            for i, name in enumerate(["A", "B", "C", "HELD_OUT"])}
    seen = []
    real_fit = m1.fit

    def spy(X, y):
        seen.append(len(X))
        return real_fit(X, y)
    monkeypatch.setattr(m1, "fit", spy)
    train = {k: v for k, v in rows.items() if k != "HELD_OUT"}
    threshold, curve = m1.select_threshold(train, CFG["alive"], fps=FPS / m1.STRIDE)
    assert threshold in m1.THRESHOLDS
    assert all(size == 6000 for size in seen) and len(seen) == 3     # inner folds: 2 of 3 matches
    assert set(curve) == set(m1.THRESHOLDS)


def test_threshold_tie_breaks_toward_half():
    curve = {t: 0.8 for t in m1.THRESHOLDS}
    assert m1.pick_threshold(curve) == 0.5
