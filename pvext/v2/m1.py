"""V2 M1: logistic-regression dead-ball model with V1-style persistence (design spec, M1 section).

Everything here is fixed before training:
  folds         7 leave-one-match-out folds, plus 2 leave-league-out folds
  data          training copies from presets mild/doc/severe with seeds 100 and 101; evaluation only
                on the held-out match(es), with the frozen presets and seeds 0, 1, 2
  model         StandardScaler -> LogisticRegression(C=1, l2, lbfgs, max_iter=2000, random_state=0)
  persistence   V1's own values (configs/v1.json alive.smooth_s/min_dead_s/min_alive_s), per block
  threshold     chosen from THRESHOLDS by inner leave-one-match-out over the training matches only

run_with_dead() puts a dead mask into the frozen V1 possession machine exactly where
alive.dead_mask() sits in pitchvision.pipeline.run. V1 itself is imported, never edited.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pitchvision import atd, features, possession

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1_CONFIG = os.path.join(ROOT, "configs", "v1.json")

# Params
BUNDESLIGA = ("J03WMX", "J03WN1")
SECOND_DIVISION = ("J03WPY", "J03WOH", "J03WOY", "J03WQQ", "J03WR9")
MATCHES = BUNDESLIGA + SECOND_DIVISION
TRAIN_PRESETS = ("mild", "doc", "severe")
TRAIN_SEEDS = (100, 101)
EVAL_PRESETS = ("mild", "doc", "severe")
EVAL_SEEDS = (0, 1, 2)
STRIDE = 5
THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))      # 0.30 .. 0.70


def folds(kind):
    """[(train_match_ids, test_match_ids)] for "lomo" (7 folds) or "league" (2 folds)."""
    if kind == "lomo":
        return [([m for m in MATCHES if m != held], [held]) for held in MATCHES]
    if kind == "league":
        return [(list(SECOND_DIVISION), list(BUNDESLIGA)),
                (list(BUNDESLIGA), list(SECOND_DIVISION))]
    raise ValueError("unknown fold kind {!r}".format(kind))


def fit(X, y):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000,
                                             random_state=0))
    return model.fit(X, y)


def persist(prob, blocks, cfg_alive, fps, threshold):
    """Dead mask from probabilities: centred smoothing, threshold, then V1's minimum-duration rules,
    inside each block (frames outside blocks stay alive)."""
    out = np.zeros(len(prob), dtype=bool)
    window = max(1, int(round(cfg_alive["smooth_s"] * fps)))
    min_dead = int(round(cfg_alive["min_dead_s"] * fps))
    min_alive = int(round(cfg_alive["min_alive_s"] * fps))
    for _, first, last in blocks:
        p = np.asarray(prob[first:last + 1], dtype=np.float64)
        dead = _centred_mean(p, window) >= threshold
        for a, b in _runs(dead):
            if b - a < min_dead:
                dead[a:b] = False
        n = len(dead)
        for a, b in _runs(~dead):
            if 0 < a and b < n and b - a < min_alive:
                dead[a:b] = True
        out[first:last + 1] = dead
    return out


def d_f1(pred, truth, blocks):
    graded = np.zeros(len(truth), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    p, t = pred[graded], truth[graded]
    tp = float(np.sum(p & t))
    denominator = 2 * tp + float(np.sum(p & ~t)) + float(np.sum(~p & t))
    return 2 * tp / denominator if denominator else 0.0


def pick_threshold(curve):
    best = max(curve.values())
    tied = [t for t, value in curve.items() if value >= best - 1e-12]
    return min(tied, key=lambda t: (abs(t - 0.5), t))


def select_threshold(train, cfg_alive, fps):
    """Threshold maximising the mean per-match D F1 of inner leave-one-match-out predictions.

    `train` maps match id -> one series or a list of series, each {"X", "y", "blocks"} at the
    training stride. Only these matches are ever used.
    """
    series = {m: (v if isinstance(v, list) else [v]) for m, v in train.items()}
    curve = {t: 0.0 for t in THRESHOLDS}
    for held in series:
        others = [s for m, items in series.items() if m != held for s in items]
        model = fit(np.concatenate([s["X"] for s in others]),
                    np.concatenate([s["y"] for s in others]))
        for t in THRESHOLDS:
            scores = [d_f1(persist(model.predict_proba(s["X"])[:, 1], s["blocks"], cfg_alive, fps, t),
                           s["y"].astype(bool), s["blocks"]) for s in series[held]]
            curve[t] += float(np.mean(scores)) / len(series)
    return pick_threshold(curve), curve


def run_with_dead(tr, cfg, fps, pitch, blocks, dead):
    """Frozen V1 labels with `dead` in place of alive.dead_mask (mirrors pipeline.run)."""
    feat = features.compute(tr, cfg["features"], fps, pitch)
    nearest = atd.nearest_team(tr)
    out = np.full(len(tr.ball_xy), "X", dtype="<U1")
    for _, first, last in blocks:
        span = slice(first, last + 1)
        block_feat = features.Features(*(array[span] for array in feat))
        out[span] = possession.label_block(block_feat, dead[span], nearest[span],
                                           cfg["possession"], fps)
    return out


def _centred_mean(values, window):
    sums = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window // 2)
    high = np.minimum(len(values), index + window - window // 2)
    return (sums[high] - sums[low]) / (high - low)


def _runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
