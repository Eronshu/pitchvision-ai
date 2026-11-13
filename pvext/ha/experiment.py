"""HA1 experiment (design: docs/superpowers/specs/2026-11-06-pitchvision-ha-possession-design.md).

  dev      the five 2. Bundesliga matches: training caches, 5 leave-one-match-out folds (each with
           inner leave-one-match-out lambda selection on its 4 training matches), held-out
           evaluation against frozen V1 and frozen M1 (lomo) on the frozen evaluation copies.
  lockbox  the two Bundesliga matches, ONCE: a model trained on all 5 dev matches, evaluated
           against frozen V1 and frozen M1 (league_J03WMX-J03WN1, trained on 2. Bundesliga only).
Dev never reads a lockbox match. The frozen M1 artifacts are only read. Outputs go to `work`.
"""

import csv
import hashlib
import json
import os

import joblib
import numpy as np

from pitchvision import config, pipeline
from pvext import benchmark
from pvext.v2 import experiment as v2x
from pvext.v2 import features as m1_features
from pvext.v2 import m1
from pvext.ha import decode as ha_decode
from pvext.ha import features as ha_features

# Params
DEV = m1.SECOND_DIVISION
LOCKBOX = m1.BUNDESLIGA
LAMBDAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)          # HA1, as pre-registered
# Versioned protocols. HA1b (after ha1_001: lambda = 0.5, the grid minimum, in every fold) changes
# ONLY the lambda grid, extending it downwards; features, model, decoder and criteria are HA1's.
# HA2 (after the frame diagnostic): HA1b's grid with a team-swap-symmetrised gradient-boosting
# classifier instead of the linear model.
VARIANTS = {
    "ha1": {"lambdas": LAMBDAS, "model": "linear"},
    "ha1b": {"lambdas": (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0), "model": "linear"},
    "ha2": {"lambdas": (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0), "model": "gb_sym"},
    # HA3 (after ha2_003): HA2 plus the five ball-touch features (feature set "ha3")
    "ha3": {"lambdas": (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0), "model": "gb_sym",
            "features": "ha3"},
}
MODELS = {
    "linear": "StandardScaler(with_mean=False) + LogisticRegression(C=1, fit_intercept=False, "
              "lbfgs, max_iter=2000, random_state=0)",
    "gb_sym": "HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, "
              "random_state=0) on <=300000 rows (rng seed 0) + mirrored copies; "
              "p = (g(x) + 1 - g(-x)) / 2",
}
GB_ROWS = 300000
FPS = 25
STRIDE = m1.STRIDE
LOCKBOX_M1 = "league_{}.joblib".format("-".join(LOCKBOX))
REF = {"dev": "m1_lomo", "lockbox": "m1_league"}


def evaluate_spell(target, pred, blocks):
    """Official spell_f1_mean over the blocks only, with an X frame between consecutive blocks so
    that no spell can straddle two blocks (strided series have no gap of their own)."""
    t, p = [], []
    for i, (_, first, last) in enumerate(blocks):
        if i:
            t.append(np.array(["X"]))
            p.append(np.array(["X"]))
        t.append(np.asarray(target[first:last + 1]))
        p.append(np.asarray(pred[first:last + 1]))
    t, p = np.concatenate(t), np.concatenate(p)
    graded = []
    start = 0
    for i, (_, first, last) in enumerate(blocks):
        start += 1 if i else 0
        graded.append((i + 1, start, start + last - first))
        start += last - first + 1
    return benchmark.score(t, p, graded)["spell_f1_mean"]


# -- model -----------------------------------------------------------------------------------------

class SymmetricGB:
    """Gradient boosting made exactly team-swap symmetric: trained on (X, y) and (-X, not y),
    predicting p(x) = (g(x) + 1 - g(-x)) / 2, so that p(-x) = 1 - p(x)."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        p = 0.5 * (self.model.predict_proba(X)[:, 1] + 1.0 - self.model.predict_proba(-X)[:, 1])
        return np.column_stack([1.0 - p, p])


def fit_gb_sym(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    idx = np.random.default_rng(0).choice(len(X), min(len(X), GB_ROWS), replace=False)
    Xa = np.concatenate([X[idx], -X[idx]])
    ya = np.concatenate([y[idx], ~y[idx]])
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                        random_state=0)
    return SymmetricGB(gb.fit(Xa, ya))


def fit_model(X, y, kind="linear"):
    return {"linear": fit, "gb_sym": fit_gb_sym}[kind](X, y)


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs",
                                             max_iter=2000, random_state=0))
    return model.fit(X, y)


def rows_for_fit(series):
    X = np.concatenate([s["X"][np.isin(s["target"], ["H", "A"])] for s in series])
    y = np.concatenate([s["target"][np.isin(s["target"], ["H", "A"])] == "H" for s in series])
    return X, y


def pick_lambda(curve):
    best = max(curve.values())
    return max(lam for lam, v in curve.items() if v >= best - 1e-12)     # ties -> larger lambda


def select_lambda(train, lambdas=LAMBDAS, kind="linear"):
    """Inner leave-one-match-out over the given training matches only. `train` maps match ->
    list of strided series {"X", "target", "blocks"}; decoding uses each series' true dead mask."""
    curve = {lam: 0.0 for lam in lambdas}
    for held in train:
        model = fit_model(*rows_for_fit([s for m, items in train.items() if m != held for s in items]),
                          kind=kind)
        for lam in lambdas:
            scores = []
            for s in train[held]:
                p = model.predict_proba(s["X"])[:, 1]
                pred = ha_decode.decode(p, s["target"] == "D", s["blocks"], lam, FPS / STRIDE)
                scores.append(evaluate_spell(s["target"], pred, s["blocks"]))
            curve[lam] += float(np.mean(scores)) / len(train)
    return pick_lambda(curve), curve


def feature_set(variant):
    return VARIANTS[variant].get("features", "ha1")


def label(bundle, tr, pitch, blocks, dead):
    names, X = ha_features.compute(tr, pitch, blocks, FPS,
                                   feature_set=bundle.get("feature_set", "ha1"))
    if names != bundle["features"]:
        raise ValueError("HA feature names/order differ from the model's")
    p = bundle["model"].predict_proba(X)[:, 1]
    return ha_decode.decode(p, dead, blocks, bundle["lam"], FPS)


# -- data ------------------------------------------------------------------------------------------

def _guard(match, allowed):
    if match not in allowed:
        raise ValueError("{} is not allowed in this stage ({})".format(match, sorted(allowed)))


def build_training(match, root, work, features="ha1"):
    """Strided HA feature caches of the training copies of one DEV match."""
    _guard(match, DEV)
    frames, target, blocks, meta = v2x._layout(match, root)
    os.makedirs(os.path.join(work, "train"), exist_ok=True)
    paths = []
    for preset in m1.TRAIN_PRESETS:
        for seed in m1.TRAIN_SEEDS:
            tr, _ = v2x._tracking(frames, meta, preset, seed, work, len(target))
            names, X = ha_features.compute(tr, meta.pitch, blocks, FPS, feature_set=features)
            rows = v2x.stride_rows(X, target, blocks, STRIDE)
            path = os.path.join(work, "train", "{}_{}_{}.npz".format(match, preset, seed))
            np.savez(path, X=rows["X"], target=target[rows["frames"]],
                     blocks=json.dumps(rows["blocks"]), names=json.dumps(names))
            paths.append(path)
    return paths


def load_series(work, match):
    _guard(match, DEV)
    out = []
    for preset in m1.TRAIN_PRESETS:
        for seed in m1.TRAIN_SEEDS:
            d = np.load(os.path.join(work, "train", "{}_{}_{}.npz".format(match, preset, seed)))
            out.append({"X": d["X"], "target": d["target"],
                        "blocks": [tuple(b) for b in json.loads(str(d["blocks"]))],
                        "names": json.loads(str(d["names"]))})
    return out


def train(kind, train_ids, test_ids, work, out=None, variant="ha1"):
    """Fit on the cached series in `work`; the model goes to `out` (default `work`)."""
    lambdas = VARIANTS[variant]["lambdas"]
    kind_model = VARIANTS[variant]["model"]
    for m in train_ids:
        _guard(m, DEV)
    if set(train_ids) & set(test_ids):
        raise ValueError("train and test overlap")
    data = {m: load_series(work, m) for m in train_ids}
    names = data[train_ids[0]][0]["names"]
    if any(s["names"] != names for items in data.values() for s in items):
        raise ValueError("training caches have different feature names")
    lam, curve = select_lambda(data, lambdas, kind_model)
    model = fit_model(*rows_for_fit([s for items in data.values() for s in items]), kind=kind_model)
    out = out or work
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    path = os.path.join(out, "models", "ha1_{}_{}.joblib".format(kind, "-".join(test_ids)))
    joblib.dump({"model": model, "lam": lam, "curve": curve, "features": names, "kind": kind,
                 "variant": variant, "model_kind": kind_model,
                 "feature_set": feature_set(variant), "lambdas": list(lambdas),
                 "train": list(train_ids), "test": list(test_ids)}, path)
    return path


# -- evaluation ------------------------------------------------------------------------------------

def frozen_hashes(m1_report):
    out = {}
    with open(m1_report) as handle:
        for r in csv.DictReader(handle):
            out[(r["match"], r["preset"], int(float(r["seed"])), r["method"])] = (
                r["input_sha256"], r["prediction_sha256"])
    return out


def _sha(pred):
    return hashlib.sha256("".join(pred.tolist()).encode("ascii")).hexdigest()


def diagnostics(target, pred, blocks):
    alive = np.isin(target, ["H", "A"]) & np.isin(pred, ["H", "A"])
    sw_true, sw_hit, sw_pred, sw_spur = 0, 0, 0, 0
    for _, first, last in blocks:
        t, p = target[first:last + 1], pred[first:last + 1]
        ts = [i for i in range(1, len(t)) if t[i] in "HA" and t[i - 1] in "HA" and t[i] != t[i - 1]]
        ps = [i for i in range(1, len(p)) if p[i] in "HA" and p[i - 1] in "HA" and p[i] != p[i - 1]]
        sw_true += len(ts)
        sw_pred += len(ps)
        sw_hit += sum(any(abs(i - j) <= FPS and p[j] == t[i] for j in ps) for i in ts)
        sw_spur += sum(not any(abs(i - j) <= 2 * FPS for j in ts) for i in ps)
    return {"team_accuracy": float(np.mean(pred[alive] == target[alive])) if alive.any() else None,
            "switch_on_time_share": sw_hit / sw_true if sw_true else None,
            "spurious_switch_share": sw_spur / sw_pred if sw_pred else None,
            "n_pred_switches": sw_pred, "n_true_switches": sw_true}


def evaluate_match(match, stage, ha_path, m1_path, root, work, m1_report):
    allowed = DEV if stage == "dev" else LOCKBOX
    _guard(match, allowed)
    ha = joblib.load(ha_path)
    mm = joblib.load(m1_path)
    v2x.check_held_out(ha, match, "ha1")
    v2x.check_held_out(mm, match, "m1")
    if any(m not in DEV for m in ha["train"]):
        raise ValueError("HA1 model trained outside the dev matches")
    frozen = frozen_hashes(m1_report)
    cfg = config.load(m1.V1_CONFIG)
    frames, target, blocks, meta = v2x._layout(match, root)
    ref = REF[stage]
    rows = []
    for preset in m1.EVAL_PRESETS:
        for seed in m1.EVAL_SEEDS:
            tr, input_sha = v2x._tracking(frames, meta, preset, seed, work, len(target))
            v1 = pipeline.run(tr, cfg, FPS, meta.pitch, blocks)
            names, X = m1_features.compute(tr, meta.pitch, blocks, FPS)
            dead = m1.persist(mm["model"].predict_proba(X)[:, 1], blocks, cfg["alive"], FPS,
                              mm["threshold"])
            m1_pred = m1.run_with_dead(tr, cfg, FPS, meta.pitch, blocks, dead)
            if (input_sha, _sha(v1)) != frozen[(match, preset, seed, "v1")]:
                raise ValueError("frozen V1 not reproduced for {} {} {}".format(match, preset, seed))
            if (input_sha, _sha(m1_pred)) != frozen[(match, preset, seed, ref)]:
                raise ValueError("frozen M1 not reproduced for {} {} {}".format(match, preset, seed))
            ha_pred = label(ha, tr, meta.pitch, blocks, dead)
            if not np.array_equal(ha_pred == "D", m1_pred == "D"):
                raise ValueError("HA1 changed the dead mask")
            for method, pred in (("v1", v1), (ref, m1_pred), ("m1_ha1", ha_pred)):
                row = {"match": match, "stage": stage, "preset": preset, "seed": seed,
                       "method": method, "input_sha256": input_sha, "prediction_sha256": _sha(pred)}
                row.update(benchmark.score(target, pred, blocks))
                row.update(diagnostics(target, pred, blocks))
                rows.append(benchmark.flatten(row))
    return rows


# -- decision --------------------------------------------------------------------------------------

def _mean(rows, match, method, preset, key):
    v = [r[key] for r in rows if r["match"] == match and r["method"] == method and r["preset"] == preset]
    if len(v) != len(m1.EVAL_SEEDS):
        raise ValueError("expected {} seeds for {} {} {}".format(len(m1.EVAL_SEEDS), match, method, preset))
    return float(np.mean(v))


def gains(rows, matches, ref, preset="doc"):
    return {m: _mean(rows, m, "m1_ha1", preset, "spell_f1_mean") - _mean(rows, m, ref, preset, "spell_f1_mean")
            for m in matches}


GRID_METRICS = ("spell_f1_mean", "n_pred_spells", "n_true_spells")


def validate_grid(rows, stage):
    """Exactly matches x {v1?, reference, m1_ha1} x presets x seeds once each, required metrics
    finite. V1 rows are optional (they are reported, not used by the criteria)."""
    matches = DEV if stage == "dev" else LOCKBOX
    methods = {REF[stage], "m1_ha1"}
    seen = set()
    for r in rows:
        key = (r["match"], r["method"], r["preset"], int(float(r["seed"])))
        if r["method"] == "v1":
            continue
        if key in seen:
            raise ValueError("duplicate row {}".format(key))
        seen.add(key)
        for metric in GRID_METRICS:
            v = r.get(metric)
            if v is None or not np.isfinite(float(v)):
                raise ValueError("{} not finite in {}".format(metric, key))
    expected = {(m, k, p, s) for m in matches for k in methods for p in m1.EVAL_PRESETS
                for s in m1.EVAL_SEEDS}
    if seen != expected:
        raise ValueError("{} grid mismatch: missing {}, unexpected {}".format(
            stage, sorted(expected - seen)[:3], sorted(seen - expected)[:3]))


def go_criteria(dev_rows, lock_rows):
    validate_grid(dev_rows, "dev")
    validate_grid(lock_rows, "lockbox")
    lock = gains(lock_rows, LOCKBOX, REF["lockbox"])
    dev = gains(dev_rows, DEV, REF["dev"])
    other = {p: float(np.mean(list(gains(lock_rows, LOCKBOX, REF["lockbox"], p).values())))
             for p in ("mild", "severe")}
    ratio = {}
    for rows, matches in ((lock_rows, LOCKBOX), (dev_rows, DEV)):
        for m in matches:
            ratio[m] = (_mean(rows, m, "m1_ha1", "doc", "n_pred_spells")
                        / _mean(rows, m, "m1_ha1", "doc", "n_true_spells"))
    c = {
        "1_lockbox_both_up_mean_ge_0.04": {
            "value": lock, "pass": all(v > 0 for v in lock.values()) and np.mean(list(lock.values())) >= 0.04},
        "2_dev_4of5_up_mean_ge_0.04": {
            "value": dev, "pass": sum(v > 0 for v in dev.values()) >= 4 and np.mean(list(dev.values())) >= 0.04},
        "3_lockbox_mild_severe_mean_gt_0": {"value": other, "pass": all(v > 0 for v in other.values())},
        "4_spell_count_ratio_0.7_1.5": {"value": ratio,
                                        "pass": all(0.7 <= v <= 1.5 for v in ratio.values())},
    }
    if all(x["pass"] for x in c.values()):
        verdict = "GO"
    elif all(v > 0 for v in lock.values()) and np.mean(list(dev.values())) > 0:
        verdict = "WITHIN_NOISE_KEEP_M1"
    else:
        verdict = "NO-GO_KEEP_M1"
    return {"criteria": c, "verdict": verdict}
