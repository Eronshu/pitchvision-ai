"""M1 transfer-robustness study (design spec: "Robustness study", fixed before evaluation).

  A. feature-group ablations: separate M1 models refit without a feature group (never zeroed at
     inference), same caches, folds, training-only threshold selection and evaluation seeds
  B. alternative label-agnostic degradations, built from tracking only, on which frozen V1, the
     official fallback and every M1 model are compared on identical inputs

The frozen M1 artifacts are only read; everything new is written under a separate directory.
"""

import hashlib
import json
import os
import tempfile

import joblib
import numpy as np

from pitchvision import atd, config, pipeline
from pvext import benchmark, convert, degrade
from pvext.v2 import experiment, features, m1
from starter import baseline

# Params
FPS = 25
SPEED_HALF_WINDOW = 2          # frames each side for the position-derived speed field

ABLATIONS = {
    "full": (),
    "no_ball_presence": ("ball_present", "ball_gap_s", "ball_present_mean_1s",
                         "ball_present_mean_3s", "ball_present_mean_10s"),
    "no_slot_speed": ("players_speed_slot_p50", "players_speed_slot_p90"),
}
ABLATIONS["no_both"] = ABLATIONS["no_ball_presence"] + ABLATIONS["no_slot_speed"]

ALT_PRESETS = {
    "alt_uniform_dropout": {"overrides": {"hard_weight": 1.0}},
    "alt_long_gaps": {"overrides": {"ball_gap_s": 6.0}},
    "alt_speed_from_positions": {"overrides": {}, "speed_field": "positions"},
    "alt_noisy_ball": {"overrides": {"false_ball_rate": 0.02, "noise_ball_m": 1.0}},
    "alt_sparse_players": {"overrides": {"player_missing": 0.25, "player_gap_s": 4.0,
                                         "tracklet_mean_s": 8.0}},
}
ALT_BASE = "doc"


def columns(names, variant):
    """Indices of the features a variant keeps; every removed name must exist."""
    drop = set(ABLATIONS[variant])
    unknown = drop - set(names)
    if unknown:
        raise ValueError("unknown features for {}: {}".format(variant, sorted(unknown)))
    return np.array([i for i, n in enumerate(names) if n not in drop], dtype=np.int64)


def speed_from_positions(players_xy, half=SPEED_HALF_WINDOW, fps=FPS):
    """Per-slot speed (m/s) by central difference over +-half frames; NaN where either end is
    missing or out of range."""
    T = players_xy.shape[0]
    out = np.full(players_xy.shape[:2], np.nan, dtype=np.float64)
    if T > 2 * half:
        step = players_xy[2 * half:].astype(np.float64) - players_xy[:-2 * half].astype(np.float64)
        out[half:T - half] = np.linalg.norm(step, axis=2) / (2 * half / fps)
    out[np.isnan(players_xy[..., 0])] = np.nan          # no speed for a slot missing at t itself
    return out


def alt_preset(name):
    return dict(degrade.PRESETS[ALT_BASE], **ALT_PRESETS[name]["overrides"],
                source="robustness study (pre-declared): {}".format(name))


def apply_alt(frames, name, seed):
    """Degraded copy for an alternative distribution: degrade.apply with the overrides, then the
    optional speed-field replacement. Uses tracking only."""
    out = degrade.apply(frames, alt_preset(name), seed)
    if ALT_PRESETS[name].get("speed_field") == "positions":
        out = out._replace(players_speed_mps=speed_from_positions(out.players_xy).astype(np.float32))
    return out


def degraded(frames, preset, seed):
    if preset in degrade.PRESETS:
        return degrade.apply(frames, degrade.PRESETS[preset], seed)
    return apply_alt(frames, preset, seed)


# -- training (A) ----------------------------------------------------------------------------------

def _names(src_work, match):
    data = np.load(os.path.join(src_work, "train", "{}_{}_{}.npz".format(
        match, m1.TRAIN_PRESETS[0], m1.TRAIN_SEEDS[0])))
    return json.loads(str(data["names"]))


def train_variant(variant, kind, train_ids, test_ids, src_work, out_work):
    """Refit M1 on the cached training rows restricted to the variant's columns; threshold by inner
    leave-one-match-out on the training matches only. Returns the model path."""
    names = _names(src_work, train_ids[0])
    keep = columns(names, variant)
    cfg = config.load(m1.V1_CONFIG)
    train = {}
    for match in train_ids:
        if _names(src_work, match) != names:
            raise ValueError("feature names differ for {}".format(match))
        train[match] = [dict(s, X=s["X"][:, keep]) for s in experiment._series(src_work, match)]
    threshold, curve = m1.select_threshold(train, cfg["alive"], fps=FPS / m1.STRIDE)
    X = np.concatenate([s["X"] for items in train.values() for s in items])
    y = np.concatenate([s["y"] for items in train.values() for s in items])
    model = m1.fit(X, y)
    os.makedirs(os.path.join(out_work, "models"), exist_ok=True)
    path = os.path.join(out_work, "models", "{}_{}_{}.joblib".format(variant, kind, "-".join(test_ids)))
    joblib.dump({"model": model, "threshold": threshold, "curve": curve, "kind": kind,
                 "variant": variant, "features": [names[i] for i in keep],
                 "train": list(train_ids), "test": list(test_ids), "n_rows": int(len(y))}, path)
    return path


# -- evaluation (A and B) --------------------------------------------------------------------------

def predict_dead(bundle, names, X, blocks, cfg_alive):
    """Dead mask from a model bundle; the bundle's feature list selects the columns (the frozen M1
    models carry none and use all)."""
    wanted = bundle.get("features")
    if wanted is not None:
        index = {n: i for i, n in enumerate(names)}
        X = X[:, [index[n] for n in wanted]]
    elif X.shape[1] != bundle["model"].n_features_in_:
        raise ValueError("feature count mismatch")
    prob = bundle["model"].predict_proba(X)[:, 1]
    return m1.persist(prob, blocks, cfg_alive, FPS, bundle["threshold"])


def evaluate_match(match, model_paths, conditions, root, work):
    """One row per (condition, method) for a held-out match. `conditions` is a list of
    (preset_name, seed) with frozen or alternative preset names; `model_paths` maps a method name
    to a model bundle whose test set holds `match`."""
    cfg = config.load(m1.V1_CONFIG)
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    for name, bundle in models.items():
        experiment.check_held_out(bundle, match, name)
    frames, target, blocks, meta = experiment._layout(match, root)
    n = len(target)
    rows = []
    for preset, seed in conditions:
        copy = degraded(frames, preset, seed)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            meta_path, raw_path = convert.write_atd(copy, meta, tmp)
            input_sha = benchmark._sha256_file(raw_path)
            tr = atd.load_arrays(meta_path, raw_path, n, meta.pitch)
            codes = baseline.nearest_player_labels(raw_path, baseline.team_of_each_track(meta_path),
                                                   n, meta.pitch)
        predictions = {"v1": pipeline.run(tr, cfg, FPS, meta.pitch, blocks),
                       "fallback": baseline.smooth(codes, benchmark.FALLBACK_WINDOW)}
        names, X = features.compute(tr, meta.pitch, blocks, FPS)
        for name, bundle in models.items():
            dead = predict_dead(bundle, names, X, blocks, cfg["alive"])
            predictions[name] = m1.run_with_dead(tr, cfg, FPS, meta.pitch, blocks, dead)
        for method, pred in predictions.items():
            row = {"match": match, "distribution": "frozen" if preset in degrade.PRESETS else "alt",
                   "preset": preset, "seed": seed, "method": method, "input_sha256": input_sha,
                   "prediction_sha256": hashlib.sha256(
                       "".join(pred.tolist()).encode("ascii")).hexdigest()}
            row.update(benchmark.score(target, pred, blocks))
            rows.append(benchmark.flatten(row))
    return rows


# -- evaluation grid -------------------------------------------------------------------------------

VARIANT_NAMES = ("full", "no_ball_presence", "no_slot_speed", "no_both")
METHODS = ("v1", "fallback") + tuple("m1_{}_{}".format(v, k) for v in VARIANT_NAMES
                                    for k in ("lomo", "league"))
GRID_METRICS = ("d_precision", "d_recall", "d_f1", "alive_dead_f1", "spell_f1_mean",
                "d_share_pred", "d_share_true")


def conditions():
    """(preset, seed) pairs of the study: the frozen presets, then the alternatives."""
    return ([(p, s) for p in m1.EVAL_PRESETS for s in m1.EVAL_SEEDS]
            + [(p, s) for p in ALT_PRESETS for s in m1.EVAL_SEEDS])


def validate_grid(rows, matches=m1.MATCHES, methods=METHODS, metrics=GRID_METRICS):
    """Raise ValueError unless `rows` is exactly matches x conditions x methods, once each, with the
    right distribution tag, and every required metric finite. Returns the row count."""
    expected = {(m, p, s, k) for m in matches for p, s in conditions() for k in methods}
    seen = set()
    for r in rows:
        seed = float(r["seed"])
        if seed != int(seed):
            raise ValueError("non-integer seed {!r}".format(r["seed"]))
        key = (r["match"], r["preset"], int(seed), r["method"])
        if key in seen:
            raise ValueError("duplicate row {}".format(key))
        seen.add(key)
        want = "frozen" if r["preset"] in m1.EVAL_PRESETS else "alt"
        if r.get("distribution") != want:
            raise ValueError("{} tagged {!r}, expected {!r}".format(key, r.get("distribution"), want))
        for metric in metrics:
            value = r.get(metric)
            if value is None or not np.isfinite(float(value)):
                raise ValueError("{} is not finite in {}: {!r}".format(metric, key, value))
    missing, extra = expected - seen, seen - expected
    if missing or extra:
        raise ValueError("grid mismatch: {} missing {}, {} unexpected {}".format(
            len(missing), sorted(missing)[:3], len(extra), sorted(extra)[:3]))
    return len(seen)


# -- reading (B) -----------------------------------------------------------------------------------

def robustness_reading(rows, preset, candidate, matches=m1.MATCHES):
    """Pre-declared descriptive reading: robust if D F1 > V1 in >= 6 of 7 matches and the median
    D precision change >= -0.05."""
    d = {m: v - experiment.per_match(rows, "v1", preset, "d_f1")[m]
         for m, v in experiment.per_match(rows, candidate, preset, "d_f1").items()}
    p = {m: v - experiment.per_match(rows, "v1", preset, "d_precision")[m]
         for m, v in experiment.per_match(rows, candidate, preset, "d_precision").items()}
    if set(d) != set(matches):
        raise ValueError("missing matches for {} / {}".format(candidate, preset))
    higher = sum(v > 0 for v in d.values())
    median_p = float(np.median(list(p.values())))
    return {"d_f1_higher": higher, "median_precision_change": median_p,
            "robust": higher >= 6 and median_p >= -0.05}
