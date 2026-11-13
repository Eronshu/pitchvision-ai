"""The final deployment M1 model: one model trained on all seven IDSSE matches.

Same architecture, 59 features, training presets/seeds, hyperparameters and persistence as the
validated M1 (pvext/v2/m1.py). The only difference from the seven leave-one-match-out validation
models is the training set (all seven matches, no held-out match) and, consequently, the threshold:
it is chosen by the same pre-declared procedure (m1.select_threshold, inner leave-one-match-out on
the training matches), here over all seven. The returned threshold is used as it is.

The performance evidence for this model is the LOMO / league-out / robustness evaluation of the
validation models. A training-set score of the final model is an implementation sanity check only.

Inputs are the frozen M1 training caches, read-only and verified against the frozen M1 manifest.
Outputs go to a new directory outside the repository; existing outputs are never overwritten.
"""

import hashlib
import json
import os
import platform
from importlib import metadata

import joblib
import numpy as np

from pitchvision import atd, config
from pvext.v2 import experiment, features, m1

# Params
FPS = 25
MODEL_FILE = "m1_final.joblib"
MANIFEST_FILE = "MANIFEST.json"
RELEASE_CONFIG_FILE = "release_config.json"
DEPENDENCIES = ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")
PERSISTENCE_KEYS = ("smooth_s", "min_dead_s", "min_alive_s")
ROOT = m1.ROOT
SOURCES = ("pvext/v2/features.py", "pvext/v2/m1.py", "pvext/v2/experiment.py", "pvext/v2/final.py",
           "pitchvision/features.py", "pitchvision/alive.py", "configs/v1.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names):
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def feature_schema():
    """The feature names, in order, that the current extractor produces (computed on a tiny
    synthetic Tracking, so the schema is the code's, not a copy of it)."""
    T = 60
    players = np.full((T, 4, 2), 30.0, dtype=np.float32)
    players[:, :, 0] += np.arange(4, dtype=np.float32)
    tr = atd.Tracking(players_xy=players, players_speed=np.zeros((T, 4), np.float32),
                      team_of_slot=np.array([0, 1, 0, 1], np.int8),
                      ball_xy=players[:, 0, :] + 0.5, ball_present=np.ones(T, bool))
    names, _ = features.compute(tr, (105.0, 68.0), [(1, 0, T - 1)], FPS)
    return names


def versions():
    return {name: metadata.version(name) for name in DEPENDENCIES}


def expected_caches():
    return ["{}_{}_{}.npz".format(m, p, s) for m in m1.MATCHES for p in m1.TRAIN_PRESETS
            for s in m1.TRAIN_SEEDS]


def verify_caches(src_work, manifest_path):
    """Check the frozen M1 training caches before any training. Raises ValueError on:
    a manifest whose training configuration differs from m1's constants; a cache set that is not
    exactly the expected 7 x 3 x 2 files; a missing file; a sha256 mismatch; feature names, order or
    width that differ from the current extractor; malformed targets or blocks.
    Returns {"names", "cache_sha256", "n_rows"}."""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    cfg = manifest.get("config", {})
    want = {"train_presets": list(m1.TRAIN_PRESETS), "train_seeds": list(m1.TRAIN_SEEDS),
            "stride": m1.STRIDE, "thresholds_grid": list(m1.THRESHOLDS)}
    for key, value in want.items():
        if cfg.get(key) != value:
            raise ValueError("manifest {} is {!r}, expected {!r}".format(key, cfg.get(key), value))
    recorded = manifest.get("train_cache_sha256", {})
    expected = expected_caches()
    if set(recorded) != set(expected):
        raise ValueError("manifest cache set differs: missing {}, unexpected {}".format(
            sorted(set(expected) - set(recorded)), sorted(set(recorded) - set(expected))))
    schema = feature_schema()
    n_rows = 0
    for name in expected:
        path = os.path.join(src_work, "train", name)
        if not os.path.isfile(path):
            raise ValueError("missing training cache {}".format(name))
        if sha256_file(path) != recorded[name]:
            raise ValueError("training cache {} differs from the manifest".format(name))
        with np.load(path) as data:
            names = json.loads(str(data["names"]))
            if names != schema:
                raise ValueError("{}: feature names/order differ from the extractor".format(name))
            X, y = data["X"], data["y"]
            if X.ndim != 2 or X.shape[1] != len(schema) or len(y) != len(X) or y.dtype != bool:
                raise ValueError("{}: malformed rows {} / {}".format(name, X.shape, y.shape))
            blocks = json.loads(str(data["blocks"]))
            if not blocks or blocks[-1][2] != len(X) - 1:
                raise ValueError("{}: blocks do not cover the rows".format(name))
            n_rows += len(y)
    return {"names": schema, "cache_sha256": {n: recorded[n] for n in expected}, "n_rows": n_rows}


def _git(*args):
    import subprocess
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _refuse_existing(out_dir):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError("{} exists and is not empty; final artifacts are never overwritten "
                              "(choose a new directory)".format(out_dir))
    real = os.path.realpath(out_dir)
    if real == os.path.realpath(ROOT) or real.startswith(os.path.realpath(ROOT) + os.sep):
        raise ValueError("final training output must be outside the repository")


def train_final(src_work, manifest_path, out_dir):
    """Verify, select the threshold on all seven matches, fit, save. Returns the manifest dict."""
    _refuse_existing(out_dir)
    checked = verify_caches(src_work, manifest_path)
    cfg = config.load(m1.V1_CONFIG)
    train = {m: experiment._series(src_work, m) for m in m1.MATCHES}
    threshold, curve = m1.select_threshold(train, cfg["alive"], fps=FPS / m1.STRIDE)
    X = np.concatenate([s["X"] for items in train.values() for s in items])
    y = np.concatenate([s["y"] for items in train.values() for s in items])
    model = m1.fit(X, y)
    persistence = {k: cfg["alive"][k] for k in PERSISTENCE_KEYS}
    names = checked["names"]
    bundle = {"model": model, "threshold": threshold, "curve": curve, "kind": "final",
              "train": list(m1.MATCHES), "test": [], "features": names, "n_features": len(names),
              "fps": FPS, "stride": m1.STRIDE, "persistence": persistence,
              "train_presets": list(m1.TRAIN_PRESETS), "train_seeds": list(m1.TRAIN_SEEDS),
              "n_rows": int(len(y)), "versions": versions()}
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, MODEL_FILE)
    joblib.dump(bundle, model_path)
    model_sha = sha256_file(model_path)

    # sanity only: training-set D F1 per match after persistence (not generalization evidence)
    sanity = {}
    for match, items in train.items():
        scores = [m1.d_f1(m1.persist(model.predict_proba(s["X"])[:, 1], s["blocks"], cfg["alive"],
                                     FPS / m1.STRIDE, threshold), s["y"], s["blocks"])
                  for s in items]
        sanity[match] = float(np.mean(scores))
    v1_cfg = config.load(m1.V1_CONFIG)
    release = {
        "schema": "pitchvision-v2-release/1",
        "v1_config": "configs/v1.json", "v1_config_sha256": config.sha256(v1_cfg),
        "model": "models/" + MODEL_FILE, "model_sha256": model_sha,
        "threshold": threshold, "fps": FPS, "n_features": len(names),
        "feature_names_sha256": names_sha256(names), "persistence": persistence,
        "dependencies": versions(),
    }
    manifest = {
        "kind": "final M1 deployment model",
        "model_file": MODEL_FILE, "model_sha256": model_sha,
        "threshold": threshold, "threshold_curve": {str(k): v for k, v in curve.items()},
        "threshold_rule": "m1.select_threshold: inner leave-one-match-out over the 7 training "
                          "matches, mean per-match D F1, ties to 0.5; used as returned",
        "training_matches": list(m1.MATCHES), "train_presets": list(m1.TRAIN_PRESETS),
        "train_seeds": list(m1.TRAIN_SEEDS), "stride": m1.STRIDE, "n_rows": int(len(y)),
        "n_rows_checked": checked["n_rows"],
        "features": names, "feature_names_sha256": names_sha256(names),
        "persistence": persistence, "fps": FPS,
        "model": "StandardScaler + LogisticRegression(C=1, l1_ratio=0, lbfgs, max_iter=2000, "
                 "random_state=0)",
        "source_manifest": os.path.abspath(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "training_cache_sha256": checked["cache_sha256"],
        "code_sha256": {p: sha256_file(os.path.join(ROOT, p)) for p in SOURCES},
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "dependencies": versions(), "python": platform.python_version(),
        "platform": platform.platform(),
        "training_set_d_f1_sanity_only": sanity,
        "sanity_note": "Training-set scores; implementation check only, not generalization "
                       "evidence and never used for selection.",
        "release_config": release,
        "attribution": _attribution(),
    }
    with open(os.path.join(out_dir, MANIFEST_FILE), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, RELEASE_CONFIG_FILE), "w") as handle:
        json.dump(release, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _attribution():
    from pvext import idsse
    return idsse.ATTRIBUTION
