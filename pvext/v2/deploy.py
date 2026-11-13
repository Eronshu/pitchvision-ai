"""Load and check the released final M1 model, then label one match (scripts/run_v2.py).

Pipeline: competition ATD -> features.compute -> final M1 probabilities -> m1.persist (V1's
persistence) -> dead mask -> frozen V1 possession machine (m1.run_with_dead) -> H/A/D/X.

Two validation stages, both raising ReleaseError:
  load_release()    BEFORE any competition data is read: release config, frozen V1 config and its
                    hash, persistence and FPS, installed dependency versions, model file sha256
                    (checked before the pickle is loaded: only our own verified artifact is ever
                    unpickled), then the bundle's type, metadata, threshold and feature schema
  label()           AFTER feature extraction: feature names and order, width, finiteness; then the
                    prediction's length, label set and X outside the graded blocks
"""

import collections
import hashlib
import json
import os
from importlib import metadata

import numpy as np

from pitchvision import config
from pvext.v2 import features, m1

ROOT = m1.ROOT
SCHEMA = "pitchvision-v2-release/1"
REQUIRED = ("schema", "v1_config", "v1_config_sha256", "model", "model_sha256", "threshold", "fps",
            "n_features", "feature_names_sha256", "persistence", "dependencies")
BUNDLE_KEYS = ("model", "threshold", "kind", "train", "test", "features", "n_features", "fps",
               "persistence", "versions")
DEPENDENCIES = ("numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl")   # = final.DEPENDENCIES
PERSISTENCE_KEYS = ("smooth_s", "min_dead_s", "min_alive_s")
LABELS = ("H", "A", "D", "X")

Release = collections.namedtuple("Release", ["config", "v1", "bundle", "model_path"])


class ReleaseError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names):
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def _path(value):
    return value if os.path.isabs(value) else os.path.join(ROOT, value)


def load_release(config_path, fps):
    """Pre-data validation; returns a Release. `fps` is the competition's (games.GAMES)."""
    try:
        with open(config_path) as handle:
            rel = json.load(handle)
    except (OSError, ValueError) as error:
        raise ReleaseError("cannot read release config {}: {}".format(config_path, error))
    missing = [k for k in REQUIRED if k not in rel]
    if missing or rel.get("schema") != SCHEMA:
        raise ReleaseError("release config: missing {} or schema {!r} != {!r}".format(
            missing, rel.get("schema"), SCHEMA))

    v1 = config.load(_path(rel["v1_config"]))
    if config.sha256(v1) != rel["v1_config_sha256"]:
        raise ReleaseError("V1 config hash differs from the release config")
    if any(v1["tagging"].values()):
        raise ReleaseError("V2 does not support V1's tagging flags; they must be off")
    want = {k: v1["alive"][k] for k in PERSISTENCE_KEYS}
    if rel["persistence"] != want:
        raise ReleaseError("persistence {} differs from V1's {}".format(rel["persistence"], want))
    if rel["fps"] != fps:
        raise ReleaseError("model fps {} differs from the competition fps {}".format(rel["fps"], fps))
    deps = rel["dependencies"]
    if not isinstance(deps, dict) or set(deps) != set(DEPENDENCIES):
        raise ReleaseError("dependency set must be exactly {}; missing {}, unexpected {}".format(
            sorted(DEPENDENCIES), sorted(set(DEPENDENCIES) - set(deps or {})),
            sorted(set(deps or {}) - set(DEPENDENCIES))))
    for name, version in deps.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            raise ReleaseError("dependency {} is {}, the release needs {}".format(
                name, installed, version))

    model_path = _path(rel["model"])
    if not os.path.isfile(model_path):
        raise ReleaseError("model file not found: {}".format(model_path))
    if _sha256(model_path) != rel["model_sha256"]:
        raise ReleaseError("model sha256 differs from the release config; refusing to load it")
    import joblib
    try:
        bundle = joblib.load(model_path)          # only after the hash check above
    except Exception as error:                   # noqa: BLE001 -- any unpickling failure
        raise ReleaseError("model file cannot be loaded: {}".format(error))
    _check_bundle(bundle, rel)
    return Release(config=rel, v1=v1, bundle=bundle, model_path=model_path)


def _check_bundle(bundle, rel):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if not isinstance(bundle, dict) or any(k not in bundle for k in BUNDLE_KEYS):
        raise ReleaseError("model bundle lacks required metadata")
    model = bundle["model"]
    steps = getattr(model, "steps", None)
    if (not isinstance(model, Pipeline) or len(steps) != 2
            or not isinstance(steps[0][1], StandardScaler)
            or not isinstance(steps[1][1], LogisticRegression)):
        raise ReleaseError("model is not StandardScaler -> LogisticRegression")
    if bundle["kind"] != "final" or list(bundle["train"]) != list(m1.MATCHES) or bundle["test"]:
        raise ReleaseError("bundle is not the final all-matches model")
    names = list(bundle["features"])
    n = rel["n_features"]
    if len(names) != n or bundle["n_features"] != n or getattr(model, "n_features_in_", None) != n:
        raise ReleaseError("feature count mismatch: config {}, bundle {}, model {}".format(
            n, len(names), getattr(model, "n_features_in_", None)))
    if names_sha256(names) != rel["feature_names_sha256"]:
        raise ReleaseError("bundle feature names differ from the release config")
    if bundle["threshold"] != rel["threshold"] or rel["threshold"] not in m1.THRESHOLDS:
        raise ReleaseError("threshold mismatch: config {}, bundle {}".format(
            rel["threshold"], bundle["threshold"]))
    if bundle["fps"] != rel["fps"] or bundle["persistence"] != rel["persistence"]:
        raise ReleaseError("bundle fps/persistence differ from the release config")
    if bundle["versions"] != rel["dependencies"]:
        raise ReleaseError("dependency versions recorded in the model bundle {} differ from the "
                           "release config {}".format(bundle["versions"], rel["dependencies"]))
    classes = getattr(steps[1][1], "classes_", None)
    if classes is None or list(classes) != [False, True]:
        raise ReleaseError("model classes are not [False, True]")


def label(release, tr, pitch, blocks):
    """Post-feature validation and labelling. Returns (labels, info)."""
    rel, bundle, v1 = release.config, release.bundle, release.v1
    fps = rel["fps"]
    names, X = features.compute(tr, pitch, blocks, fps)
    if names != list(bundle["features"]):
        raise ReleaseError("extracted feature names/order differ from the model's training schema")
    if X.ndim != 2 or X.shape != (len(tr.ball_xy), rel["n_features"]):
        raise ReleaseError("feature matrix shape {} is wrong".format(X.shape))
    if not np.isfinite(X).all():
        raise ReleaseError("non-finite feature values")
    prob = bundle["model"].predict_proba(X)[:, 1]
    dead = m1.persist(prob, blocks, v1["alive"], fps, rel["threshold"])
    labels = m1.run_with_dead(tr, v1, fps, pitch, blocks, dead)
    n = len(tr.ball_xy)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    if labels.shape != (n,) or not set(np.unique(labels)) <= set(LABELS):
        raise ReleaseError("prediction has the wrong length or labels")
    if (labels[~graded] != "X").any():
        raise ReleaseError("prediction outside the graded blocks is not X")
    info = {"decision_margin_min": decision_margin(prob, blocks, v1["alive"], fps, rel["threshold"]),
            "dead_share_graded": float(np.mean(dead[graded])) if graded.any() else None}
    return labels, info


def decision_margin(prob, blocks, cfg_alive, fps, threshold):
    """Smallest |smoothed probability - threshold| over graded frames: how far the closest decision
    is from flipping (a guard for cross-platform floating-point differences)."""
    window = max(1, int(round(cfg_alive["smooth_s"] * fps)))
    margins = [np.min(np.abs(m1._centred_mean(np.asarray(prob[a:b + 1], np.float64), window)
                             - threshold)) for _, a, b in blocks if b >= a]
    return float(min(margins)) if margins else None
