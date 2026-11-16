"""PitchVision AI release C: HGB dead-ball model + HA5a H/A model + joint three-state decoding.
Two validation stages, as in pvext/ha/deploy.py.

  load_release()  before any match data: RELEASE.lock (sha256 of the config and of both models,
                  checked before the config is trusted and before anything is unpickled), config
                  schema, Python and dependency versions, hashes of every pinned source file
                  (run script, games.py and all prediction code), bundle types and schemas
  label()         after feature extraction: names/order, shape, finiteness, label alphabet,
                  X exactly outside the graded blocks
"""

import collections
import hashlib
import json
import os
from importlib import metadata

import numpy as np

from astra5.s2_joint import decode as jd
from pvext.ha5 import features as f5
from pvext.v2 import features as m1_features
from pvext.v2 import m1

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = "pitchvision-release-c/2"
LOCK = "RELEASE.lock"
PYTHON = "3.13"
FPS = 25
DEPENDENCIES = ("joblib", "numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")
SOURCES = (
    "astra5/release/deploy.py",
    "astra5/s2_joint/decode.py",
    "configs/v1.json",
    "evaluation/possession_metrics.py",
    "games.py",
    "pitchvision/__init__.py",
    "pitchvision/alive.py",
    "pitchvision/atd.py",
    "pitchvision/config.py",
    "pitchvision/features.py",
    "pitchvision/possession.py",
    "pvext/__init__.py",
    "pvext/ha/features.py",
    "pvext/ha5/features.py",
    "pvext/ha5/model.py",
    "pvext/v2/features.py",
    "pvext/v2/m1.py",
    "requirements-release.txt",
    "scripts/run_astra5_combo.py",
    "starter/baseline.py",
)
REQUIRED = ("schema", "status", "dead_model", "ha_model", "joint", "fps", "python", "dependencies", "sources")
LABELS = ("H", "A", "D", "X")
DEV = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")

Release = collections.namedtuple("Release", ["config", "dead", "ha", "base"])


class ReleaseError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def names_sha256(names):
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def installed_versions():
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_hashes():
    return {p: sha256(os.path.join(ROOT, p)) for p in SOURCES}


def _load(base, entry, what):
    path = os.path.join(base, entry["model"])
    if not os.path.isfile(path):
        raise ReleaseError("{} file not found: {}".format(what, path))
    if sha256(path) != entry["model_sha256"]:
        raise ReleaseError("{} sha256 differs from the release config; refusing to load it".format(what))
    import joblib
    try:
        return joblib.load(path)
    except Exception as error:      # noqa: BLE001
        raise ReleaseError("{} cannot be loaded: {}".format(what, error))


def python_version():
    import sys
    return "{}.{}".format(*sys.version_info[:2])


def verify_lock(base, config_path):
    """RELEASE.lock lists the sha256 of the config and of both model files. It is checked first,
    so an edited config (decoding costs included) or model is refused before it is used."""
    lock_path = os.path.join(base, LOCK)
    try:
        want = dict(line.split(None, 1)[::-1] for line in open(lock_path).read().splitlines() if line.strip())
    except (OSError, ValueError) as error:
        raise ReleaseError("cannot read {}: {}".format(lock_path, error))
    want = {k.strip(): v for k, v in want.items()}
    name = os.path.basename(config_path)
    if name not in want or len(want) != 3:
        raise ReleaseError("{} must list exactly the config and the two model files".format(LOCK))
    for rel_path, digest in sorted(want.items()):
        path = os.path.join(base, rel_path)
        if not os.path.isfile(path) or sha256(path) != digest:
            raise ReleaseError("{} sha256 differs from {}; refusing to use it".format(rel_path, LOCK))
    return want


def load_release(config_path, fps):
    base = os.path.dirname(os.path.abspath(config_path))
    lock = verify_lock(base, config_path)
    try:
        with open(config_path) as handle:
            rel = json.load(handle)
    except (OSError, ValueError) as error:
        raise ReleaseError("cannot read release config {}: {}".format(config_path, error))
    missing = [k for k in REQUIRED if k not in rel]
    if missing or rel.get("schema") != SCHEMA:
        raise ReleaseError("release config: missing {} or schema {!r}".format(missing, rel.get("schema")))
    if rel["fps"] != FPS or fps != FPS:
        raise ReleaseError("the release is validated at {} fps only".format(FPS))
    if rel["python"] != PYTHON or python_version() != PYTHON:
        raise ReleaseError("Python is {}, the release needs {}".format(python_version(), PYTHON))
    for entry in (rel["dead_model"], rel["ha_model"]):
        if lock.get(entry["model"]) != entry["model_sha256"]:
            raise ReleaseError("config and {} disagree on {}".format(LOCK, entry["model"]))
    installed = installed_versions()
    if set(rel["dependencies"]) != set(DEPENDENCIES):
        raise ReleaseError("dependency set must be exactly {}".format(sorted(DEPENDENCIES)))
    for name, version in rel["dependencies"].items():
        if installed[name] != version:
            raise ReleaseError("dependency {} is {}, the release needs {}".format(name, installed[name], version))
    now = source_hashes()
    if set(rel["sources"]) != set(SOURCES):
        raise ReleaseError("sources must list exactly {}".format(list(SOURCES)))
    changed = [p for p in SOURCES if now[p] != rel["sources"][p]]
    if changed:
        raise ReleaseError("prediction code differs from the frozen release: {}".format(changed))
    j = rel["joint"]
    if not all(isinstance(j.get(k), (int, float)) and j[k] > 0 for k in ("lam_ha", "lam_on", "lam_off")) \
            or j["lam_on"] + j["lam_off"] < j["lam_ha"]:
        raise ReleaseError("invalid joint decoding costs {}".format(j))
    dead = _load(base, rel["dead_model"], "dead-ball model")
    ha = _load(base, rel["ha_model"], "H/A model")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from pvext.ha5 import model as m5
    if not isinstance(dead, dict) or not isinstance(dead.get("model"), HistGradientBoostingClassifier):
        raise ReleaseError("dead-ball model is not a HistGradientBoostingClassifier bundle")
    if not isinstance(ha, dict) or not isinstance(ha.get("model"), m5.MixedSymmetricGB) \
            or not isinstance(ha["model"].model, HistGradientBoostingClassifier):
        raise ReleaseError("H/A model is not the mixed symmetric gradient boosting bundle")
    for what, bundle, entry in (("dead-ball", dead, rel["dead_model"]), ("H/A", ha, rel["ha_model"])):
        if sorted(bundle.get("train", ())) != list(DEV) or sorted(entry["train"]) != list(DEV):
            raise ReleaseError("{} model was not trained on exactly the five IDSSE dev matches".format(what))
        names = list(bundle["features"])
        inner = bundle["model"] if what == "dead-ball" else bundle["model"].model
        if len(names) != entry["n_features"] or getattr(inner, "n_features_in_", None) != entry["n_features"] \
                or names_sha256(names) != entry["feature_names_sha256"]:
            raise ReleaseError("{} feature schema differs from the release config".format(what))
        if list(inner.classes_) != [False, True]:
            raise ReleaseError("{} model classes are not [False, True]".format(what))
    if not np.array_equal(ha["anti"], f5.anti_mask()) or not np.array_equal(ha["model"].anti, f5.anti_mask()):
        raise ReleaseError("H/A antisymmetry mask differs from the HA5a extractor's")
    return Release(config=rel, dead=dead, ha=ha, base=base)


def probabilities(release, tr, pitch, blocks):
    """(p_dead, p_home) with the post-feature checks."""
    rel = release.config
    n = len(tr.ball_xy)
    names, Xd = m1_features.compute(tr, pitch, blocks, rel["fps"])
    if names != list(release.dead["features"]):
        raise ReleaseError("extracted dead-ball feature names/order differ from the model's schema")
    names, Xh, anti = f5.compute(tr, pitch, blocks, rel["fps"])
    if names != list(release.ha["features"]) or not np.array_equal(anti, release.ha["anti"]):
        raise ReleaseError("extracted H/A feature names/order differ from the model's schema")
    for what, X, entry in (("dead-ball", Xd, rel["dead_model"]), ("H/A", Xh, rel["ha_model"])):
        if X.ndim != 2 or X.shape != (n, entry["n_features"]):
            raise ReleaseError("{} feature matrix shape {} is wrong".format(what, X.shape))
        if not np.isfinite(X).all():
            raise ReleaseError("non-finite {} feature values".format(what))
    d = release.dead["model"].predict_proba(Xd)[:, 1]
    h = release.ha["model"].predict_proba(Xh)[:, 1]
    if not (np.isfinite(d).all() and np.isfinite(h).all()):
        raise ReleaseError("non-finite probabilities")
    return d, h, Xd, names


def label(release, tr, pitch, blocks):
    """H/A/D/X labels and (p_dead, p_home)."""
    j = release.config["joint"]
    d, h, _, _ = probabilities(release, tr, pitch, blocks)
    labels = jd.decode(d, h, blocks, j["lam_ha"], j["lam_on"], j["lam_off"], release.config["fps"])
    n = len(tr.ball_xy)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    if labels.shape != (n,) or not set(np.unique(labels)) <= set(LABELS):
        raise ReleaseError("prediction has the wrong length or labels")
    if (labels[~graded] != "X").any() or (labels[graded] == "X").any():
        raise ReleaseError("X frames are not exactly the frames outside the graded blocks")
    return labels, d, h
