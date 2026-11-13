"""Load and check a frozen HA3 release, then label one match: the pipeline evaluated on the lockbox.

Pipeline: competition ATD -> M1 features (pvext/v2/features) -> the M1 dead-ball model the HA3
lockbox was evaluated with (the five-dev-match "league" model, NOT the all-seven M1 final) -> V1
persistence -> dead mask -> frozen V1 possession machine (= M1 labels) -> HA3 features (feature set
"ha3") -> symmetric gradient boosting P(home) -> switch-penalty Viterbi per alive segment (lambda
from the release) -> H/A/D/X. This is exactly ha_experiment.evaluate_match for the lockbox.

HA3 only re-labels the alive frames: the D mask and the X frames are those of M1 with the HA3 dead
model, unchanged (checked). The M1 release (configs/v2.json, M1 final) is the fallback submission;
it is loaded and fully checked here too, and supplies the frozen V1 config, persistence, dependency
set and M1 feature schema, but its model is not used for the HA3 labels.

Two validation stages, both raising ReleaseError, as in pvext/v2/deploy.py:
  load_release()  BEFORE any competition data is read: the HA3 release config, the frozen HA3
                  prediction sources (byte-identical to the frozen candidate), the M1 release
                  (pvext/v2/deploy.load_release), fps, dependency versions, the dead-ball model and
                  the HA3 model (each sha256 checked before unpickling), then each bundle's type,
                  provenance (dead model: league, trained on the dev matches only) and schema
  label()         AFTER feature extraction: M1 and HA3 feature names/order, width, finiteness; then
                  the prediction's length and labels, X outside the blocks and the D mask unchanged
"""

import collections
import hashlib
import json
import os
from importlib import metadata

import numpy as np

from pvext.ha import decode as ha_decode
from pvext.ha import experiment as hx
from pvext.ha import features as ha_features
from pvext.v2 import deploy as m1_deploy
from pvext.v2 import m1

ROOT = m1.ROOT
SCHEMA = "pitchvision-ha3-release/1"
VARIANT = "ha3"
FEATURE_SET = "ha3"
FPS = 25
REQUIRED = ("schema", "m1_release_config", "dead_model", "model", "model_sha256", "variant",
            "feature_set", "n_features", "feature_names_sha256", "lam", "fps", "dependencies",
            "frozen_sources")
DEAD_REQUIRED = ("model", "model_sha256", "kind", "train", "test", "threshold")
DEAD_BUNDLE_KEYS = ("model", "threshold", "kind", "train", "test")
DEAD_KIND = "league"                 # the dead-ball model of the HA3 lockbox evaluation
BUNDLE_KEYS = ("model", "lam", "features", "variant", "feature_set", "model_kind", "train")
DEPENDENCIES = m1_deploy.DEPENDENCIES
# The prediction code of the frozen candidate: these files must be byte-identical to the freeze.
PREDICTION_SOURCES = ("pvext/ha/features.py", "pvext/ha/decode.py", "pvext/ha/experiment.py")
LABELS = m1_deploy.LABELS

Release = collections.namedtuple("Release", ["config", "m1", "bundle", "model_path", "dead",
                                             "dead_path"])


class ReleaseError(ValueError):
    pass


def _sha256(path):
    return m1_deploy._sha256(path)


def names_sha256(names):
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def _path(value, base=ROOT):
    return value if os.path.isabs(value) else os.path.join(base, value)


def feature_schema():
    """The HA3 feature names, in order, that the current extractor produces (tiny synthetic
    Tracking, so the schema is the code's)."""
    from pitchvision import atd
    T = 60
    players = np.full((T, 4, 2), 30.0, dtype=np.float32)
    players[:, :, 0] += np.arange(4, dtype=np.float32)
    tr = atd.Tracking(players_xy=players, players_speed=np.zeros((T, 4), np.float32),
                      team_of_slot=np.array([0, 1, 0, 1], np.int8),
                      ball_xy=players[:, 0, :] + 0.5, ball_present=np.ones(T, bool))
    names, _ = ha_features.compute(tr, (105.0, 68.0), [(1, 0, T - 1)], FPS, feature_set=FEATURE_SET)
    return names


def installed_versions():
    out = {}
    for name in DEPENDENCIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def make_release_config(model_path, m1_release_config, dead_model_path):
    """The HA3 release config for a frozen HA3 bundle and the dead-ball model it was evaluated
    with (written by the release step, not here)."""
    import joblib
    bundle = joblib.load(model_path)
    dead = joblib.load(dead_model_path)
    names = list(bundle["features"])
    return {"schema": SCHEMA, "m1_release_config": m1_release_config,
            "dead_model": {"model": dead_model_path, "model_sha256": _sha256(dead_model_path),
                           "kind": dead["kind"], "train": list(dead["train"]),
                           "test": list(dead["test"]), "threshold": dead["threshold"]},
            "model": model_path,
            "model_sha256": _sha256(model_path), "variant": bundle["variant"],
            "feature_set": bundle["feature_set"], "n_features": len(names),
            "feature_names_sha256": names_sha256(names), "lam": bundle["lam"], "fps": FPS,
            "dependencies": installed_versions(),
            "frozen_sources": {p: _sha256(os.path.join(ROOT, p)) for p in PREDICTION_SOURCES}}


def load_release(config_path, fps):
    """Pre-data validation; returns a Release. `fps` is the competition's (games.GAMES)."""
    try:
        with open(config_path) as handle:
            rel = json.load(handle)
    except (OSError, ValueError) as error:
        raise ReleaseError("cannot read HA3 release config {}: {}".format(config_path, error))
    missing = [k for k in REQUIRED if k not in rel]
    if missing or rel.get("schema") != SCHEMA:
        raise ReleaseError("HA3 release config: missing {} or schema {!r} != {!r}".format(
            missing, rel.get("schema"), SCHEMA))
    if rel["variant"] != VARIANT or rel["feature_set"] != FEATURE_SET:
        raise ReleaseError("release is not the frozen HA3 candidate ({}/{})".format(
            rel["variant"], rel["feature_set"]))
    if rel["fps"] != FPS or fps != FPS:
        # the touch features use fixed frame windows, validated at 25 fps only
        raise ReleaseError("HA3 is validated at {} fps only (release {}, competition {})".format(
            FPS, rel["fps"], fps))
    frozen = rel["frozen_sources"]
    if not isinstance(frozen, dict) or set(frozen) != set(PREDICTION_SOURCES):
        raise ReleaseError("frozen_sources must list exactly {}".format(list(PREDICTION_SOURCES)))
    changed = [p for p in PREDICTION_SOURCES if _sha256(os.path.join(ROOT, p)) != frozen[p]]
    if changed:
        raise ReleaseError("HA3 prediction code differs from the frozen candidate: {}".format(changed))
    if rel["lam"] not in hx.VARIANTS[VARIANT]["lambdas"]:
        raise ReleaseError("lambda {} is not on the frozen HA3 grid".format(rel["lam"]))
    deps = rel["dependencies"]
    if not isinstance(deps, dict) or set(deps) != set(DEPENDENCIES):
        raise ReleaseError("dependency set must be exactly {}".format(sorted(DEPENDENCIES)))
    installed = installed_versions()
    for name, version in deps.items():
        if installed[name] != version:
            raise ReleaseError("dependency {} is {}, the HA3 release needs {}".format(
                name, installed[name], version))

    try:
        m1_release = m1_deploy.load_release(_path(rel["m1_release_config"]), fps)
    except m1_deploy.ReleaseError as error:
        raise ReleaseError("M1 release check failed: {}".format(error))

    dead_rel = rel["dead_model"]
    if not isinstance(dead_rel, dict) or any(k not in dead_rel for k in DEAD_REQUIRED):
        raise ReleaseError("dead_model must give {}".format(list(DEAD_REQUIRED)))
    dead_path = _path(dead_rel["model"])
    dead = _load_verified(dead_path, dead_rel["model_sha256"], "dead-ball model")
    _check_dead(dead, dead_rel, m1_release)

    model_path = _path(rel["model"])
    bundle = _load_verified(model_path, rel["model_sha256"], "HA3 model")
    _check_bundle(bundle, rel)
    return Release(config=rel, m1=m1_release, bundle=bundle, model_path=model_path, dead=dead,
                   dead_path=dead_path)


def _load_verified(path, sha256, what):
    if not os.path.isfile(path):
        raise ReleaseError("{} file not found: {}".format(what, path))
    if _sha256(path) != sha256:
        raise ReleaseError("{} sha256 differs from the release config; refusing to load it".format(what))
    import joblib
    try:
        return joblib.load(path)                  # only after the hash check above
    except Exception as error:                   # noqa: BLE001 -- any unpickling failure
        raise ReleaseError("{} file cannot be loaded: {}".format(what, error))


def _check_dead(dead, dead_rel, m1_release):
    """The dead-ball model must be the M1 logistic regression the HA3 lockbox used: kind league,
    trained on the dev matches only, held out on the lockbox, at M1's threshold and schema."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if not isinstance(dead, dict) or any(k not in dead for k in DEAD_BUNDLE_KEYS):
        raise ReleaseError("dead-ball model bundle lacks required metadata")
    model = dead["model"]
    steps = getattr(model, "steps", None)
    if (not isinstance(model, Pipeline) or len(steps) != 2
            or not isinstance(steps[0][1], StandardScaler)
            or not isinstance(steps[1][1], LogisticRegression)):
        raise ReleaseError("dead-ball model is not StandardScaler -> LogisticRegression")
    if (dead["kind"] != DEAD_KIND or list(dead["train"]) != list(hx.DEV)
            or list(dead["test"]) != list(hx.LOCKBOX)):
        raise ReleaseError("dead-ball model is not the dev-trained league model of the HA3 lockbox "
                           "({} train {} test {})".format(dead["kind"], dead["train"], dead["test"]))
    if (dead_rel["kind"] != dead["kind"] or list(dead_rel["train"]) != list(dead["train"])
            or list(dead_rel["test"]) != list(dead["test"])):
        raise ReleaseError("dead-ball model provenance differs from the release config")
    threshold = m1_release.config["threshold"]
    if not (dead["threshold"] == dead_rel["threshold"] == threshold) or threshold not in m1.THRESHOLDS:
        raise ReleaseError("dead-ball threshold mismatch: config {}, bundle {}, M1 release {}".format(
            dead_rel["threshold"], dead["threshold"], threshold))
    n = m1_release.config["n_features"]
    if getattr(model, "n_features_in_", None) != n:
        raise ReleaseError("dead-ball model feature count {} differs from M1's {}".format(
            getattr(model, "n_features_in_", None), n))
    classes = getattr(steps[1][1], "classes_", None)
    if classes is None or list(classes) != [False, True]:
        raise ReleaseError("dead-ball model classes are not [False, True]")


def _check_bundle(bundle, rel):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if not isinstance(bundle, dict) or any(k not in bundle for k in BUNDLE_KEYS):
        raise ReleaseError("HA3 bundle lacks required metadata")
    model = bundle["model"]
    if (not isinstance(model, hx.SymmetricGB)
            or not isinstance(model.model, HistGradientBoostingClassifier)):
        raise ReleaseError("HA3 model is not the symmetric gradient boosting classifier")
    if (bundle["variant"] != VARIANT or bundle["feature_set"] != FEATURE_SET
            or bundle["model_kind"] != "gb_sym"):
        raise ReleaseError("bundle is not an HA3 model")
    if any(m not in m1.MATCHES for m in bundle["train"]):
        raise ReleaseError("HA3 bundle trained on unknown matches {}".format(bundle["train"]))
    names = list(bundle["features"])
    n = rel["n_features"]
    if len(names) != n or getattr(model.model, "n_features_in_", None) != n:
        raise ReleaseError("HA3 feature count mismatch: config {}, bundle {}, model {}".format(
            n, len(names), getattr(model.model, "n_features_in_", None)))
    if names_sha256(names) != rel["feature_names_sha256"] or names != feature_schema():
        raise ReleaseError("HA3 bundle feature names differ from the release config / extractor")
    if bundle["lam"] != rel["lam"]:
        raise ReleaseError("lambda mismatch: config {}, bundle {}".format(rel["lam"], bundle["lam"]))
    classes = getattr(model.model, "classes_", None)
    if classes is None or list(classes) != [False, True]:
        raise ReleaseError("HA3 model classes are not [False, True]")


def label(release, tr, pitch, blocks):
    """Post-feature validation and labelling. Returns (labels, info, m1_labels): the HA3
    labels, diagnostics, and the M1 labels they were derived from (same D and X frames)."""
    rel, bundle = release.config, release.bundle
    m1_labels, m1_info, dead = _m1_with_dead_model(release, tr, pitch, blocks)
    names, X = ha_features.compute(tr, pitch, blocks, rel["fps"], feature_set=rel["feature_set"])
    if names != list(bundle["features"]):
        raise ReleaseError("extracted HA3 feature names/order differ from the model's schema")
    if X.ndim != 2 or X.shape != (len(tr.ball_xy), rel["n_features"]):
        raise ReleaseError("HA3 feature matrix shape {} is wrong".format(X.shape))
    if not np.isfinite(X).all():
        raise ReleaseError("non-finite HA3 feature values")
    p = bundle["model"].predict_proba(X)[:, 1]
    if not np.isfinite(p).all():
        raise ReleaseError("non-finite HA3 probabilities")
    labels = ha_decode.decode(p, dead, blocks, rel["lam"], rel["fps"])
    n = len(tr.ball_xy)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    if labels.shape != (n,) or not set(np.unique(labels)) <= set(LABELS):
        raise ReleaseError("HA3 prediction has the wrong length or labels")
    if (labels[~graded] != "X").any():
        raise ReleaseError("HA3 prediction outside the graded blocks is not X")
    if not np.array_equal(labels == "D", m1_labels == "D"):
        raise ReleaseError("HA3 changed the M1 dead mask")
    alive = graded & ~dead
    info = dict(m1_info)
    info.update({
        "ha3_lam": rel["lam"],
        "ha3_changed_share_alive": float(np.mean(labels[alive] != m1_labels[alive])) if alive.any() else None,
        "ha3_p_margin_min": float(np.min(np.abs(p[alive] - 0.5))) if alive.any() else None,
        "n_spells_m1": _spells(m1_labels, blocks), "n_spells_ha3": _spells(labels, blocks)})
    return labels, info, m1_labels


def _m1_with_dead_model(release, tr, pitch, blocks):
    """M1 labels with the HA3 dead-ball model, as in the lockbox evaluation; also the raw dead
    mask that HA3 decodes with. Feature schema checks as in pvext/v2/deploy.label."""
    m1_rel, v1 = release.m1.config, release.m1.v1
    fps = m1_rel["fps"]
    names, X = m1_deploy.features.compute(tr, pitch, blocks, fps)
    if names != list(release.m1.bundle["features"]) or \
            m1_deploy.names_sha256(names) != m1_rel["feature_names_sha256"]:
        raise ReleaseError("extracted M1 feature names/order differ from the M1 schema")
    if X.ndim != 2 or X.shape != (len(tr.ball_xy), m1_rel["n_features"]):
        raise ReleaseError("M1 feature matrix shape {} is wrong".format(X.shape))
    if not np.isfinite(X).all():
        raise ReleaseError("non-finite M1 feature values")
    prob = release.dead["model"].predict_proba(X)[:, 1]
    threshold = release.dead["threshold"]
    dead = m1.persist(prob, blocks, v1["alive"], fps, threshold)
    labels = m1.run_with_dead(tr, v1, fps, pitch, blocks, dead)
    n = len(tr.ball_xy)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    if labels.shape != (n,) or not set(np.unique(labels)) <= set(LABELS):
        raise ReleaseError("M1 prediction has the wrong length or labels")
    if (labels[~graded] != "X").any():
        raise ReleaseError("M1 prediction outside the graded blocks is not X")
    if not np.array_equal(labels[graded] == "D", dead[graded]):
        raise ReleaseError("M1 D frames differ from the dead mask")
    info = {"decision_margin_min": m1_deploy.decision_margin(prob, blocks, v1["alive"], fps, threshold),
            "dead_share_graded": float(np.mean(dead[graded])) if graded.any() else None}
    return labels, info, dead


def _spells(labels, blocks):
    n = 0
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        ha = np.isin(seg, ["H", "A"])
        starts = ha & np.r_[True, seg[1:] != seg[:-1]]
        n += int(starts.sum())
    return n
