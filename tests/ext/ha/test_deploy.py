"""HA3 release validation (before data) and labelling checks (after features). Synthetic only:
a synthetic M1 release (tests/ext/v2/release_fixture) and an HA3 bundle fitted on synthetic
tracking. No IDSSE, lockbox or competition data."""

import json
import os

import joblib
import numpy as np
import pytest

from pvext.ha import deploy, features
from pvext.ha import experiment as hx
from tests.ext.v2 import release_fixture as rf
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking

FPS = 25


def _bundle(**changes):
    tr = tracking(seed=7)
    names, X = features.compute(tr, PITCH, BLOCKS, FPS, feature_set="ha3")
    y = (np.arange(len(X)) // 150) % 2 == 0
    b = {"model": hx.fit_model(X, y, "gb_sym"), "lam": 0.5, "features": names, "variant": "ha3",
         "feature_set": "ha3", "model_kind": "gb_sym", "train": ["J03WPY", "J03WOH"],
         "test": ["J03WMX"], "kind": "final_dev"}
    b.update(changes)
    return b


def _dead_bundle(m1_config, **changes):
    """A synthetic stand-in for the league dead-ball model: the synthetic M1 logistic regression
    with the real league bundle's metadata (kind league, dev train, lockbox test)."""
    with open(m1_config) as handle:
        final = joblib.load(json.load(handle)["model"])
    b = {"model": final["model"], "threshold": final["threshold"], "kind": "league",
         "train": list(hx.DEV), "test": list(hx.LOCKBOX), "n_rows": 1}
    b.update(changes)
    return b


def build_ha3_release(base):
    """Synthetic M1 release + synthetic dead-ball and HA3 bundles + HA3 release config; returns
    its path."""
    m1_config, _ = rf.build_release(base)
    dead = os.path.join(str(base), "dead.joblib")
    joblib.dump(_dead_bundle(m1_config), dead)
    model = os.path.join(str(base), "ha3.joblib")
    joblib.dump(_bundle(), model)
    cfg = deploy.make_release_config(model, m1_config, dead)
    path = os.path.join(str(base), "ha3.json")
    with open(path, "w") as handle:
        json.dump(cfg, handle)
    return path


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return build_ha3_release(tmp_path_factory.mktemp("ha3release"))


def _edit(release, tmp_path, **changes):
    with open(release) as handle:
        cfg = json.load(handle)
    cfg.update(changes)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _with_bundle(release, tmp_path, bundle):
    model = str(tmp_path / "other.joblib")
    joblib.dump(bundle, model)
    return _edit(release, tmp_path, model=model, model_sha256=rf.sha(model))


def test_schema_is_the_frozen_26_features():
    names = deploy.feature_schema()
    assert len(names) == 26 and names[-5:] == ["last_touch_team", "last_touch_decayed",
                                               "next_touch_team", "next_touch_decayed",
                                               "touch_balance_3s"]


def test_valid_release_labels_only_alive_frames(release):
    rel = deploy.load_release(release, FPS)
    tr = tracking(seed=3, ball_missing=(400, 500))
    labels, info, m1_labels = deploy.label(rel, tr, PITCH, BLOCKS)
    assert labels.shape == (len(tr.ball_xy),) and set(np.unique(labels)) <= {"H", "A", "D", "X"}
    assert np.array_equal(labels == "D", m1_labels == "D")
    assert np.array_equal(labels == "X", m1_labels == "X")
    assert info["ha3_lam"] == 0.5 and info["n_spells_ha3"] >= 1
    assert 0 <= info["ha3_changed_share_alive"] <= 1


def test_labelling_is_deterministic_and_survives_a_reload(release, tmp_path):
    rel = deploy.load_release(release, FPS)
    tr = tracking(seed=5)
    a = deploy.label(rel, tr, PITCH, BLOCKS)[0]
    b = deploy.label(deploy.load_release(release, FPS), tr, PITCH, BLOCKS)[0]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("change, fragment", [
    ({"schema": "other"}, "schema"),
    ({"variant": "ha2"}, "frozen HA3"),
    ({"feature_set": "ha1"}, "frozen HA3"),
    ({"fps": 30}, "25 fps"),
    ({"lam": 0.3}, "grid"),
    ({"frozen_sources": {"pvext/ha/features.py": "0" * 64, "pvext/ha/decode.py": "0" * 64,
                         "pvext/ha/experiment.py": "0" * 64}}, "differs from the frozen"),
    ({"frozen_sources": {"pvext/ha/features.py": "0" * 64}}, "exactly"),
    ({"dependencies": {"numpy": "0.0"}}, "dependency set"),
    ({"model_sha256": "0" * 64}, "sha256"),
    ({"model": "/nonexistent/ha3.joblib"}, "not found"),
    ({"n_features": 25}, "feature count"),
    ({"feature_names_sha256": "0" * 64}, "feature names"),
    ({"m1_release_config": "/nonexistent/v2.json"}, "M1 release"),
])
def test_release_config_problems_are_rejected(release, tmp_path, change, fragment):
    with pytest.raises(deploy.ReleaseError, match=fragment):
        deploy.load_release(_edit(release, tmp_path, **change), FPS)


def test_competition_fps_must_be_25(release):
    with pytest.raises(deploy.ReleaseError, match="25 fps"):
        deploy.load_release(release, 30)


def test_installed_dependency_version_must_match(release, tmp_path):
    with open(release) as handle:
        deps = json.load(handle)["dependencies"]
    deps["numpy"] = "0.0.1"
    with pytest.raises(deploy.ReleaseError, match="numpy"):
        deploy.load_release(_edit(release, tmp_path, dependencies=deps), FPS)


def test_wrong_hash_never_unpickles(release, tmp_path, monkeypatch):
    calls = []
    real = joblib.load

    def spy(path, *a, **k):
        calls.append(str(path))
        return real(path, *a, **k)
    monkeypatch.setattr(joblib, "load", spy)
    with pytest.raises(deploy.ReleaseError, match="sha256"):
        deploy.load_release(_edit(release, tmp_path, model_sha256="0" * 64), FPS)
    assert calls and not [c for c in calls if c.endswith("ha3.joblib")]   # M1 loaded, HA3 never


@pytest.mark.parametrize("change, fragment", [
    ({"model": hx.fit(np.random.default_rng(0).normal(size=(100, 26)), np.arange(100) % 2 == 0)},
     "symmetric"),
    ({"variant": "ha2"}, "not an HA3"),
    ({"feature_set": "ha1"}, "not an HA3"),
    ({"model_kind": "linear"}, "not an HA3"),
    ({"train": ["SOMEWHERE"]}, "unknown matches"),
    ({"lam": 0.25}, "lambda mismatch"),
])
def test_inconsistent_bundles_are_rejected(release, tmp_path, change, fragment):
    path = _with_bundle(release, tmp_path, _bundle(**change))
    with pytest.raises(deploy.ReleaseError, match=fragment):
        deploy.load_release(path, FPS)


def test_bundle_missing_metadata(release, tmp_path):
    b = _bundle()
    del b["feature_set"]
    with pytest.raises(deploy.ReleaseError, match="metadata"):
        deploy.load_release(_with_bundle(release, tmp_path, b), FPS)


def test_feature_order_mismatch_is_caught_after_extraction(release, monkeypatch):
    rel = deploy.load_release(release, FPS)
    real = features.compute

    def reversed_names(*a, **k):
        names, X = real(*a, **k)
        return list(reversed(names)), X[:, ::-1]
    monkeypatch.setattr(deploy.ha_features, "compute", reversed_names)
    with pytest.raises(deploy.ReleaseError, match="names/order"):
        deploy.label(rel, tracking(seed=1), PITCH, BLOCKS)


def test_non_finite_features_are_rejected(release, monkeypatch):
    rel = deploy.load_release(release, FPS)
    real = features.compute

    def nan_features(*a, **k):
        names, X = real(*a, **k)
        X = X.copy()
        X[10, 0] = np.nan
        return names, X
    monkeypatch.setattr(deploy.ha_features, "compute", nan_features)
    with pytest.raises(deploy.ReleaseError, match="non-finite"):
        deploy.label(rel, tracking(seed=1), PITCH, BLOCKS)


def _with_dead(release, tmp_path, **changes):
    with open(release) as handle:
        cfg = json.load(handle)
    dead = str(tmp_path / "dead_other.joblib")
    joblib.dump(_dead_bundle(cfg["m1_release_config"], **changes), dead)
    entry = dict(cfg["dead_model"], model=dead, model_sha256=rf.sha(dead))
    for key in ("kind", "train", "test", "threshold"):
        if key in changes:
            entry[key] = changes[key]
    return _edit(release, tmp_path, dead_model=entry)


@pytest.mark.parametrize("change, fragment", [
    ({"kind": "final"}, "league model"),
    ({"train": list(hx.DEV) + list(hx.LOCKBOX)}, "league model"),
    ({"test": []}, "league model"),
    ({"threshold": 0.45}, "threshold"),
    ({"model": "not a pipeline"}, "LogisticRegression"),
])
def test_dead_model_must_be_the_lockbox_league_model(release, tmp_path, change, fragment):
    with pytest.raises(deploy.ReleaseError, match=fragment):
        deploy.load_release(_with_dead(release, tmp_path, **change), FPS)


def test_dead_model_config_provenance_must_match_the_bundle(release, tmp_path):
    with open(release) as handle:
        entry = json.load(handle)["dead_model"]
    entry["train"] = ["J03WPY"]
    with pytest.raises(deploy.ReleaseError, match="provenance"):
        deploy.load_release(_edit(release, tmp_path, dead_model=entry), FPS)


def test_dead_model_entry_is_required(release, tmp_path):
    with pytest.raises(deploy.ReleaseError, match="dead_model"):
        deploy.load_release(_edit(release, tmp_path, dead_model={"model": "x"}), FPS)


def test_dead_model_wrong_hash_never_unpickles(release, tmp_path, monkeypatch):
    with open(release) as handle:
        entry = json.load(handle)["dead_model"]
    entry["model_sha256"] = "0" * 64
    calls = []
    real = joblib.load

    def spy(path, *a, **k):
        calls.append(str(path))
        return real(path, *a, **k)
    monkeypatch.setattr(joblib, "load", spy)
    with pytest.raises(deploy.ReleaseError, match="dead-ball model sha256"):
        deploy.load_release(_edit(release, tmp_path, dead_model=entry), FPS)
    assert not [c for c in calls if c.endswith("dead.joblib") or c.endswith("ha3.joblib")]


class _AllDead:
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_labels_use_the_dead_model_not_m1_final(release):
    rel = deploy.load_release(release, FPS)
    rel = rel._replace(dead=dict(rel.dead, model=_AllDead()))
    labels, _, m1_labels = deploy.label(rel, tracking(seed=4), PITCH, BLOCKS)
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in BLOCKS:
        graded[first:last + 1] = True
    assert (labels[graded] == "D").all() and (m1_labels[graded] == "D").all()


def test_team_swap_swaps_h_and_a_under_the_same_dead_mask(release):
    from pvext.ha import decode
    rel = deploy.load_release(release, FPS)
    tr = tracking(seed=2)
    sw = tr._replace(team_of_slot=(1 - tr.team_of_slot).astype(np.int8))
    _, _, m1_labels = deploy.label(rel, tr, PITCH, BLOCKS)
    dead = m1_labels == "D"
    model = rel.bundle["model"]
    p = model.predict_proba(features.compute(tr, PITCH, BLOCKS, FPS, feature_set="ha3")[1])[:, 1]
    q = model.predict_proba(features.compute(sw, PITCH, BLOCKS, FPS, feature_set="ha3")[1])[:, 1]
    assert np.allclose(q, 1 - p, atol=1e-6)
    a = decode.decode(p, dead, BLOCKS, rel.config["lam"], FPS)
    b = decode.decode(1 - p, dead, BLOCKS, rel.config["lam"], FPS)
    flip = np.array([{"H": "A", "A": "H"}.get(x, x) for x in a])
    assert np.array_equal(b, flip) and np.isin(a, ["H", "A"]).any()
