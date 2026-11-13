"""Release validation (before data) and labelling checks (after features), synthetic only."""

import json
import os

import joblib
import numpy as np
import pytest

from pitchvision import config
from pvext.v2 import deploy, features, final, m1
from tests.ext.v2 import release_fixture as rf
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking

FPS = 25


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    path, manifest = rf.build_release(tmp_path_factory.mktemp("release"))
    return path


def test_valid_release_loads(release):
    rel = deploy.load_release(release, FPS)
    assert rel.bundle["kind"] == "final" and rel.config["threshold"] in m1.THRESHOLDS


@pytest.mark.parametrize("change, fragment", [
    ({"schema": "other"}, "schema"),
    ({"fps": 30}, "fps"),
    ({"persistence": {"smooth_s": 2.0, "min_dead_s": 2.0, "min_alive_s": 1.0}}, "persistence"),
    ({"v1_config_sha256": "0" * 64}, "V1 config hash"),
    ({"dependencies": {"scikit-learn": "0.0.1"}}, "dependency set"),
    ({"model_sha256": "0" * 64}, "sha256"),
    ({"model": "/nonexistent/m1_final.joblib"}, "not found"),
    ({"n_features": 58}, "feature count"),
    ({"feature_names_sha256": "0" * 64}, "feature names"),
])
def test_release_config_problems_are_rejected(release, tmp_path, change, fragment):
    bad = rf.edit_release(release, tmp_path / "bad.json", **change)
    with pytest.raises(deploy.ReleaseError, match=fragment):
        deploy.load_release(bad, FPS)


def test_threshold_mismatch(release, tmp_path):
    with open(release) as handle:
        t = json.load(handle)["threshold"]
    other = 0.5 if t != 0.5 else 0.45
    bad = rf.edit_release(release, tmp_path / "bad.json", threshold=other)
    with pytest.raises(deploy.ReleaseError, match="threshold"):
        deploy.load_release(bad, FPS)


def test_competition_fps_must_match(release):
    with pytest.raises(deploy.ReleaseError, match="fps"):
        deploy.load_release(release, 30)


def test_missing_config_and_missing_keys(release, tmp_path):
    with pytest.raises(deploy.ReleaseError, match="cannot read"):
        deploy.load_release(str(tmp_path / "none.json"), FPS)
    with open(release) as handle:
        data = json.load(handle)
    del data["threshold"]
    (tmp_path / "partial.json").write_text(json.dumps(data))
    with pytest.raises(deploy.ReleaseError, match="missing"):
        deploy.load_release(str(tmp_path / "partial.json"), FPS)


def test_wrong_hash_never_unpickles(release, tmp_path, monkeypatch):
    bad = rf.edit_release(release, tmp_path / "bad.json", model_sha256="0" * 64)
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("unverified pickle loaded"))
    with pytest.raises(deploy.ReleaseError, match="sha256"):
        deploy.load_release(bad, FPS)


def _rebundle(release, tmp_path, mutate):
    """A copy of the released bundle, mutated, saved, with a release config whose hash fits it."""
    with open(release) as handle:
        rel = json.load(handle)
    bundle = joblib.load(rel["model"])
    mutate(bundle)
    path = str(tmp_path / "m.joblib")
    joblib.dump(bundle, path)
    return rf.edit_release(release, tmp_path / "rel.json", model=path, model_sha256=rf.sha(path))


def test_corrupted_model_file(release, tmp_path):
    path = tmp_path / "broken.joblib"
    path.write_bytes(b"not a pickle at all")
    bad = rf.edit_release(release, tmp_path / "bad.json", model=str(path),
                          model_sha256=rf.sha(str(path)))
    with pytest.raises(deploy.ReleaseError, match="cannot be loaded"):
        deploy.load_release(bad, FPS)


@pytest.mark.parametrize("mutation, fragment", [
    ("wrong_type", "StandardScaler"),
    ("fewer_features", "feature count"),
    ("not_final", "final"),
    ("held_out_match", "final"),
    ("missing_metadata", "metadata"),
    ("bundle_threshold", "threshold"),
    ("bundle_persistence", "persistence"),
])
def test_inconsistent_bundles_are_rejected(release, tmp_path, mutation, fragment):
    def mutate(bundle):
        if mutation == "wrong_type":
            bundle["model"] = bundle["model"].steps[1][1]
        elif mutation == "fewer_features":
            rng = np.random.default_rng(0)
            bundle["model"] = m1.fit(rng.normal(size=(200, 58)), np.arange(200) % 2 == 0)
        elif mutation == "not_final":
            bundle["kind"] = "lomo"
        elif mutation == "held_out_match":
            bundle["train"], bundle["test"] = list(m1.MATCHES[1:]), [m1.MATCHES[0]]
        elif mutation == "missing_metadata":
            del bundle["features"]
        elif mutation == "bundle_threshold":
            bundle["threshold"] = 0.7 if bundle["threshold"] != 0.7 else 0.3
        elif mutation == "bundle_persistence":
            bundle["persistence"] = dict(bundle["persistence"], min_dead_s=3.0)
    with pytest.raises(deploy.ReleaseError, match=fragment):
        deploy.load_release(_rebundle(release, tmp_path, mutate), FPS)


def test_feature_order_mismatch_is_caught_after_extraction(release, tmp_path):
    def mutate(bundle):
        bundle["features"] = list(reversed(bundle["features"]))
    path = _rebundle(release, tmp_path, mutate)
    reversed_names = list(reversed(final.feature_schema()))
    path = rf.edit_release(path, tmp_path / "rel2.json",
                           feature_names_sha256=deploy.names_sha256(reversed_names))
    loaded = deploy.load_release(path, FPS)          # self-consistent bundle and config...
    with pytest.raises(deploy.ReleaseError, match="names/order"):
        deploy.label(loaded, tracking(), PITCH, BLOCKS)   # ...but not the extractor's schema


def test_non_finite_features_are_rejected(release, monkeypatch):
    rel = deploy.load_release(release, FPS)
    real = features.compute

    def broken(*args, **kwargs):
        names, X = real(*args, **kwargs)
        X = X.copy()
        X[10, 3] = np.nan
        return names, X
    monkeypatch.setattr(features, "compute", broken)
    with pytest.raises(deploy.ReleaseError, match="non-finite"):
        deploy.label(rel, tracking(), PITCH, BLOCKS)


def test_label_uses_the_frozen_v1_possession_machine(release):
    rel = deploy.load_release(release, FPS)
    tr = tracking(slowdown_at=900)
    blocks = [(1, 100, 1399)]
    labels, info = deploy.label(rel, tr, PITCH, blocks)
    names, X = features.compute(tr, PITCH, blocks, FPS)
    dead = m1.persist(rel.bundle["model"].predict_proba(X)[:, 1], blocks, rel.v1["alive"], FPS,
                      rel.config["threshold"])
    assert np.array_equal(labels, m1.run_with_dead(tr, rel.v1, FPS, PITCH, blocks, dead))
    assert set(labels[:100]) == {"X"} and set(labels[1400:]) == {"X"}
    assert set(labels[100:1400]) <= {"H", "A", "D"}
    assert info["decision_margin_min"] >= 0.0


def test_reloaded_model_gives_the_same_labels_as_a_refit(release, tmp_path):
    rel = deploy.load_release(release, FPS)
    src = os.path.join(os.path.dirname(os.path.dirname(rel.model_path)), "src")
    from pvext.v2 import experiment
    series = [s for m in m1.MATCHES for s in experiment._series(src, m)]
    refit = m1.fit(np.concatenate([s["X"] for s in series]),
                   np.concatenate([s["y"] for s in series]))
    tr = tracking(seed=4, slowdown_at=700)
    labels, _ = deploy.label(rel, tr, PITCH, BLOCKS)
    twin = rel._replace(bundle=dict(rel.bundle, model=refit))
    assert np.array_equal(labels, deploy.label(twin, tr, PITCH, BLOCKS)[0])


def test_tagging_flags_are_refused(release, tmp_path):
    v1_path = tmp_path / "v1_tagging.json"
    v1_path.write_text(json.dumps({"tagging": {"use_setpiece_dead": True}}))
    bad = rf.edit_release(release, tmp_path / "bad.json", v1_config=str(v1_path),
                          v1_config_sha256=config.sha256(config.load(str(v1_path))))
    with pytest.raises(deploy.ReleaseError, match="tagging"):
        deploy.load_release(bad, FPS)


# -- dependency metadata (review follow-up 2) ------------------------------------------------------

def test_dependency_tuple_matches_training():
    assert deploy.DEPENDENCIES == final.DEPENDENCIES


@pytest.mark.parametrize("change", ["missing", "unexpected", "wrong_version", "not_a_dict"])
def test_dependency_metadata_must_be_complete_and_installed(release, tmp_path, change):
    with open(release) as handle:
        deps = dict(json.load(handle)["dependencies"])
    if change == "missing":
        del deps["threadpoolctl"]
    elif change == "unexpected":
        deps["pandas"] = "3.0.6"
    elif change == "wrong_version":
        deps["scipy"] = "0.1"
    else:
        deps = ["numpy"]
    bad = rf.edit_release(release, tmp_path / "bad.json", dependencies=deps)
    with pytest.raises(deploy.ReleaseError, match="dependenc"):
        deploy.load_release(bad, FPS)


@pytest.mark.parametrize("mutation", ["other_version", "missing_entry", "no_versions"])
def test_bundle_recorded_versions_must_match_the_release(release, tmp_path, mutation):
    def mutate(bundle):
        if mutation == "other_version":
            bundle["versions"] = dict(bundle["versions"], numpy="1.0.0")
        elif mutation == "missing_entry":
            bundle["versions"] = {k: v for k, v in bundle["versions"].items() if k != "joblib"}
        else:
            del bundle["versions"]
    with pytest.raises(deploy.ReleaseError, match="dependency versions|metadata"):
        deploy.load_release(_rebundle(release, tmp_path, mutate), FPS)
