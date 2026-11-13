"""Final-model training: cache verification, training-only threshold selection, artifacts."""

import json
import os

import joblib
import numpy as np
import pytest

from pvext.v2 import final, m1
from tests.ext.v2 import release_fixture as rf
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking


def test_feature_schema_is_the_extractors_59_features():
    from pvext.v2 import features
    names, X = features.compute(tracking(), PITCH, BLOCKS, 25)
    assert final.feature_schema() == names and len(names) == 59 == X.shape[1]


@pytest.fixture
def src(tmp_path):
    path = str(tmp_path / "src")
    return path, rf.write_caches(path)


def test_verify_caches_accepts_the_expected_set(src):
    path, manifest = src
    out = final.verify_caches(path, manifest)
    assert len(out["cache_sha256"]) == 42 and out["n_rows"] == 42 * rf.ROWS
    assert out["names"] == final.feature_schema()


def test_cache_hash_mismatch(src):
    path, manifest = src
    target = os.path.join(path, "train", "J03WQQ_doc_101.npz")
    with np.load(target) as d:
        data = dict(d)
    data["y"] = ~data["y"]
    np.savez(target, **data)
    with pytest.raises(ValueError, match="differs from the manifest"):
        final.verify_caches(path, manifest)


def test_missing_cache(src):
    path, manifest = src
    os.remove(os.path.join(path, "train", "J03WMX_mild_100.npz"))
    with pytest.raises(ValueError, match="missing training cache"):
        final.verify_caches(path, manifest)


def test_wrong_feature_order(tmp_path):
    names = final.feature_schema()
    names[0], names[1] = names[1], names[0]
    path = str(tmp_path / "src")
    manifest = rf.write_caches(path, names=names)          # hashes consistent, order wrong
    with pytest.raises(ValueError, match="names/order"):
        final.verify_caches(path, manifest)


def test_wrong_feature_width(tmp_path):
    path = str(tmp_path / "src")
    manifest = rf.write_caches(path, names=final.feature_schema()[:58])
    with pytest.raises(ValueError, match="names/order"):
        final.verify_caches(path, manifest)


@pytest.mark.parametrize("change", ["extra_match", "missing_match"])
def test_unexpected_or_missing_training_matches(src, change):
    path, manifest = src
    with open(manifest) as handle:
        hashes = json.load(handle)["train_cache_sha256"]
    if change == "extra_match":
        hashes["J03XXX_doc_100.npz"] = "0" * 64
    else:
        hashes = {k: v for k, v in hashes.items() if not k.startswith("J03WR9")}
    rf.rewrite_manifest(manifest, train_cache_sha256=hashes)
    with pytest.raises(ValueError, match="cache set differs"):
        final.verify_caches(path, manifest)


@pytest.mark.parametrize("key, value", [("train_seeds", [0, 1]), ("stride", 1),
                                        ("train_presets", ["doc"]), ("thresholds_grid", [0.5])])
def test_training_configuration_must_match_m1(src, key, value):
    path, manifest = src
    with open(manifest) as handle:
        cfg = json.load(handle)["config"]
    cfg[key] = value
    rf.rewrite_manifest(manifest, config=cfg)
    with pytest.raises(ValueError, match=key):
        final.verify_caches(path, manifest)


def test_train_final_uses_the_predeclared_threshold_procedure(src, tmp_path, monkeypatch):
    path, manifest = src
    calls = []
    real = m1.select_threshold

    def spy(train, cfg_alive, fps):
        calls.append((sorted(train), fps, {m: len(v) for m, v in train.items()}))
        return real(train, cfg_alive, fps)
    monkeypatch.setattr(m1, "select_threshold", spy)
    out = str(tmp_path / "final")
    result = final.train_final(path, manifest, out)
    assert len(calls) == 1
    matches, fps, copies = calls[0]
    assert matches == sorted(m1.MATCHES) and fps == 25 / m1.STRIDE
    assert set(copies.values()) == {len(m1.TRAIN_PRESETS) * len(m1.TRAIN_SEEDS)}
    bundle = joblib.load(os.path.join(out, final.MODEL_FILE))
    assert bundle["threshold"] == result["threshold"] in m1.THRESHOLDS
    assert bundle["kind"] == "final" and bundle["train"] == list(m1.MATCHES) and bundle["test"] == []
    assert bundle["features"] == final.feature_schema() and bundle["n_rows"] == 42 * rf.ROWS


def test_train_final_artifacts_and_manifest(src, tmp_path):
    path, manifest = src
    out = str(tmp_path / "final")
    result = final.train_final(path, manifest, out)
    assert sorted(os.listdir(out)) == sorted([final.MODEL_FILE, final.MANIFEST_FILE,
                                              final.RELEASE_CONFIG_FILE])
    model_sha = rf.sha(os.path.join(out, final.MODEL_FILE))
    with open(os.path.join(out, final.MANIFEST_FILE)) as handle:
        written = json.load(handle)
    assert written["model_sha256"] == model_sha == result["model_sha256"]
    assert set(written["threshold_curve"]) == {str(t) for t in m1.THRESHOLDS}
    assert written["training_matches"] == list(m1.MATCHES)
    assert len(written["training_cache_sha256"]) == 42
    assert "training_set_d_f1_sanity_only" in written
    assert "not generalization" in written["sanity_note"]
    with open(os.path.join(out, final.RELEASE_CONFIG_FILE)) as handle:
        release = json.load(handle)
    assert release["model_sha256"] == model_sha and release["threshold"] == written["threshold"]
    assert release["n_features"] == 59 and release["fps"] == 25
    assert "model_sha256" not in joblib.load(os.path.join(out, final.MODEL_FILE))  # hash kept outside


def test_export_and_reload_are_consistent(src, tmp_path):
    path, manifest = src
    out = str(tmp_path / "final")
    final.train_final(path, manifest, out)
    bundle = joblib.load(os.path.join(out, final.MODEL_FILE))
    from pvext.v2 import experiment
    series = [s for m in m1.MATCHES for s in experiment._series(path, m)]
    X = np.concatenate([s["X"] for s in series])
    y = np.concatenate([s["y"] for s in series])
    refit = m1.fit(X, y)
    assert np.array_equal(bundle["model"].predict_proba(X), refit.predict_proba(X))


def test_existing_output_is_never_overwritten(src, tmp_path):
    path, manifest = src
    out = tmp_path / "final"
    out.mkdir()
    (out / "m1_final.joblib").write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        final.train_final(path, manifest, str(out))
    assert (out / "m1_final.joblib").read_bytes() == b"keep me"


def test_output_inside_the_repository_is_refused(src):
    path, manifest = src
    target = os.path.join(final.ROOT, "final_model_should_not_exist")
    with pytest.raises(ValueError, match="outside the repository"):
        final.train_final(path, manifest, target)
    assert not os.path.exists(target)


def test_cli_refuses_the_frozen_directories(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_final_cli", os.path.join(final.ROOT, "scripts", "v2_train_final.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (tmp_path / "v2" / "m1").mkdir(parents=True)
    (tmp_path / "v2" / "m1_robust").mkdir(parents=True)
    os.symlink(tmp_path / "v2" / "m1", tmp_path / "link_to_m1")
    for target in (tmp_path / "v2" / "m1", tmp_path / "v2" / "m1_robust",
                   tmp_path / "v2" / "m1" / "models" / "final",          # subdirectories too
                   tmp_path / "v2" / "m1_robust" / "x",
                   tmp_path / "v2" / "m1" / "." / "nested" / ".." / "other",
                   tmp_path / "link_to_m1" / "final"):                   # and through a symlink
        with pytest.raises(SystemExit, match="frozen"):
            cli.main(["--root", str(tmp_path), "--out", str(target)])
    assert not (tmp_path / "v2" / "m1" / "models").exists()
    assert cli.inside(str(tmp_path / "v2" / "m1_final"), str(tmp_path / "v2" / "m1")) is False
