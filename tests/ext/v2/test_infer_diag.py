"""The BAR-ATH diagnostic script, exercised on synthetic tracking only."""

import importlib.util
import os

import numpy as np
import pytest

from pitchvision import config
from pvext.v2 import features, m1
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
spec = importlib.util.spec_from_file_location("infer_diag",
                                              os.path.join(ROOT, "scripts", "v2_m1_infer_diag.py"))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
CFG = config.load(m1.V1_CONFIG)


def _bundle(seed):
    tr = tracking(seed=seed, slowdown_at=900)
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    y = np.arange(len(X)) >= 900
    return {"model": m1.fit(X, y), "threshold": 0.5}


def test_diagnose_aggregates_only_and_checks():
    tr = tracking(seed=3, slowdown_at=1000)
    out = diag.diagnose(tr, BLOCKS, PITCH, 25, {"m1_a": _bundle(1), "m1_b": _bundle(2)}, CFG)
    assert out["checks"]["n_frames"] == 1500 and out["checks"]["features_finite"]
    assert set(out["methods"]) == {"v1", "m1_a", "m1_b"}
    for d in out["methods"].values():
        assert sum(d["share"].values()) == pytest.approx(1.0)
        assert 0.0 <= d["agreement_with_v1"] <= 1.0
    assert out["methods"]["v1"]["agreement_with_v1"] == 1.0
    assert out["m1_spread"]["d_share_min"] <= out["m1_spread"]["d_share_max"]
    text = repr(out)
    assert "array" not in text                       # no per-frame arrays leave the function


def test_diagnose_rejects_a_model_with_the_wrong_feature_count():
    bundle = {"model": m1.fit(np.random.default_rng(0).normal(size=(100, 3)),
                              np.arange(100) % 2 == 0), "threshold": 0.5}
    with pytest.raises(ValueError):
        diag.diagnose(tracking(), BLOCKS, PITCH, 25, {"m1_x": bundle}, CFG)


def test_main_refuses_an_output_inside_the_repository(tmp_path):
    with pytest.raises(SystemExit):
        diag.main(["--out", os.path.join(ROOT, "diag_out"), "--models", str(tmp_path)])
    assert not os.path.exists(os.path.join(ROOT, "diag_out"))


def test_dead_runs_and_jaccard():
    pred = np.array(list("HHDDDDAAXDD"))
    assert diag._dead_runs_s(pred, [(1, 0, 10)], 1) == [4.0, 2.0]
    assert diag._jaccard(np.array([True, False]), np.array([False, False])) == 0.0
    assert diag._jaccard(np.zeros(3, bool), np.zeros(3, bool)) == 1.0


# -- model-set validation (review finding 2) -------------------------------------------------------

def _model_dir(tmp_path, manifest=True, tamper=None):
    import hashlib
    import json
    import joblib
    models = tmp_path / "models"
    models.mkdir()
    entries = {}
    for name, (train, test) in diag.expected_models().items():
        bundle = {"model": None, "threshold": 0.4, "train": list(train), "test": list(test)}
        if tamper == "membership" and test == ["J03WQQ"]:
            bundle["train"] = [m for m in train if m != "J03WMX"]          # wrong training set
        if tamper == "leak" and test == ["J03WQQ"]:
            bundle["train"] = list(train) + ["J03WQQ"]
        joblib.dump(bundle, models / name)
        entries[name] = {"sha256": hashlib.sha256((models / name).read_bytes()).hexdigest(),
                         "threshold": 0.4}
    if tamper == "hash":
        entries["lomo_J03WN1.joblib"]["sha256"] = "0" * 64
    if tamper == "threshold":
        entries["lomo_J03WN1.joblib"]["threshold"] = 0.5
    if tamper == "not_in_manifest":
        del entries["lomo_J03WN1.joblib"]
    if manifest:
        (tmp_path / "MANIFEST.json").write_text(json.dumps({"models": entries}))
    return str(models), str(tmp_path / "MANIFEST.json")


def test_load_models_accepts_the_exact_verified_set(tmp_path):
    models, manifest = _model_dir(tmp_path)
    bundles, prov = diag.load_models(models, manifest)
    assert sorted(bundles) == sorted("m1_lomo_" + m for m in m1.MATCHES)
    assert prov["manifest_verified"] is True and len(prov["models_sha256"]) == 7


@pytest.mark.parametrize("tamper", ["hash", "threshold", "not_in_manifest", "membership", "leak"])
def test_load_models_rejects_inconsistent_models(tmp_path, tamper):
    models, manifest = _model_dir(tmp_path, tamper=tamper)
    with pytest.raises(ValueError):
        diag.load_models(models, manifest)


def test_load_models_rejects_missing_or_unexpected_files(tmp_path):
    import os as _os
    import shutil
    models, manifest = _model_dir(tmp_path)
    shutil.copy(_os.path.join(models, "lomo_J03WMX.joblib"), _os.path.join(models, "lomo_EXTRA.joblib"))
    with pytest.raises(ValueError, match="unexpected"):
        diag.load_models(models, manifest)
    _os.remove(_os.path.join(models, "lomo_EXTRA.joblib"))
    _os.remove(_os.path.join(models, "lomo_J03WR9.joblib"))
    with pytest.raises(ValueError, match="missing"):
        diag.load_models(models, manifest)


def test_load_models_requires_a_manifest_unless_explicitly_allowed(tmp_path):
    models, manifest = _model_dir(tmp_path, manifest=False)
    with pytest.raises(ValueError, match="manifest"):
        diag.load_models(models, manifest)
    _, prov = diag.load_models(models, manifest, allow_unverified=True)
    assert prov["manifest_verified"] is False


def test_main_exits_on_a_bad_model_set_before_touching_data(tmp_path, monkeypatch):
    models, manifest = _model_dir(tmp_path, tamper="hash")
    import games
    monkeypatch.setattr(games, "require", lambda *a, **k: pytest.fail("data must not be read"))
    with pytest.raises(SystemExit, match="model check failed"):
        diag.main(["--out", str(tmp_path / "out"), "--models", models])
    assert not (tmp_path / "out").exists()
