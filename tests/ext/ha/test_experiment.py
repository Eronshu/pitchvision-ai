"""HA1 experiment protocol: stage isolation, training-only lambda selection, one-shot lockbox and
the pre-declared GO criteria. Synthetic data only."""

import json
import os

import joblib
import numpy as np
import pytest

from pvext.ha import experiment as hx
from pvext.v2 import m1


def test_stages_are_disjoint_and_as_declared():
    assert set(hx.DEV) == set(m1.SECOND_DIVISION) and set(hx.LOCKBOX) == set(m1.BUNDESLIGA)
    assert not set(hx.DEV) & set(hx.LOCKBOX)
    assert hx.LOCKBOX_M1 == "league_J03WMX-J03WN1.joblib"


@pytest.mark.parametrize("match", m1.BUNDESLIGA)
def test_dev_code_paths_refuse_lockbox_matches(match, tmp_path):
    with pytest.raises(ValueError):
        hx.build_training(match, str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError):
        hx.load_series(str(tmp_path), match)
    with pytest.raises(ValueError):
        hx.train("lomo", [match], ["J03WPY"], str(tmp_path))


def test_evaluate_refuses_a_match_from_the_other_stage(tmp_path):
    with pytest.raises(ValueError):
        hx.evaluate_match("J03WMX", "dev", "x", "y", str(tmp_path), str(tmp_path), "z")
    with pytest.raises(ValueError):
        hx.evaluate_match("J03WPY", "lockbox", "x", "y", str(tmp_path), str(tmp_path), "z")


def _series(seed, n=600, switch_every=60):
    rng = np.random.default_rng(seed)
    team = (np.arange(n) // switch_every) % 2 == 0
    target = np.where(team, "H", "A").astype("<U1")
    target[280:300] = "D"
    X = rng.normal(0, 1, (n, 4)).astype(np.float32)
    X[:, 0] += np.where(team, 1.5, -1.5)
    return {"X": X, "target": target, "blocks": [(1, 0, n // 2 - 1), (2, n // 2, n - 1)],
            "names": ["a", "b", "c", "d"]}


def test_lambda_selection_uses_only_the_given_training_matches(monkeypatch):
    seen = []
    real = hx.fit

    def spy(X, y):
        seen.append(len(X))
        return real(X, y)
    monkeypatch.setattr(hx, "fit", spy)
    train = {m: [_series(i), _series(i + 10)] for i, m in enumerate(["A", "B", "C"])}
    lam, curve = hx.select_lambda(train)
    assert lam in hx.LAMBDAS and set(curve) == set(hx.LAMBDAS)
    alive_rows = int(np.isin(_series(0)["target"], ["H", "A"]).sum())
    assert seen == [4 * alive_rows] * 3          # each inner fit: 2 matches x 2 series, alive rows


def test_lambda_ties_go_to_the_larger_value():
    assert hx.pick_lambda({lam: 0.5 for lam in hx.LAMBDAS}) == max(hx.LAMBDAS)
    assert hx.pick_lambda({0.5: 0.6, 1.0: 0.6, 2.0: 0.4}) == 1.0


def test_strided_spell_scoring_never_spans_blocks():
    target = np.array(list("HHHH" + "HHHH"))
    pred = target.copy()
    blocks = [(1, 0, 3), (2, 4, 7)]
    assert hx.evaluate_spell(target, pred, blocks) == pytest.approx(1.0)
    one_spell = np.array(list("HHHHHHHH"))
    # scored per block, a prediction is two spells as well: still a perfect match
    assert hx.evaluate_spell(target, one_spell, blocks) == pytest.approx(1.0)


def test_train_saves_membership_and_refuses_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(hx, "load_series", lambda work, m: [_series(hash(m) % 7)])
    path = hx.train("lomo", ["J03WOH", "J03WOY", "J03WQQ"], ["J03WPY"], str(tmp_path))
    bundle = joblib.load(path)
    assert bundle["train"] == ["J03WOH", "J03WOY", "J03WQQ"] and bundle["test"] == ["J03WPY"]
    assert bundle["lam"] in hx.LAMBDAS and bundle["features"] == ["a", "b", "c", "d"]
    with pytest.raises(ValueError):
        hx.train("lomo", ["J03WOH", "J03WPY"], ["J03WPY"], str(tmp_path))


def test_label_keeps_the_dead_mask_and_checks_feature_names(monkeypatch):
    from tests.ext.v2.test_features import BLOCKS, PITCH, tracking
    from pvext.ha import features
    tr = tracking(seed=4)
    names, X = features.compute(tr, PITCH, BLOCKS, 25)
    model = hx.fit(X, (np.arange(len(X)) // 300) % 2 == 0)
    dead = np.zeros(len(X), bool)
    dead[500:600] = True
    out = hx.label({"model": model, "lam": 2.0, "features": names}, tr, PITCH, BLOCKS, dead)
    assert (out[500:600] == "D").all() and set(out[BLOCKS[0][1]:500]) <= {"H", "A"}
    with pytest.raises(ValueError, match="names"):
        hx.label({"model": model, "lam": 2.0, "features": list(reversed(names))}, tr, PITCH,
                 BLOCKS, dead)


def _rows(dev_gain, lock_gain, ratio=1.0, other=0.01):
    rows = []
    for stage, matches, gains in (("dev", hx.DEV, dev_gain), ("lockbox", hx.LOCKBOX, lock_gain)):
        ref = hx.REF[stage]
        for m in matches:
            for preset in m1.EVAL_PRESETS:
                g = gains[m] if preset == "doc" else other
                for seed in m1.EVAL_SEEDS:
                    for method, val in ((ref, 0.4), ("m1_ha1", 0.4 + g)):
                        rows.append({"match": m, "stage": stage, "preset": preset, "seed": seed,
                                     "method": method, "spell_f1_mean": val,
                                     "n_pred_spells": 100 * ratio, "n_true_spells": 100})
    return [r for r in rows if r["stage"] == "dev"], [r for r in rows if r["stage"] == "lockbox"]


def test_go_criteria():
    up = {m: 0.06 for m in hx.DEV + hx.LOCKBOX}
    dev, lock = _rows(up, up)
    assert hx.go_criteria(dev, lock)["verdict"] == "GO"
    dev, lock = _rows(up, dict(up, J03WN1=-0.01))
    assert hx.go_criteria(dev, lock)["verdict"] == "NO-GO_KEEP_M1"
    small = {m: 0.01 for m in hx.DEV + hx.LOCKBOX}
    dev, lock = _rows(small, small)
    assert hx.go_criteria(dev, lock)["verdict"] == "WITHIN_NOISE_KEEP_M1"
    dev, lock = _rows(up, up, ratio=1.8)
    assert hx.go_criteria(dev, lock)["criteria"]["4_spell_count_ratio_0.7_1.5"]["pass"] is False
    dev, lock = _rows(up, up, other=-0.02)
    assert hx.go_criteria(dev, lock)["verdict"] != "GO"


@pytest.mark.parametrize("damage", ["drop_dev", "drop_lock", "duplicate", "nan", "extra_match"])
def test_go_criteria_requires_the_complete_grid(damage):
    up = {m: 0.06 for m in hx.DEV + hx.LOCKBOX}
    dev, lock = _rows(up, up)
    if damage == "drop_dev":
        dev = dev[1:]
    elif damage == "drop_lock":
        lock = lock[:-1]
    elif damage == "duplicate":
        dev = dev + [dict(dev[0])]
    elif damage == "nan":
        lock[0]["spell_f1_mean"] = float("nan")
    else:
        lock = lock + [dict(lock[0], match="J03WPY")]
    with pytest.raises(ValueError):
        hx.go_criteria(dev, lock)


def _cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ha_cli", os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "ha_experiment.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_lockbox_runs_once_and_only_after_dev(tmp_path):
    cli = _cli()
    (tmp_path / "v2" / "m1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="dev-run"):
        cli.main(["--stage", "lockbox", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="dev stage first"):
        cli.main(["--stage", "lockbox", "--dev-run", "ha1_001", "--root", str(tmp_path)])
    run = tmp_path / "v3" / "ha" / "runs" / "ha1_001"
    (run / "lockbox").mkdir(parents=True)
    (run / "dev_per_run.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="once only"):
        cli.main(["--stage", "lockbox", "--dev-run", "ha1_001", "--root", str(tmp_path)])


def test_dev_runs_never_overwrite_an_earlier_run(tmp_path):
    cli = _cli()
    (tmp_path / "v2" / "m1").mkdir(parents=True)
    run = tmp_path / "v3" / "ha" / "runs" / "ha1_001"
    run.mkdir(parents=True)
    (run / "keep.txt").write_text("earlier")
    with pytest.raises(SystemExit, match="never overwritten"):
        cli.main(["--stage", "dev", "--run-id", "ha1_001", "--root", str(tmp_path)])
    assert (run / "keep.txt").read_text() == "earlier"
    with pytest.raises(SystemExit, match="plain"):
        cli.main(["--stage", "dev", "--run-id", "../escape", "--root", str(tmp_path)])


def test_variants_change_only_the_lambda_grid(monkeypatch, tmp_path):
    assert hx.VARIANTS["ha1"]["lambdas"] == hx.LAMBDAS == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert set(hx.VARIANTS) == {"ha1", "ha1b", "ha2", "ha3"}
    assert {k: v["model"] for k, v in hx.VARIANTS.items()} == {"ha1": "linear", "ha1b": "linear",
                                                               "ha2": "gb_sym", "ha3": "gb_sym"}
    assert {k: hx.feature_set(k) for k in hx.VARIANTS} == {"ha1": "ha1", "ha1b": "ha1",
                                                           "ha2": "ha1", "ha3": "ha3"}
    assert hx.VARIANTS["ha3"]["lambdas"] == hx.VARIANTS["ha2"]["lambdas"]
    assert hx.VARIANTS["ha2"]["lambdas"] == hx.VARIANTS["ha1b"]["lambdas"]
    assert min(hx.VARIANTS["ha1b"]["lambdas"]) < min(hx.LAMBDAS)
    monkeypatch.setattr(hx, "load_series", lambda work, m: [_series(hash(m) % 7)])
    path = hx.train("lomo", ["J03WOH", "J03WOY"], ["J03WPY"], str(tmp_path), variant="ha1b")
    bundle = joblib.load(path)
    assert bundle["variant"] == "ha1b" and set(bundle["curve"]) == set(hx.VARIANTS["ha1b"]["lambdas"])
    assert bundle["lam"] in hx.VARIANTS["ha1b"]["lambdas"]


def test_gb_sym_is_exactly_team_symmetric_and_learns():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (4000, 5)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] ** 3 > 0
    model = hx.fit_model(X, y, "gb_sym")
    p = model.predict_proba(X)
    assert np.allclose(p.sum(1), 1) and np.allclose(model.predict_proba(-X)[:, 1], 1 - p[:, 1])
    assert np.mean((p[:, 1] >= 0.5) == y) > 0.9
    assert np.allclose(model.predict_proba(np.zeros((3, 5), np.float32))[:, 1], 0.5)


def test_gb_sym_caps_rows_and_mirrors(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier
    seen = []
    real = HistGradientBoostingClassifier.fit

    def spy(self, X, y):
        seen.append((len(X), int(y.sum())))
        return real(self, X, y)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", spy)
    monkeypatch.setattr(hx, "GB_ROWS", 500)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 3)).astype(np.float32)
    hx.fit_model(X, X[:, 0] > 0, "gb_sym")
    assert seen == [(1000, 500)]                 # 500 sampled + 500 mirrored, labels balanced


def test_ha2_trains_gb_inside_lambda_selection_too(monkeypatch, tmp_path):
    kinds = []
    real = hx.fit_model

    def spy(X, y, kind="linear"):
        kinds.append(kind)
        return real(X, y, kind)
    monkeypatch.setattr(hx, "fit_model", spy)
    monkeypatch.setattr(hx, "load_series", lambda work, m: [_series(hash(m) % 7)])
    path = hx.train("lomo", ["J03WOH", "J03WOY", "J03WQQ"], ["J03WPY"], str(tmp_path), variant="ha2")
    bundle = joblib.load(path)
    assert kinds == ["gb_sym"] * 4               # 3 inner folds + the final fit
    assert bundle["variant"] == "ha2" and bundle["model_kind"] == "gb_sym"
    p = bundle["model"].predict_proba(_series(1)["X"])[:, 1]
    assert np.allclose(bundle["model"].predict_proba(-_series(1)["X"])[:, 1], 1 - p)


def test_label_uses_the_bundle_feature_set():
    from tests.ext.v2.test_features import BLOCKS, PITCH, tracking
    from pvext.ha import features
    tr = tracking(seed=4)
    names, X = features.compute(tr, PITCH, BLOCKS, 25, feature_set="ha3")
    model = hx.fit(X, (np.arange(len(X)) // 300) % 2 == 0)
    dead = np.zeros(len(X), bool)
    bundle = {"model": model, "lam": 2.0, "features": names, "feature_set": "ha3"}
    assert set(hx.label(bundle, tr, PITCH, BLOCKS, dead)[BLOCKS[0][1]:BLOCKS[0][2] + 1]) <= {"H", "A"}
    with pytest.raises(ValueError, match="names"):
        hx.label(dict(bundle, feature_set="ha1"), tr, PITCH, BLOCKS, dead)
