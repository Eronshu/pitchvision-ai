import copy
import inspect
import json

import joblib
import numpy as np
import pytest

from pvext import degrade
from pvext.v2 import experiment, features, m1, robust
from tests.ext.test_degrade import synthetic_frames
from tests.ext.v2.test_features import BLOCKS, PITCH, tracking


def feature_names():
    names, _ = features.compute(tracking(), PITCH, BLOCKS, 25)
    return names


# -- ablation columns -------------------------------------------------------------------------------

def test_every_ablated_feature_exists_and_variants_drop_exactly_their_group():
    names = feature_names()
    assert len(robust.columns(names, "full")) == len(names)
    for variant, removed in robust.ABLATIONS.items():
        kept = [names[i] for i in robust.columns(names, variant)]
        assert set(names) - set(kept) == set(removed)
        assert kept == [n for n in names if n not in removed]          # order preserved
    assert set(robust.ABLATIONS["no_both"]) == (set(robust.ABLATIONS["no_ball_presence"])
                                                | set(robust.ABLATIONS["no_slot_speed"]))


def test_unknown_ablation_feature_raises():
    with pytest.raises(ValueError):
        robust.columns(["a", "b"], "no_slot_speed")


# -- refit, not zeroing ----------------------------------------------------------------------------

def _caches(tmp_path, matches, names, seed=0):
    rng = np.random.default_rng(seed)
    (tmp_path / "train").mkdir(exist_ok=True)
    for i, match in enumerate(matches):
        for preset in m1.TRAIN_PRESETS:
            for s in m1.TRAIN_SEEDS:
                y = (np.arange(600) // 60) % 2 == 1
                X = rng.normal(0, 1, (600, len(names))).astype(np.float32)
                X[:, 0] += 2.0 * y                                   # signal in the first column
                np.savez(tmp_path / "train" / "{}_{}_{}.npz".format(match, preset, s),
                         X=X, y=y, frames=np.arange(600) * 5, blocks=json.dumps([(1, 0, 599)]),
                         names=json.dumps(names))


def test_train_variant_refits_on_the_kept_columns_only(tmp_path):
    names = list(robust.ABLATIONS["no_both"]) + ["other_a", "other_b"]
    names = ["other_a"] + [n for n in names if n != "other_a"]
    _caches(tmp_path, ["A", "B", "C"], names)                         # no cache for HELD
    path = robust.train_variant("no_both", "lomo", ["A", "B", "C"], ["HELD"], str(tmp_path),
                                str(tmp_path / "out"))
    bundle = joblib.load(path)
    assert bundle["features"] == ["other_a", "other_b"]
    assert bundle["model"].n_features_in_ == 2
    assert bundle["train"] == ["A", "B", "C"] and bundle["test"] == ["HELD"]
    assert bundle["threshold"] in m1.THRESHOLDS
    experiment.check_held_out(bundle, "HELD")


def test_predict_dead_selects_columns_by_name():
    names = ["x", "drop_me", "y"]
    rng = np.random.default_rng(0)
    yv = (np.arange(2000) // 200) % 2 == 1
    Xfull = rng.normal(0, 1, (2000, 3))
    Xfull[:, 0] += 3 * yv
    model = m1.fit(Xfull[:, [0, 2]], yv)
    bundle = {"model": model, "threshold": 0.5, "features": ["x", "y"]}
    cfg = robust.config.load(m1.V1_CONFIG)["alive"]
    got = robust.predict_dead(bundle, names, Xfull, [(1, 0, 1999)], cfg)
    want = m1.persist(model.predict_proba(Xfull[:, [0, 2]])[:, 1], [(1, 0, 1999)], cfg, 25, 0.5)
    assert np.array_equal(got, want)
    with pytest.raises(ValueError):                                   # frozen bundle, wrong width
        robust.predict_dead({"model": model, "threshold": 0.5}, names, Xfull, [(1, 0, 1999)], cfg)


# -- alternative degradations -----------------------------------------------------------------------

def test_alt_presets_are_label_agnostic_and_leave_frozen_presets_untouched():
    frozen = copy.deepcopy(degrade.PRESETS)
    for name in robust.ALT_PRESETS:
        robust.alt_preset(name)
    assert degrade.PRESETS == frozen
    assert list(inspect.signature(robust.apply_alt).parameters) == ["frames", "name", "seed"]
    assert set(robust.ALT_PRESETS).isdisjoint(degrade.PRESETS)


def test_alt_overrides_take_effect():
    assert robust.alt_preset("alt_uniform_dropout")["hard_weight"] == 1.0
    assert robust.alt_preset("alt_long_gaps")["ball_gap_s"] == 6.0
    assert robust.alt_preset("alt_noisy_ball")["false_ball_rate"] == 0.02
    assert robust.alt_preset("alt_sparse_players")["tracklet_mean_s"] == 8.0
    doc = degrade.PRESETS["doc"]
    assert robust.alt_preset("alt_speed_from_positions")["ball_present"] == doc["ball_present"]


def test_apply_alt_is_deterministic_and_differs_from_doc():
    frames = synthetic_frames(T=3000, gap=(1000, 1100))
    a = robust.apply_alt(frames, "alt_uniform_dropout", 0)
    b = robust.apply_alt(frames, "alt_uniform_dropout", 0)
    doc = degrade.apply(frames, degrade.PRESETS["doc"], 0)
    assert np.array_equal(a.ball_xy, b.ball_xy, equal_nan=True)
    assert not np.array_equal(a.ball_xy, doc.ball_xy, equal_nan=True)


def test_speed_field_replaced_from_positions():
    frames = synthetic_frames(T=3000, gap=(1000, 1100))
    out = robust.apply_alt(frames, "alt_speed_from_positions", 0)
    doc = degrade.apply(frames, degrade.PRESETS["doc"], 0)
    assert np.array_equal(out.players_xy, doc.players_xy, equal_nan=True)
    assert np.array_equal(out.ball_xy, doc.ball_xy, equal_nan=True)
    expected = robust.speed_from_positions(doc.players_xy).astype(np.float32)
    assert np.array_equal(out.players_speed_mps, expected, equal_nan=True)
    assert np.isnan(out.players_speed_mps[np.isnan(out.players_xy[..., 0])]).all()


def test_speed_from_positions_values():
    T = 50
    xy = np.zeros((T, 2, 2))
    xy[:, 0, 0] = np.arange(T) * 2.0 / 25          # 2 m/s along x
    xy[20, 1] = np.nan
    s = robust.speed_from_positions(xy)
    assert np.isnan(s[:2]).all() and np.isnan(s[-2:]).all()
    assert np.allclose(s[2:-2, 0], 2.0)
    assert np.isnan(s[[18, 22], 1]).all() and np.allclose(s[[17, 19, 21, 23], 1], 0.0)


# -- pre-declared reading ---------------------------------------------------------------------------

def test_robustness_reading():
    rows = []
    for i, match in enumerate(m1.MATCHES):
        for method, f1, prec in (("v1", 0.5, 0.9), ("cand", 0.6 if i else 0.4, 0.88)):
            rows.append({"match": match, "preset": "alt_x", "method": method, "d_f1": f1,
                         "d_precision": prec})
    r = robust.robustness_reading(rows, "alt_x", "cand")
    assert r["d_f1_higher"] == 6 and r["robust"] is True
    rows = [dict(r, d_precision=0.8) if r["method"] == "cand" else r for r in rows]
    assert robust.robustness_reading(rows, "alt_x", "cand")["robust"] is False


def test_speed_is_nan_for_a_slot_missing_at_t_even_with_both_neighbours_present():
    """Regression: the central difference over t-2..t+2 once produced a speed for a slot that was
    missing at t itself (found while testing alt_speed_from_positions)."""
    xy = np.zeros((20, 1, 2))
    xy[:, 0, 0] = np.arange(20) * 0.1
    xy[10, 0] = np.nan                       # missing only at t = 10; t = 8 and t = 12 present
    s = robust.speed_from_positions(xy)
    assert np.isnan(s[10, 0])
    assert np.isnan(s[[8, 12], 0]).all()     # their windows reach the missing frame
    assert np.allclose(s[[7, 9, 11, 13], 0], 2.5)


# -- robustness grid and frozen reproducibility (review findings 1 and 3) ---------------------------

REPORT_KEYS = ("boundary_dead_start_tol25_f1", "boundary_dead_start_tol50_f1",
               "boundary_dead_end_tol25_f1", "boundary_dead_end_tol50_f1")


def _grid():
    rows = []
    for match in m1.MATCHES:
        for preset, seed in robust.conditions():
            for method in robust.METHODS:
                row = {"match": match, "preset": preset, "seed": seed, "method": method,
                       "distribution": "frozen" if preset in m1.EVAL_PRESETS else "alt",
                       "input_sha256": "in-{}-{}-{}".format(match, preset, seed),
                       "prediction_sha256": "pred-{}-{}-{}-{}".format(match, preset, seed, method)}
                for k in robust.GRID_METRICS + REPORT_KEYS:
                    row[k] = 0.5
                rows.append(row)
    return rows


def test_robust_grid_accepts_the_full_grid():
    assert robust.validate_grid(_grid()) == 7 * (9 + 15) * len(robust.METHODS)


@pytest.mark.parametrize("damage", ["drop", "duplicate", "nan", "none", "extra_method",
                                    "wrong_distribution", "unknown_preset", "extra_seed"])
def test_robust_grid_rejects_damage(damage):
    rows = _grid()
    if damage == "drop":
        rows.pop(100)
    elif damage == "duplicate":
        rows.append(dict(rows[7]))
    elif damage == "nan":
        rows[9]["d_f1"] = float("nan")
    elif damage == "none":
        rows[9]["d_share_pred"] = None
    elif damage == "extra_method":
        rows.append(dict(rows[0], method="m1_other_lomo"))
    elif damage == "wrong_distribution":
        rows[0]["distribution"] = "alt"
    elif damage == "unknown_preset":
        rows.append(dict(rows[0], preset="alt_unknown", distribution="alt"))
    elif damage == "extra_seed":
        rows.append(dict(rows[0], seed=7))
    with pytest.raises(ValueError):
        robust.validate_grid(rows)


def _script():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "v2_m1_robustness.py")
    spec = importlib.util.spec_from_file_location("robustness_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_csv(tmp_path, rows):
    import csv
    rename = {"v1": "v1", "m1_full_lomo": "m1_lomo", "m1_full_league": "m1_league"}
    frozen = [dict({k: r[k] for k in ("match", "preset", "seed", "input_sha256", "prediction_sha256",
                                      "d_f1", "boundary_dead_start_tol25_f1", "d_precision",
                                      "spell_f1_mean")}, method=rename[r["method"]])
              for r in rows if r["distribution"] == "frozen" and r["method"] in rename]
    path = tmp_path / "frozen.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frozen[0]))
        writer.writeheader()
        writer.writerows(frozen)
    return str(path)


def test_report_passes_when_the_frozen_run_is_reproduced(tmp_path):
    rows = _grid()
    out = _script().report(rows, _frozen_csv(tmp_path, rows))
    assert out["reproducibility"] == {"frozen_runs_checked": 189, "mismatched": 0}


@pytest.mark.parametrize("field", ["prediction_sha256", "input_sha256"])
def test_report_fails_on_any_frozen_hash_mismatch(tmp_path, field):
    rows = _grid()
    frozen = _frozen_csv(tmp_path, rows)
    victim = next(r for r in rows if r["method"] == "m1_full_lomo" and r["preset"] == "doc")
    victim[field] = "changed"
    with pytest.raises(ValueError, match="not reproduced"):
        _script().report(rows, frozen)


def test_report_validates_the_grid_first(tmp_path):
    rows = _grid()
    frozen = _frozen_csv(tmp_path, rows)
    rows.pop(0)
    with pytest.raises(ValueError, match="grid mismatch"):
        _script().report(rows, frozen)
