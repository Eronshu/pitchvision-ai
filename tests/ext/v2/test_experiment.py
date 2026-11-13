import numpy as np
import pytest

from pvext.v2 import experiment


def labels_of(text):
    return np.array(list(text), dtype="<U1")


def test_stride_rows_and_blocks():
    target = labels_of("X" * 10 + "H" * 20 + "D" * 10 + "X" * 7 + "A" * 30)
    X = np.arange(len(target) * 2, dtype=np.float32).reshape(len(target), 2)
    blocks = [(1, 10, 39), (2, 47, 76)]
    rows = experiment.stride_rows(X, target, blocks, stride=5)
    assert rows["frames"].tolist() == [10, 15, 20, 25, 30, 35, 47, 52, 57, 62, 67, 72]
    assert rows["blocks"] == [(1, 0, 5), (2, 6, 11)]
    assert rows["y"].tolist() == [False] * 4 + [True] * 2 + [False] * 6
    assert np.array_equal(rows["X"], X[rows["frames"]])


@pytest.mark.parametrize("pred, category", [
    ("H" * 100 + "D" * 50 + "A" * 50, "matched"),
    ("H" * 140 + "D" * 10 + "A" * 50, "late"),         # starts 40 frames (1.6 s) after the truth
    ("H" * 60 + "D" * 90 + "A" * 50, "early"),         # starts 40 frames before
    ("H" * 150 + "A" * 50, "missed"),                  # no D inside the true stoppage at all
])
def test_start_categories(pred, category):
    truth = labels_of("H" * 100 + "D" * 50 + "A" * 50)
    out = experiment.start_errors(truth, labels_of(pred), [(1, 0, 199)],
                                  ball_out=np.zeros(200, bool), ball_present=np.ones(200, bool))
    assert len(out) == 1 and out[0]["category"] == category
    assert out[0]["duration_s"] == pytest.approx(2.0)
    if category == "late":
        assert out[0]["delay_s"] == pytest.approx(1.6)


def test_start_context_out_of_play():
    truth = labels_of("H" * 100 + "D" * 50 + "A" * 50)
    ball_out = np.zeros(200, bool)
    ball_out[105:110] = True
    out = experiment.start_errors(truth, truth, [(1, 0, 199)], ball_out=ball_out,
                                  ball_present=np.ones(200, bool))
    assert out[0]["out_of_play"] is True and out[0]["ball_present"] is True


def test_summarize_starts():
    entries = [{"category": c, "out_of_play": o, "duration_s": d, "delay_s": l, "ball_present": True}
               for c, o, d, l in [("matched", True, 10, None), ("missed", False, 3, None),
                                  ("late", False, 20, 2.5), ("late", True, 30, 1.5)]]
    s = experiment.summarize_starts(entries)
    assert s["n"] == 4 and s["share"]["late"] == 0.5 and s["late_delay_s_median"] == 2.0
    assert s["by_context"]["in_play"]["n"] == 2 and s["by_context"]["in_play"]["missed"] == 1
    assert s["short_stoppages"]["n"] == 1 and s["short_stoppages"]["missed"] == 1


def _rows(gains):
    """V1 at 0.5 everywhere; the candidate adds gains[metric] per match (doc), mild/severe same."""
    from pvext.v2 import m1
    rows = []
    for match in m1.MATCHES:
        for preset in ("mild", "doc", "severe"):
            for seed in m1.EVAL_SEEDS:
                for method in ("v1", "m1_lomo", "m1_league"):
                    row = {"match": match, "preset": preset, "seed": seed, "method": method}
                    for key in ("d_f1", "boundary_dead_start_tol25_f1", "d_precision",
                                "spell_f1_mean"):
                        row[key] = 0.5 + (0.0 if method == "v1"
                                          else gains.get(key, {}).get(match, 0.0))
                    rows.append(row)
    return rows


def test_go_criteria_go_and_no_go():
    from pvext.v2 import m1
    all_up = {k: {m: 0.1 for m in m1.MATCHES} for k in
              ("d_f1", "boundary_dead_start_tol25_f1")}
    assert experiment.go_criteria(_rows(all_up))["verdict"] == "GO"
    one_bundesliga_down = {"d_f1": dict({m: 0.1 for m in m1.MATCHES}, J03WMX=-0.1),
                           "boundary_dead_start_tol25_f1": {m: 0.1 for m in m1.MATCHES}}
    result = experiment.go_criteria(_rows(one_bundesliga_down))
    assert result["criteria"]["1_d_f1_higher_6of7_incl_both_bundesliga"]["pass"] is False
    assert result["verdict"] == "NO-GO"
    precision_drop = dict(all_up, d_precision={m: -0.2 for m in m1.MATCHES})
    assert experiment.go_criteria(_rows(precision_drop))["verdict"] == "NO-GO"
    starts_flat = {"d_f1": {m: 0.1 for m in m1.MATCHES}}
    assert experiment.go_criteria(_rows(starts_flat))["verdict"] == "PARTIAL"


# -- evaluation-grid validation and held-out membership (review findings 1 and 4) -------------------

def test_validate_grid_accepts_the_full_grid():
    assert experiment.validate_grid(_rows({})) == 7 * 3 * 3 * 3


@pytest.mark.parametrize("damage", ["drop_seed", "duplicate", "nan", "none_metric", "extra_method",
                                    "extra_match", "fractional_seed"])
def test_validate_grid_rejects_a_damaged_grid(damage):
    rows = _rows({})
    if damage == "drop_seed":
        rows = [r for r in rows if not (r["match"] == "J03WQQ" and r["seed"] == 2
                                         and r["method"] == "m1_lomo" and r["preset"] == "doc")]
    elif damage == "duplicate":
        rows.append(dict(rows[5]))
    elif damage == "nan":
        rows[3]["d_f1"] = float("nan")
    elif damage == "none_metric":
        rows[3]["spell_f1_mean"] = None
    elif damage == "extra_method":
        rows.append(dict(rows[0], method="fallback"))
    elif damage == "extra_match":
        rows.append(dict(rows[0], match="J03XXX"))
    elif damage == "fractional_seed":
        rows[0]["seed"] = 0.5
    with pytest.raises(ValueError):
        experiment.validate_grid(rows)
    with pytest.raises(ValueError):
        experiment.go_criteria(rows)


def test_validate_grid_accepts_float_seeds_from_csv():
    rows = [dict(r, seed=float(r["seed"])) for r in _rows({})]
    assert experiment.validate_grid(rows) == 189


def test_check_held_out():
    good = {"train": ["A", "B"], "test": ["C"]}
    experiment.check_held_out(good, "C")
    for bundle, match in [(good, "A"), (good, "D"), ({"train": ["A", "C"], "test": ["C"]}, "C"),
                          ({"train": [], "test": ["C"]}, "C"), ({}, "C")]:
        with pytest.raises(ValueError):
            experiment.check_held_out(bundle, match)


def test_evaluate_match_rejects_a_model_trained_on_the_match(tmp_path):
    import joblib
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": None, "threshold": 0.5, "train": ["J03WMX", "J03WN1"], "test": ["J03WN1"]},
                path)
    with pytest.raises(ValueError):
        experiment.evaluate_match("J03WN1", {"m1_lomo": str(path)}, str(tmp_path), str(tmp_path))
