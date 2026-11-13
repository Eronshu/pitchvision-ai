"""Amended criterion 4 is reported beside, never instead of, the pre-registered one. Synthetic."""

import json

import pytest

from pvext.ha import criteria
from pvext.ha import experiment as hx
from pvext.v2 import m1
from tests.ext.ha.test_experiment import _cli


def _rows(gain=0.06, r_ha=0.68, r_m1=0.66, other=0.01):
    """Every match: candidate ratio r_ha, M1 ratio r_m1 (dicts or numbers)."""
    rows = []
    for stage, matches in (("dev", hx.DEV), ("lockbox", hx.LOCKBOX)):
        ref = hx.REF[stage]
        for m in matches:
            rh = r_ha[m] if isinstance(r_ha, dict) else r_ha
            rm = r_m1[m] if isinstance(r_m1, dict) else r_m1
            for preset in m1.EVAL_PRESETS:
                g = gain if preset == "doc" else other
                for seed in m1.EVAL_SEEDS:
                    for method, val, r in ((ref, 0.4, rm), ("m1_ha1", 0.4 + g, rh)):
                        rows.append({"match": m, "stage": stage, "preset": preset, "seed": seed,
                                     "method": method, "spell_f1_mean": val,
                                     "n_pred_spells": 100 * r, "n_true_spells": 100})
    return [r for r in rows if r["stage"] == "dev"], [r for r in rows if r["stage"] == "lockbox"]


def test_original_no_go_is_never_relabelled():
    dev, lock = _rows(r_ha=0.68, r_m1=0.66)          # below 0.7 but no worse than M1
    out = criteria.report(dev, lock)
    assert out["original"]["criteria"]["4_spell_count_ratio_0.7_1.5"]["pass"] is False
    assert out["verdict_original"] == out["original"]["verdict"] == "WITHIN_NOISE_KEEP_M1"
    assert out["amended_criterion_4"]["pass"] is True and out["amended_criterion_4"]["post_hoc"]
    assert out["verdict_amended"] == "GO_UNDER_AMENDED_CRITERION_4"
    assert criteria.original_4(dev, lock)["pass"] is False


@pytest.mark.parametrize("r_ha,r_m1,ok", [
    (0.68, 0.66, True),     # closer to 1 than M1
    (0.64, 0.66, True),     # 0.36 <= 0.34 + 0.02
    (0.63, 0.66, False),    # 0.37 >  0.36
    (1.40, 1.45, True),
    (1.52, 1.60, False),    # above the hard 1.5 cap
    (1.0, 1.0, True),
])
def test_amended_rule(r_ha, r_m1, ok):
    dev, lock = _rows(r_ha=r_ha, r_m1=r_m1)
    out = criteria.amended_4(dev, lock)
    assert out["pass"] is ok
    v = out["value"]["J03WMX"]
    assert v["r_ha"] == pytest.approx(r_ha) and v["r_m1"] == pytest.approx(r_m1)


def test_amended_rule_is_per_match():
    r_ha = {m: 0.9 for m in hx.DEV + hx.LOCKBOX}
    r_ha["J03WOY"] = 0.60
    dev, lock = _rows(r_ha=r_ha, r_m1=0.66)
    out = criteria.amended_4(dev, lock)
    assert out["pass"] is False and out["value"]["J03WOY"]["pass"] is False
    assert out["value"]["J03WPY"]["pass"] is True


def test_other_criteria_still_gate_the_amended_verdict():
    dev, lock = _rows(gain=0.01)                      # dev mean below +0.04
    assert criteria.report(dev, lock)["verdict_amended"] == "WITHIN_NOISE_KEEP_M1"
    dev, lock = _rows(gain=-0.01)
    out = criteria.report(dev, lock)
    assert out["verdict_original"] == out["verdict_amended"] == "NO-GO_KEEP_M1"


def test_dev_only_evaluation():
    dev, _ = _rows()
    out = criteria.amended_4(dev)
    assert set(out["value"]) == set(hx.DEV)


def test_lockbox_refuses_changed_prediction_sources(tmp_path):
    cli = _cli()
    recorded = {p: cli._sha(cli.os.path.join(cli.ROOT, p)) for p in cli.SOURCES}
    cli.check_frozen_sources({"source_sha256": dict(recorded)})          # identical: allowed
    changed = dict(recorded, **{"pvext/ha/features.py": "0" * 64})
    with pytest.raises(SystemExit, match="features.py"):
        cli.check_frozen_sources({"source_sha256": changed})
    for p in cli.NOT_PREDICTION:                                        # reporting code may differ
        cli.check_frozen_sources({"source_sha256": dict(recorded, **{p: "0" * 64})})
    missing = {p: h for p, h in recorded.items() if p != "pvext/ha/decode.py"}
    with pytest.raises(SystemExit, match="decode.py"):
        cli.check_frozen_sources({"source_sha256": missing})


def test_lockbox_checks_sources_before_creating_its_directory(tmp_path):
    cli = _cli()
    (tmp_path / "v2" / "m1").mkdir(parents=True)
    run = tmp_path / "v3" / "ha" / "runs" / "ha3_004"
    run.mkdir(parents=True)
    (run / "dev_per_run.csv").write_text("x\n")
    (run / "MANIFEST.json").write_text(json.dumps(
        {"source_sha256": {"pvext/ha/features.py": "0" * 64}, "config": {"variant": "ha3"}}))
    with pytest.raises(SystemExit, match="sources differ"):
        cli.main(["--stage", "lockbox", "--dev-run", "ha3_004", "--root", str(tmp_path)])
    assert not (run / "lockbox").exists()
