"""Amended GO criterion 4 (approved by the project lead on 2026-11-07, AFTER the development
results were seen: post hoc with respect to the development data).

The original pre-registered criterion 4 (spell-count ratio within [0.7, 1.5] on every required
match) is NOT changed; it stays in `experiment.go_criteria` and its verdict is reported unchanged.
This module adds a separately reported, baseline-relative non-regression safeguard:

    r_HA = candidate predicted spells / true spells,  r_M1 = frozen M1 predicted / true spells
    pass  iff  |r_HA - 1| <= |r_M1 - 1| + 0.02  and  r_HA <= 1.5          (doc preset, seed mean)

It does not claim that under-segmentation is solved; it only checks that the candidate does not
segment worse than frozen M1 on any required match.
"""

from pvext.ha import experiment as hx

# Params
TOLERANCE = 0.02
MAX_RATIO = 1.5
NAME = "4_amended_ratio_no_worse_than_m1_plus_0.02"


def ratios(rows, matches, method):
    return {m: hx._mean(rows, m, method, "doc", "n_pred_spells") / hx._mean(rows, m, method, "doc", "n_true_spells")
            for m in matches}


def amended_4(dev_rows, lock_rows=None):
    """Per required match r_HA, r_M1 and pass; the reference is each stage's frozen M1."""
    value, ok = {}, True
    for rows, stage in ((dev_rows, "dev"), (lock_rows, "lockbox")):
        if rows is None:
            continue
        hx.validate_grid(rows, stage)
        matches = hx.DEV if stage == "dev" else hx.LOCKBOX
        r_ha = ratios(rows, matches, "m1_ha1")
        r_m1 = ratios(rows, matches, hx.REF[stage])
        for m in matches:
            p = abs(r_ha[m] - 1) <= abs(r_m1[m] - 1) + TOLERANCE and r_ha[m] <= MAX_RATIO
            value[m] = {"r_ha": r_ha[m], "r_m1": r_m1[m], "pass": bool(p)}
            ok &= p
    return {"value": value, "pass": bool(ok), "post_hoc": True}


def original_4(dev_rows, lock_rows=None):
    """The pre-registered criterion 4, per required match (as in experiment.go_criteria)."""
    value, ok = {}, True
    for rows, stage in ((dev_rows, "dev"), (lock_rows, "lockbox")):
        if rows is None:
            continue
        hx.validate_grid(rows, stage)
        matches = hx.DEV if stage == "dev" else hx.LOCKBOX
        for m, r in ratios(rows, matches, "m1_ha1").items():
            p = 0.7 <= r <= 1.5
            value[m] = {"r_ha": r, "pass": bool(p)}
            ok &= p
    return {"value": value, "pass": bool(ok)}


def report(dev_rows, lock_rows):
    """Original verdict unchanged, plus a separately labelled amended verdict."""
    original = hx.go_criteria(dev_rows, lock_rows)
    c = original["criteria"]
    amended = amended_4(dev_rows, lock_rows)
    others = all(c[k]["pass"] for k in c if not k.startswith("4_"))
    if others and amended["pass"]:
        verdict_amended = "GO_UNDER_AMENDED_CRITERION_4"
    elif original["verdict"] == "NO-GO_KEEP_M1":
        verdict_amended = "NO-GO_KEEP_M1"
    else:
        verdict_amended = "WITHIN_NOISE_KEEP_M1"
    return {"original": original, "verdict_original": original["verdict"],
            "amended_criterion_4": amended, "verdict_amended": verdict_amended,
            "note": "criterion 4 amended post hoc (after dev results); original verdict reported unchanged"}
