"""Run the M1 transfer-robustness study (design spec: "Robustness study").

    python scripts/v2_m1_robustness.py                    # train -> evaluate -> report
    python scripts/v2_m1_robustness.py --steps report

Reads the frozen M1 training caches and models in $IDSSE_DIR/v2/m1 (never written) and writes
everything new under $IDSSE_DIR/v2/m1_robust:
  models/                ablation models (variant x lomo/league folds)
  report/per_run.csv     match x condition x seed x method (v1, fallback, M1 variants)
  report/summary.json    reproducibility check against the frozen M1 run, GO criteria per ablation
                         (reference), alternative-distribution tables and the pre-declared reading
IDSSE / DFL labels only; not an estimate of competition performance.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pvext import idsse  # noqa: E402
from pvext.v2 import experiment, m1, robust  # noqa: E402

VARIANTS = ("no_ball_presence", "no_slot_speed", "no_both")
KEYS = ("d_precision", "d_recall", "d_f1", "boundary_dead_start_tol25_f1",
        "boundary_dead_start_tol50_f1", "boundary_dead_end_tol25_f1", "boundary_dead_end_tol50_f1",
        "alive_dead_f1", "spell_f1_mean", "d_share_pred", "d_share_true")
STRINGS = ("match", "distribution", "preset", "method", "input_sha256", "prediction_sha256")


def _folds():
    out = [("lomo", train, test) for train, test in m1.folds("lomo")]
    return out + [("league", train, test) for train, test in m1.folds("league")]


def _model_paths(match, frozen_models, models):
    paths = {}
    for kind, train, test in _folds():
        if match not in test:
            continue
        tag = "-".join(test)
        paths["m1_full_" + kind] = os.path.join(frozen_models, "{}_{}.joblib".format(kind, tag))
        for variant in VARIANTS:
            paths["m1_{}_{}".format(variant, kind)] = os.path.join(
                models, "{}_{}_{}.joblib".format(variant, kind, tag))
    return paths


def _read(path):
    with open(path) as handle:
        return [{k: (v if k in STRINGS else (None if v in ("", "None") else float(v)))
                 for k, v in r.items()} for r in csv.DictReader(handle)]


def _git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def report(rows, frozen_report):
    """Validate the grid, require the frozen M1 run to be reproduced exactly, then summarise.
    Raises ValueError on any grid problem or any input/prediction hash mismatch."""
    robust.validate_grid(rows)
    frozen_rows = _read(frozen_report)
    experiment.validate_grid(frozen_rows)
    # 1. reproducibility: the frozen methods on the frozen inputs must reproduce the M1 run exactly
    rename = {"v1": "v1", "m1_full_lomo": "m1_lomo", "m1_full_league": "m1_league"}
    old = {(r["match"], r["preset"], int(r["seed"]), r["method"]):
           (r["input_sha256"], r["prediction_sha256"]) for r in frozen_rows}
    checked = mismatched = 0
    for r in rows:
        if r["distribution"] == "frozen" and r["method"] in rename:
            key = (r["match"], r["preset"], int(r["seed"]), rename[r["method"]])
            checked += 1
            mismatched += old[key] != (r["input_sha256"], r["prediction_sha256"])
    if checked != len(old):
        raise ValueError("reproducibility check covered {} of {} frozen runs".format(checked, len(old)))
    if mismatched:
        raise ValueError("frozen M1 run not reproduced: {} of {} runs differ in input or prediction "
                         "hash".format(mismatched, checked))

    frozen = [r for r in rows if r["distribution"] == "frozen"]
    go = {}
    for variant in ("full",) + VARIANTS:
        sub = [dict(r, method={"v1": "v1", "m1_{}_lomo".format(variant): "m1_lomo",
                               "m1_{}_league".format(variant): "m1_league"}[r["method"]])
               for r in frozen if r["method"] in ("v1", "m1_{}_lomo".format(variant),
                                                  "m1_{}_league".format(variant))]
        go[variant] = experiment.go_criteria(sub)

    methods = sorted({r["method"] for r in rows})
    tables = {preset: {method: {key: experiment.per_match(rows, method, preset, key) for key in KEYS}
                       for method in methods}
              for preset in list(m1.EVAL_PRESETS) + list(robust.ALT_PRESETS)}
    leagues = {preset: {method: {group: {key: float(np.mean([t[key][m] for m in ids]))
                                         for key in KEYS}
                                 for group, ids in (("bundesliga", m1.BUNDESLIGA),
                                                    ("second_division", m1.SECOND_DIVISION))}
                        for method, t in by_method.items()}
               for preset, by_method in tables.items()}
    reading = {preset: {method: robust.robustness_reading(rows, preset, method)
                        for method in methods if method.endswith("_lomo")}
               for preset in robust.ALT_PRESETS}
    return {"exploratory": True,
            "note": "IDSSE / DFL labels; not an estimate of competition performance.",
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "reproducibility": {"frozen_runs_checked": checked, "mismatched": mismatched},
            "ablations": {v: list(robust.ABLATIONS[v]) for v in VARIANTS},
            "alt_presets": {k: {"overrides": v["overrides"], "speed_field": v.get("speed_field")}
                            for k, v in robust.ALT_PRESETS.items()},
            "eval_seeds": m1.EVAL_SEEDS, "go_reference": go, "per_match": tables,
            "per_league": leagues, "alt_reading": reading, "attribution": idsse.ATTRIBUTION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--steps", default="train,evaluate,report")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args(argv)
    src = idsse.check_root(os.path.join(args.root, "v2", "m1"))
    work = idsse.check_root(os.path.join(args.root, "v2", "m1_robust"))
    if os.path.realpath(src) == os.path.realpath(work):
        raise SystemExit("refusing to write into the frozen M1 directory")
    os.makedirs(os.path.join(work, "report"), exist_ok=True)
    steps = args.steps.split(",")

    if "train" in steps:
        jobs = [(v, kind, train, test) for v in VARIANTS for kind, train, test in _folds()]
        with ProcessPoolExecutor(args.workers) as pool:
            for job, path in zip(jobs, pool.map(robust.train_variant, *zip(*jobs),
                                                [src] * len(jobs), [work] * len(jobs))):
                print("trained", job[0], job[1], "-".join(job[3]), flush=True)

    if "evaluate" in steps:
        paths = [_model_paths(m, os.path.join(src, "models"), os.path.join(work, "models"))
                 for m in m1.MATCHES]
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(robust.evaluate_match, m1.MATCHES, paths,
                                    [robust.conditions()] * 7, [args.root] * 7, [work] * 7))
        rows = [r for match_rows in results for r in match_rows]
        with open(os.path.join(work, "report", "per_run.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("evaluated", len(rows), "runs", flush=True)

    if "report" in steps:
        rows = _read(os.path.join(work, "report", "per_run.csv"))
        summary = report(rows, os.path.join(src, "report", "per_run.csv"))
        with open(os.path.join(work, "report", "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)
        print("reproducibility:", summary["reproducibility"], flush=True)
        for variant, result in summary["go_reference"].items():
            print("GO reference", variant, result["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
