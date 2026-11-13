"""Run the V2 M1 experiment end to end (design spec: M1 section, folds, go criteria).

    python scripts/v2_m1_experiment.py                      # build -> train -> evaluate -> report
    python scripts/v2_m1_experiment.py --steps evaluate,report

Writes training caches, fold models and reports under $IDSSE_DIR/v2/m1 (outside the repository):
  report/per_run.csv         one row per match x preset x seed x method (v1, m1_lomo, m1_league)
  report/summary.json        per-match means, paired differences against V1, go criteria, the
                             missed-start error analysis, thresholds, and the attribution
The evaluation uses only held-out matches with the frozen presets and seeds (0, 1, 2); training
uses seeds 100 and 101. These are IDSSE / DFL labels, not competition performance.
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from pvext import idsse  # noqa: E402
from pvext.v2 import experiment, m1  # noqa: E402

METRICS = ("d_precision", "d_recall", "d_f1", "boundary_dead_start_tol25_f1",
           "boundary_dead_end_tol25_f1", "boundary_dead_start_tol50_f1", "alive_dead_f1",
           "spell_f1_mean", "d_share_pred", "d_share_true")


def _git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--steps", default="build,train,evaluate,report")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args(argv)
    work = idsse.check_root(os.path.join(args.root, "v2", "m1"))
    os.makedirs(work, exist_ok=True)
    steps = args.steps.split(",")
    folds = {("lomo", tuple(test)): train for train, test in m1.folds("lomo")}
    folds.update({("league", tuple(test)): train for train, test in m1.folds("league")})

    if "build" in steps:
        with ProcessPoolExecutor(args.workers) as pool:
            for match, paths in zip(m1.MATCHES, pool.map(
                    experiment.build_training, m1.MATCHES, [args.root] * 7, [work] * 7)):
                print("built", match, len(paths), "training copies", flush=True)

    if "train" in steps:
        keys = list(folds)
        with ProcessPoolExecutor(min(args.workers, len(keys))) as pool:
            for key, path in zip(keys, pool.map(
                    experiment.train_fold, [k[0] for k in keys], [folds[k] for k in keys],
                    [list(k[1]) for k in keys], [work] * len(keys))):
                print("trained", key[0], "-".join(key[1]), "->", os.path.basename(path), flush=True)

    model_dir = os.path.join(work, "models")
    if "evaluate" in steps:
        def models_for(match):
            league = next(k for k in folds if k[0] == "league" and match in k[1])
            return {"m1_lomo": os.path.join(model_dir, "lomo_{}.joblib".format(match)),
                    "m1_league": os.path.join(model_dir, "league_{}.joblib".format("-".join(league[1])))}
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(experiment.evaluate_match, m1.MATCHES,
                                    [models_for(m) for m in m1.MATCHES], [args.root] * 7, [work] * 7))
        rows = [r for match_rows, _ in results for r in match_rows]
        starts = {m: s for m, (_, s) in zip(m1.MATCHES, results)}
        os.makedirs(os.path.join(work, "report"), exist_ok=True)
        with open(os.path.join(work, "report", "per_run.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(work, "report", "starts.json"), "w") as handle:
            json.dump(starts, handle, indent=2)
        print("evaluated", len(rows), "runs", flush=True)

    if "report" in steps:
        with open(os.path.join(work, "report", "per_run.csv")) as handle:
            rows = [{k: (float(v) if k not in ("match", "preset", "method", "input_sha256",
                                               "prediction_sha256") and v not in ("", "None")
                         else (None if v in ("", "None") else v)) for k, v in r.items()}
                    for r in csv.DictReader(handle)]
        with open(os.path.join(work, "report", "starts.json")) as handle:
            starts = json.load(handle)
        per = {method: {preset: {key: experiment.per_match(rows, method, preset, key)
                                 for key in METRICS}
                        for preset in m1.EVAL_PRESETS}
               for method in ("v1", "m1_lomo", "m1_league")}
        thresholds = {os.path.basename(p): joblib.load(os.path.join(model_dir, p))["threshold"]
                      for p in sorted(os.listdir(model_dir))}
        summary = {
            "exploratory": True,
            "note": "IDSSE / DFL labels; not an estimate of competition performance.",
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "train_presets": m1.TRAIN_PRESETS, "train_seeds": m1.TRAIN_SEEDS,
            "eval_presets": m1.EVAL_PRESETS, "eval_seeds": m1.EVAL_SEEDS,
            "thresholds": thresholds, "per_match": per,
            "go": experiment.go_criteria(rows), "starts_doc": starts,
            "attribution": idsse.ATTRIBUTION,
        }
        with open(os.path.join(work, "report", "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)
        print("verdict:", summary["go"]["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
