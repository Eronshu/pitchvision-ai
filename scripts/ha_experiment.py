"""Run the HA1 experiment (docs/superpowers/specs/2026-11-06-pitchvision-ha-possession-design.md).

    python scripts/ha_experiment.py --stage dev --run-id ha1_001
    python scripts/ha_experiment.py --stage lockbox --dev-run ha1_001      # ONCE, after approval

Every run writes into its own new directory $IDSSE_DIR/v3/ha/runs/<run-id>/ (outside the
repository) and refuses to reuse one: earlier runs are never overwritten. Each run directory holds
MANIFEST.json with the source hashes, configuration, git state, frozen inputs and output hashes.
The frozen M1 models and report in $IDSSE_DIR/v2/m1 are only read.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402

from pvext import idsse  # noqa: E402
from pvext.ha import criteria  # noqa: E402
from pvext.ha import experiment as hx  # noqa: E402
from pvext.v2 import m1  # noqa: E402

STRINGS = ("match", "stage", "preset", "method", "input_sha256", "prediction_sha256")
SOURCES = ("pvext/ha/features.py", "pvext/ha/decode.py", "pvext/ha/experiment.py",
           "scripts/ha_experiment.py", "pvext/ha/criteria.py", "pvext/v2/experiment.py", "pvext/v2/m1.py",
           "pvext/v2/features.py", "pvext/benchmark.py", "pvext/degrade.py", "pvext/convert.py",
           "pitchvision/features.py", "pitchvision/possession.py", "pitchvision/alive.py",
           "evaluation/possession_metrics.py", "configs/v1.json")


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with open(path) as handle:
        return [{k: (v if k in STRINGS else (None if v in ("", "None") else float(v)))
                 for k, v in r.items()} for r in csv.DictReader(handle)]


def _new_dir(path):
    if os.path.exists(path):
        raise SystemExit("{} already exists; runs are never overwritten".format(path))
    os.makedirs(path)
    return path


def _provenance(argv, m1_dir):
    return {"started": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": [sys.executable, os.path.relpath(__file__, ROOT)] + list(argv),
            "git_head": _git("rev-parse", "HEAD"), "git_status": _git("status", "--porcelain"),
            "source_sha256": {p: _sha(os.path.join(ROOT, p)) for p in SOURCES},
            "frozen_inputs_sha256": {
                "m1_report": _sha(os.path.join(m1_dir, "report", "per_run.csv")),
                **{os.path.basename(p): _sha(p) for p in sorted(
                    os.path.join(m1_dir, "models", f) for f in os.listdir(os.path.join(m1_dir, "models")))}},
            "python": platform.python_version()}


def _config(variant):
    return {"variant": variant, "dev": list(hx.DEV), "lockbox": list(hx.LOCKBOX),
            "train_presets": list(m1.TRAIN_PRESETS), "train_seeds": list(m1.TRAIN_SEEDS),
            "eval_presets": list(m1.EVAL_PRESETS), "eval_seeds": list(m1.EVAL_SEEDS),
            "stride": hx.STRIDE, "feature_set": hx.feature_set(variant), "lambdas": list(hx.VARIANTS[variant]["lambdas"]), "lambda_tie": "larger",
            "model": hx.MODELS[hx.VARIANTS[variant]["model"]],
            "decoder": "two-state Viterbi per M1-alive segment, dt-scaled log evidence, uniform start",
            "dead_mask": {"dev": "frozen M1 lomo_<match>", "lockbox": "frozen M1 " + hx.LOCKBOX_M1}}


def _summary(rows, stage, matches):
    ref = hx.REF[stage]
    out = {}
    for m in matches:
        def mean(method, key, preset="doc"):
            v = [r[key] for r in rows if r["match"] == m and r["method"] == method and r["preset"] == preset]
            v = [x for x in v if x is not None]
            return sum(v) / len(v) if v else None
        out[m] = {method: {k: mean(method, k) for k in (
            "spell_f1_mean", "macro_f1", "n_pred_spells", "n_true_spells", "team_accuracy",
            "switch_on_time_share", "spurious_switch_share", "n_pred_switches", "n_true_switches")}
            for method in ("v1", ref, "m1_ha1")}
        for preset in ("mild", "severe"):
            out[m]["gain_" + preset] = mean("m1_ha1", "spell_f1_mean", preset) - mean(ref, "spell_f1_mean", preset)
        out[m]["gain_doc"] = out[m]["m1_ha1"]["spell_f1_mean"] - out[m][ref]["spell_f1_mean"]
    return out


def _finish(run, rows, csv_name, extra, provenance):
    _write(rows, os.path.join(run, csv_name))
    provenance["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    provenance["outputs_sha256"] = {}
    for dirpath, _, files in os.walk(run):
        for f in sorted(files):
            if f != "MANIFEST.json":
                p = os.path.join(dirpath, f)
                provenance["outputs_sha256"][os.path.relpath(p, run)] = _sha(p)
    provenance.update(extra)
    with open(os.path.join(run, "MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, default=float)


# Orchestration/reporting only: may differ from the dev run. Every other recorded source must be
# byte-identical to the dev run's manifest before the lockbox runs.
NOT_PREDICTION = ("scripts/ha_experiment.py", "pvext/ha/criteria.py")


def check_frozen_sources(dev_manifest):
    recorded = dev_manifest["source_sha256"]
    bad = [p for p, h in recorded.items() if p not in NOT_PREDICTION and _sha(os.path.join(ROOT, p)) != h]
    missing = [p for p in SOURCES if p not in recorded and p not in NOT_PREDICTION]
    if bad or missing:
        raise SystemExit("sources differ from the dev run ({}); the lockbox must use the frozen "
                         "candidate's code".format(", ".join(bad + missing)))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "lockbox"))
    parser.add_argument("--run-id", help="dev: new run directory name")
    parser.add_argument("--dev-run", help="lockbox: the reviewed dev run")
    parser.add_argument("--variant", default="ha1", choices=sorted(hx.VARIANTS))
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    m1_dir = os.path.join(args.root, "v2", "m1")
    m1_report = os.path.join(m1_dir, "report", "per_run.csv")
    runs = os.path.join(idsse.check_root(os.path.join(args.root, "v3", "ha")), "runs")

    if args.stage == "dev":
        if not args.run_id or os.path.basename(args.run_id) != args.run_id:
            raise SystemExit("--run-id must be a plain new directory name")
        run = _new_dir(os.path.join(runs, args.run_id))
        prov = _provenance(argv, m1_dir)
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(hx.build_training, hx.DEV, [args.root] * 5, [run] * 5,
                          [hx.feature_set(args.variant)] * 5))
            folds = [([m for m in hx.DEV if m != held], [held]) for held in hx.DEV]
            paths = list(pool.map(hx.train, ["lomo"] * 5, [f[0] for f in folds],
                                  [f[1] for f in folds], [run] * 5, [None] * 5,
                                  [args.variant] * 5))
            m1_paths = [os.path.join(m1_dir, "models", "lomo_{}.joblib".format(m)) for m in hx.DEV]
            results = list(pool.map(hx.evaluate_match, hx.DEV, ["dev"] * 5, paths, m1_paths,
                                    [args.root] * 5, [run] * 5, [m1_report] * 5))
        rows = [r for rs in results for r in rs]
        hx.validate_grid(rows, "dev")
        folds_out = {}
        for p in paths:
            b = joblib.load(p)
            folds_out[os.path.basename(p)] = {"train": b["train"], "test": b["test"],
                                              "lam": b["lam"], "curve": b["curve"]}
        summary = _summary(rows, "dev", hx.DEV)
        _finish(run, rows, "dev_per_run.csv",
                {"stage": "dev", "config": _config(args.variant), "folds": folds_out, "summary": summary,
                 "attribution": idsse.ATTRIBUTION}, prov)
        print(json.dumps({m: round(v["gain_doc"], 4) for m, v in summary.items()}))
        return 0

    if not args.dev_run:
        raise SystemExit("--dev-run is required for the lockbox")
    dev_run = os.path.join(runs, args.dev_run)
    dev_report = os.path.join(dev_run, "dev_per_run.csv")
    if not os.path.exists(dev_report):
        raise SystemExit("run and review the dev stage first")
    lock = os.path.join(dev_run, "lockbox")
    if os.path.exists(lock):
        raise SystemExit("the lockbox has already been evaluated; it is evaluated once only")
    with open(os.path.join(dev_run, "MANIFEST.json")) as handle:
        dev_manifest = json.load(handle)
    check_frozen_sources(dev_manifest)
    variant = dev_manifest["config"]["variant"]
    _new_dir(lock)
    prov = _provenance(argv, m1_dir)
    path = hx.train("final_dev", list(hx.DEV), list(hx.LOCKBOX), dev_run, out=lock, variant=variant)
    m1_path = os.path.join(m1_dir, "models", hx.LOCKBOX_M1)
    with ProcessPoolExecutor(2) as pool:
        results = list(pool.map(hx.evaluate_match, hx.LOCKBOX, ["lockbox"] * 2, [path] * 2,
                                [m1_path] * 2, [args.root] * 2, [lock] * 2, [m1_report] * 2))
    rows = [r for rs in results for r in rs]
    go = criteria.report(_read(dev_report), rows)
    _finish(lock, rows, "lockbox_per_run.csv",
            {"stage": "lockbox", "config": _config(variant), "go": go, "attribution": idsse.ATTRIBUTION}, prov)
    print("verdict (original, pre-registered):", go["verdict_original"])
    print("verdict (amended criterion 4, post hoc):", go["verdict_amended"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
