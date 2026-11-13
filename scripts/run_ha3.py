"""Write the PitchVision HA3 (M1 dead-ball model of the HA3 lockbox + HA3 team assignment) submission.

    python scripts/run_ha3.py --config configs/ha3.json --out /path/outside/repo/pv_ha3

ATD -> M1 features -> the dev-trained "league" M1 dead-ball model the HA3 lockbox was evaluated with
-> V1 persistence -> dead mask -> HA3 features -> symmetric gradient boosting -> switch-penalty
Viterbi per alive segment -> H/A/D/X. The M1 release (configs/v2.json, the fallback), the dead-ball
model and the HA3 model are all checked before any competition data is read (pvext/ha/deploy.py). The data location comes from
games.py (MTAH_DATA_DIR), as for run_v1.py / run_v2.py.

Outputs go to --out only: the CSV (team name) and manifest.json with the run_v2.py keys plus the
HA3 release, so scripts/verify_submission.py checks it unchanged. Exits non-zero if the file fails
the official validation. scripts/run_v2.py (frozen M1) stays the fallback and is not modified.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from pitchvision import atd, config  # noqa: E402
from pvext.ha import deploy  # noqa: E402
from starter import baseline  # noqa: E402

TEAM_FILE = "PitchVision AI.csv"
SOURCES = ("scripts/run_ha3.py", "pvext/ha/deploy.py", "pvext/ha/features.py", "pvext/ha/decode.py",
           "pvext/ha/experiment.py", "pvext/v2/deploy.py", "pvext/v2/features.py", "pvext/v2/m1.py",
           "pitchvision/atd.py", "pitchvision/features.py", "pitchvision/possession.py",
           "pitchvision/alive.py", "configs/v1.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "ha3.json"),
                        help="HA3 release config (configs/ha3.json)")
    parser.add_argument("--out", required=True, help="output directory, outside the repository")
    parser.add_argument("--output-name", default=TEAM_FILE, help="CSV file name (team name)")
    parser.add_argument("--atd-meta", help=argparse.SUPPRESS)
    parser.add_argument("--atd-raw", help=argparse.SUPPRESS)
    parser.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--blocks", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rel(path):
    path = os.path.realpath(path)
    root = os.path.realpath(ROOT)
    return os.path.relpath(path, root) if path.startswith(root + os.sep) else path


def main(argv=None):
    args = parse_args(argv)
    out = os.path.realpath(args.out)
    if out == os.path.realpath(ROOT) or out.startswith(os.path.realpath(ROOT) + os.sep):
        print("--out must be outside the repository", file=sys.stderr)
        return 2
    if os.path.basename(args.output_name) != args.output_name or not args.output_name.endswith(".csv"):
        print("--output-name must be a plain .csv file name", file=sys.stderr)
        return 2
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]

    # stage 1: both releases and models, before any competition data is touched
    try:
        release = deploy.load_release(args.config, fps)
    except deploy.ReleaseError as error:
        print("release check failed: {}".format(error), file=sys.stderr)
        return 3

    if args.atd_meta and args.atd_raw:
        meta_path, raw_path = args.atd_meta, args.atd_raw
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = args.n_frames or source_frames("tactical", GAME)
    blocks = json.loads(args.blocks) if args.blocks else half_video_blocks(GAME)
    tr = atd.load_arrays(meta_path, raw_path, n_frames, pitch)

    # stage 2: features, prediction and their checks
    try:
        labels, info, m1_labels = deploy.label(release, tr, pitch, blocks)
    except (deploy.ReleaseError, deploy.m1_deploy.ReleaseError) as error:
        print("inference check failed: {}".format(error), file=sys.stderr)
        return 4

    os.makedirs(out, exist_ok=True)
    path = baseline.write_submission(os.path.join(out, args.output_name), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)

    m1_rel = release.m1
    run_config = {"ha3_release": release.config, "release": m1_rel.config, "v1": m1_rel.v1}
    status = _git("status", "--porcelain")
    manifest = {
        "mode": "v2_m1_ha3",
        "config": run_config,
        "config_sha256": config.sha256(run_config),
        "ha3_release_config_path": _rel(args.config),
        "ha3_release_config_sha256": _sha256(args.config),
        "ha3_model_sha256": _sha256(release.model_path),
        "ha3_lam": release.config["lam"],
        "ha3_feature_names_sha256": release.config["feature_names_sha256"],
        "dead_model_path": _rel(release.dead_path),
        "dead_model_sha256": _sha256(release.dead_path),
        "dead_model_kind": release.dead["kind"],
        "dead_model_train": list(release.dead["train"]),
        "release_config_path": _rel(deploy._path(release.config["m1_release_config"])),
        "release_config_sha256": _sha256(deploy._path(release.config["m1_release_config"])),
        "model_sha256": _sha256(m1_rel.model_path),
        "feature_names_sha256": m1_rel.config["feature_names_sha256"],
        "threshold": m1_rel.config["threshold"],
        "decision_margin_min": info["decision_margin_min"],
        "ha3_p_margin_min": info["ha3_p_margin_min"],
        "ha3_changed_share_alive": info["ha3_changed_share_alive"],
        "n_spells": {"m1": info["n_spells_m1"], "ha3": info["n_spells_ha3"]},
        "source_sha256": {p: _sha256(os.path.join(ROOT, p)) for p in SOURCES},
        "dependencies": release.config["dependencies"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "inputs": {os.path.basename(p): os.path.getsize(p) for p in (meta_path, raw_path)},
        "blocks": blocks,
        "output": args.output_name,
        "output_sha256": _sha256(path),
        "n_rows": int(len(labels)),
        "label_counts": {k: int(np.sum(labels == k)) for k in ("H", "A", "D", "X")},
        "m1_dead_model_label_counts": {k: int(np.sum(m1_labels == k)) for k in ("H", "A", "D", "X")},
        "valid": reason is None,
    }
    with open(os.path.join(out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if reason:
        print("INVALID {}: {}".format(path, reason), file=sys.stderr)
        return 1
    print("wrote {} ({} rows) and manifest.json".format(path, len(labels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
