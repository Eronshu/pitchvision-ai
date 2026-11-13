"""Write the PitchVision V2 (final M1) submission and a manifest that proves where it came from.

    python scripts/run_v2.py --config configs/v2.json --out /path/outside/repo/pv_v2

ATD -> V2 features -> final M1 -> V1 persistence -> dead mask -> frozen V1 possession machine ->
H/A/D/X. The release config and the model are checked before any competition data is read
(pvext/v2/deploy.py). The data location comes from games.py (MTAH_DATA_DIR), as for run_v1.py.

Outputs go to --out only: the CSV (named after the team, SUBMISSION.md) and manifest.json, whose
keys are those of run_v1.py, so scripts/verify_submission.py checks it unchanged. The run exits
non-zero if the file fails the official validation. Frozen V1 (scripts/run_v1.py) stays the
fallback.
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
from pvext.v2 import deploy  # noqa: E402
from starter import baseline  # noqa: E402

TEAM_FILE = "PitchVision AI.csv"
SOURCES = ("scripts/run_v2.py", "pvext/v2/deploy.py", "pvext/v2/features.py", "pvext/v2/m1.py",
           "pitchvision/atd.py", "pitchvision/features.py", "pitchvision/possession.py",
           "pitchvision/alive.py", "configs/v1.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "v2.json"),
                        help="release config (configs/v2.json)")
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


def _dirty():
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


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

    # stage 1: release and model, before any competition data is touched
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
        labels, info = deploy.label(release, tr, pitch, blocks)
    except deploy.ReleaseError as error:
        print("inference check failed: {}".format(error), file=sys.stderr)
        return 4

    os.makedirs(out, exist_ok=True)
    path = baseline.write_submission(os.path.join(out, args.output_name), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)

    run_config = {"release": release.config, "v1": release.v1}
    config_path = os.path.realpath(args.config)
    if config_path.startswith(os.path.realpath(ROOT) + os.sep):
        config_path = os.path.relpath(config_path, os.path.realpath(ROOT))
    manifest = {
        "mode": "v2_m1",
        "config": run_config,
        "config_sha256": config.sha256(run_config),
        "release_config_path": config_path,
        "release_config_sha256": _sha256(args.config),
        "model_sha256": deploy._sha256(release.model_path),
        "feature_names_sha256": release.config["feature_names_sha256"],
        "threshold": release.config["threshold"],
        "decision_margin_min": info["decision_margin_min"],
        "source_sha256": {p: _sha256(os.path.join(ROOT, p)) for p in SOURCES},
        "dependencies": release.config["dependencies"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _dirty(),
        "inputs": {os.path.basename(p): os.path.getsize(p) for p in (meta_path, raw_path)},
        "blocks": blocks,
        "output": args.output_name,
        "output_sha256": _sha256(path),
        "n_rows": int(len(labels)),
        "label_counts": {k: int(np.sum(labels == k)) for k in ("H", "A", "D", "X")},
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
