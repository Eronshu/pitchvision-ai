"""Write the PitchVision AI release C possession file.

    python scripts/run_astra5_combo.py --config candidate/combo.json \
        --out /path/outside/repo/pv_c --output-name "PitchVision AI.csv"

ATD -> 59 dead-ball features -> HGB dead-ball model; 38 HA5a features -> symmetric gradient
boosting P(home); joint three-state Viterbi -> H/A/D/X. RELEASE.lock, the config, both models,
the Python and dependency versions and every pinned source file (this script and games.py
included) are checked before any match data is read. Outputs go to --out only (never inside a
repository): the CSV and manifest.json. No ground truth is read and no other model is loaded.
"""

import argparse
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from pitchvision import atd  # noqa: E402
from astra5.release import deploy  # noqa: E402
from starter import baseline  # noqa: E402

DEFAULT_NAME = "PitchVision AI.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="candidate config (combo.json)")
    parser.add_argument("--out", required=True, help="output directory, outside every repository")
    parser.add_argument("--output-name", default=DEFAULT_NAME, help="CSV file name")
    parser.add_argument("--atd-meta", help=argparse.SUPPRESS)
    parser.add_argument("--atd-raw", help=argparse.SUPPRESS)
    parser.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--blocks", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _inside_git(path):
    path = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(path, ".git")) and os.path.basename(path) != os.path.basename(os.path.expanduser("~")):
            return True
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent


def main(argv=None):
    args = parse_args(argv)
    out = os.path.realpath(args.out)
    if out == os.path.realpath(ROOT) or out.startswith(os.path.realpath(ROOT) + os.sep) or _inside_git(out):
        print("--out must be outside every repository", file=sys.stderr)
        return 2
    if os.path.basename(args.output_name) != args.output_name or not args.output_name.endswith(".csv"):
        print("--output-name must be a plain .csv file name", file=sys.stderr)
        return 2
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    try:
        release = deploy.load_release(args.config, fps)          # stage 1: before any match data
    except deploy.ReleaseError as error:
        print("release check failed: {}".format(error), file=sys.stderr)
        return 3
    if args.atd_meta and args.atd_raw:
        meta_path, raw_path = args.atd_meta, args.atd_raw
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = args.n_frames or source_frames("tactical", GAME)
    blocks = json.loads(args.blocks) if args.blocks else half_video_blocks(GAME)
    blocks = [tuple(b) for b in blocks]
    tr = atd.load_arrays(meta_path, raw_path, n_frames, pitch)
    try:
        labels, d, h = deploy.label(release, tr, pitch, blocks)  # stage 2
    except deploy.ReleaseError as error:
        print("inference check failed: {}".format(error), file=sys.stderr)
        return 4
    os.makedirs(out, exist_ok=True)
    path = baseline.write_submission(os.path.join(out, args.output_name), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)

    manifest = {"mode": "pitchvision_release_c", "status": release.config["status"],
                "config_sha256": deploy.sha256(args.config),
                "lock_sha256": deploy.sha256(os.path.join(release.base, deploy.LOCK)), "joint": {k: release.config["joint"][k] for k in ("lam_ha", "lam_on", "lam_off")},
                "dead_model_sha256": release.config["dead_model"]["model_sha256"],
                "ha_model_sha256": release.config["ha_model"]["model_sha256"],
                "source_sha256": deploy.source_hashes(), "dependencies": release.config["dependencies"],
                "python": platform.python_version(), "platform": platform.platform(),
                "inputs": {os.path.basename(p): os.path.getsize(p) for p in (meta_path, raw_path)},
                "blocks": blocks, "output": args.output_name, "output_sha256": deploy.sha256(path),
                "n_rows": int(len(labels)), "valid": reason is None}
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
