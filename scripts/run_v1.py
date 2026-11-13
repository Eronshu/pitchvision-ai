"""Write a PitchVision V1 submission and a manifest that proves where it came from.

    python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_v1
    python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_v1 --fallback-baseline

The data location comes from games.py (MTAH_DATA_DIR). --fallback-baseline writes the official
baseline_smooth50 through the official code, so the fallback stays a verbatim copy of the
published floor, not a re-implementation of it.

Outputs go to --out and nowhere else: submission.csv (or baseline_smooth50.csv) and
manifest.json. The run exits non-zero if the file fails the official validation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from pitchvision import atd, config, pipeline  # noqa: E402
from starter import baseline, load_smart_tagging  # noqa: E402

# Params
FALLBACK_WINDOW = 50  # the published floor, baseline_smooth50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/*.json")
    parser.add_argument("--out", required=True, help="output directory (created if missing)")
    parser.add_argument("--fallback-baseline", action="store_true",
                        help="write the official baseline_smooth50 instead of V1")
    # Test-only overrides: synthetic fixtures instead of the match bundle.
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
    """True / False for a dirty / clean working tree, None when git cannot tell (no checkout)."""
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def main(argv=None):
    args = parse_args(argv)
    cfg = config.load(args.config)
    data_dir = args.data_dir
    events = team_names = tag_path = None
    if pipeline.uses_tagging(cfg) and not args.fallback_baseline:
        try:
            tag_path = require(["smart_tagging"], GAME, data_dir)[0]
        except IOError as error:
            print("smart_tagging: {}".format(error), file=sys.stderr)
            return 2
        events = load_smart_tagging.load(GAME, data_dir)
    if args.atd_meta and args.atd_raw:
        meta_path, raw_path = args.atd_meta, args.atd_raw
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = args.n_frames or source_frames("tactical", GAME)
    blocks = json.loads(args.blocks) if args.blocks else half_video_blocks(GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]

    if args.fallback_baseline:
        mode, name = "fallback_baseline_smooth50", "baseline_smooth50.csv"
        codes = baseline.nearest_player_labels(raw_path, baseline.team_of_each_track(meta_path),
                                               n_frames, pitch)
        labels = baseline.smooth(codes, FALLBACK_WINDOW)
    else:
        mode, name = "v1", "submission.csv"
        if events is not None:
            team_names = baseline.team_names(meta_path)
        labels = pipeline.run(atd.load_arrays(meta_path, raw_path, n_frames, pitch),
                              cfg, fps, pitch, blocks, events, team_names)

    os.makedirs(args.out, exist_ok=True)
    path = baseline.write_submission(os.path.join(args.out, name), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), n_frames)

    manifest = {
        "mode": mode,
        "config": cfg,
        "config_sha256": config.sha256(cfg),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _dirty(),
        "inputs": {os.path.basename(p): os.path.getsize(p)
                   for p in (meta_path, raw_path, tag_path) if p},
        "output": name,
        "output_sha256": _sha256(path),
        "n_rows": int(len(labels)),
        "label_counts": {k: int(np.sum(labels == k)) for k in ("H", "A", "D", "X")},
        "valid": reason is None,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if reason:
        print("INVALID {}: {}".format(path, reason), file=sys.stderr)
        return 1
    print("wrote {} ({} rows) and manifest.json".format(path, len(labels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
