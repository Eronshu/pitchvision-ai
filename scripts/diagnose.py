"""L1 diagnostics for a submission: plausibility statistics and stability, aggregates only.

    python scripts/diagnose.py --config configs/v1.json \\
        --submission /tmp/pv_v1/submission.csv \\
        --baseline /tmp/pv_fb/baseline_smooth50.csv --out /tmp/pv_v1/diagnostics.json

Reports, per half and in total, for the submission and for the fallback baseline: possession
count, duration percentiles, share of spells under 1 s, transitions per minute, dead-ball share
and run length, home share. Then agreement of the submission with the baseline, and stability:
the pipeline is rerun with 10 % of the ball observations removed (fixed seed) and compared with
its own unperturbed output. When the Smart Tagging file is present, L2: how predicted restarts
line up with SET PIECES tags, each figure marked independent or not (a tag family the model used
as input is not independent evidence).

None of this is accuracy, and no ground truth is read. The output holds aggregates only, but it
is still derived from the match data: check the data licence before sharing it with anyone.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from pitchvision import atd, config, diagnostics, pipeline, tagging  # noqa: E402
from starter import baseline, load_smart_tagging  # noqa: E402

# Params
PERTURB_FRAC = 0.1
PERTURB_SEED = 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the config the submission was made with")
    parser.add_argument("--submission", required=True, help="V1 submission CSV")
    parser.add_argument("--baseline", required=True, help="fallback baseline_smooth50 CSV")
    parser.add_argument("--out", required=True, help="JSON report to write")
    parser.add_argument("--atd-meta", help=argparse.SUPPRESS)
    parser.add_argument("--atd-raw", help=argparse.SUPPRESS)
    parser.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--blocks", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    if args.atd_meta and args.atd_raw:
        meta_path, raw_path = args.atd_meta, args.atd_raw
    else:
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    try:
        require(["smart_tagging"], GAME, args.data_dir)
        events = load_smart_tagging.load(GAME, args.data_dir)
    except IOError:
        if pipeline.uses_tagging(cfg):
            raise
        events = None   # L2 is skipped; L1 does not need the tagging
    team_names = baseline.team_names(meta_path)
    n_frames = args.n_frames or source_frames("tactical", GAME)
    blocks = json.loads(args.blocks) if args.blocks else half_video_blocks(GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]

    labels = {}
    for name, path in (("submission", args.submission), ("baseline", args.baseline)):
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, n_frames)
        if reason:
            print("INVALID {}: {}".format(path, reason), file=sys.stderr)
            return 1
        labels[name] = table["label"].astype(str).str.strip().str.upper().values

    tr = atd.load_arrays(meta_path, raw_path, n_frames, pitch)
    rerun = pipeline.run(tr, cfg, fps, pitch, blocks, events, team_names)
    perturbed = pipeline.run(diagnostics.perturb_ball(tr, PERTURB_FRAC, PERTURB_SEED),
                             cfg, fps, pitch, blocks, events, team_names)

    report = {
        "config_sha256": config.sha256(cfg),
        "submission_matches_config": bool((rerun == labels["submission"]).all()),
        "submission": diagnostics.summarize(labels["submission"], blocks, fps),
        "baseline": diagnostics.summarize(labels["baseline"], blocks, fps),
        "agreement_with_baseline": diagnostics.agreement(labels["submission"],
                                                         labels["baseline"], blocks),
        "stability": diagnostics.agreement(perturbed, rerun, blocks),
        "perturbation": {"ball_frac_removed": PERTURB_FRAC, "seed": PERTURB_SEED},
        "l2_setpieces": None,
    }
    if events is not None:
        tmap = tagging.team_map(tagging.tag_teams(events), team_names)
        report["l2_setpieces"] = {
            "submission": diagnostics.setpiece_consistency(
                labels["submission"], events, cfg, blocks, tmap),
            # The baseline never reads the tagging, so its L2 is always independent.
            "baseline": diagnostics.setpiece_consistency(
                labels["baseline"], events, config.defaults(), blocks, tmap),
        }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
