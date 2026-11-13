"""Run the external benchmark: frozen V1 vs the official fallback on degraded IDSSE tracking.

    python scripts/run_benchmark.py --match J03WPY --presets clean,mild,doc,severe --seeds 0,1,2

Writes per_run.csv (one row per method, preset and seed) and summary.json (mean/min/max over
seeds) to --out, which must be outside the repository. Converted ATD files live in a temporary
directory under --work and are deleted after each run.

The reports hold aggregates only and carry the CC BY 4.0 attribution. Results from one match are
exploratory diagnostics: not an estimate of competition performance, and not for choosing model
parameters.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pitchvision import config  # noqa: E402
from pvext import benchmark, degrade, idsse  # noqa: E402


def _git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", help="an IDSSE match id; files read from --root")
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--meta")
    parser.add_argument("--positions")
    parser.add_argument("--presets", default="clean,mild,doc,severe")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--work", help="scratch directory for converted ATD files")
    parser.add_argument("--out", help="report directory (outside the repository)")
    args = parser.parse_args(argv)

    if args.match:
        paths = idsse.local_paths(args.match, args.root)
        meta_xml, positions_xml = paths["meta"], paths["positions"]
    else:
        meta_xml, positions_xml = args.meta, args.positions
    label = args.match or "custom"
    out = idsse.check_root(args.out or os.path.join(args.root, "benchmark_" + label))
    work = idsse.check_root(args.work or os.path.join(args.root, "work"))
    presets = args.presets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = benchmark.run(meta_xml, positions_xml, presets, seeds, work)
    os.makedirs(out, exist_ok=True)
    flat = [benchmark.flatten(r) for r in rows]
    with open(os.path.join(out, "per_run.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {
        "match": label, "exploratory": True,
        "note": "One match: exploratory diagnostics only, not an estimate of competition "
                "performance and not a basis for selecting parameters.",
        "primary_metrics": list(benchmark.PRIMARY),
        "v1_config_sha256": config.sha256(config.load(benchmark.V1_CONFIG)),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "presets": {p: degrade.PRESETS[p] for p in presets},
        "seeds": seeds,
        "results": benchmark.summarize(rows),
        "attribution": idsse.ATTRIBUTION,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("wrote {} and {}".format(os.path.join(out, "per_run.csv"),
                                   os.path.join(out, "summary.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
