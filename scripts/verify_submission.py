"""Check a run directory before its CSV is uploaded: the file is the one the manifest describes.

    python scripts/verify_submission.py /tmp/pv_v1/manifest.json
    python scripts/verify_submission.py /tmp/pv_v1/manifest.json --strict

Checks, each printed as OK / FAIL / WARN:
  output_sha256   the CSV next to the manifest is byte for byte the one the run wrote
  valid           the run itself passed the official validation
  format          the CSV passes possession_metrics.validate_submission now, with --rows rows
  config_sha256   the recorded config hashes to the recorded hash (nobody edited it after)
  git_dirty       the run was made from a clean working tree
  git_commit      ...at the commit checked out now

The two git checks are warnings by default and failures with --strict, which is the mode for
the final upload: the public repository at that commit must regenerate this exact file.
This checks provenance and format. It says nothing about whether the labels are right.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, source_frames  # noqa: E402
from pitchvision import config  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checks(manifest_path, rows):
    """[(name, passed, is_provenance, detail)]"""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), manifest["output"])
    out = []

    exists = os.path.isfile(csv_path)
    actual = _sha256(csv_path) if exists else None
    out.append(("output_sha256", actual == manifest["output_sha256"], False,
                "{} {}".format(manifest["output"], "matches" if actual == manifest["output_sha256"]
                               else "differs from the manifest" if exists else "is missing")))
    out.append(("valid", manifest.get("valid") is True, False,
                "the run recorded valid={}".format(manifest.get("valid"))))
    if exists:
        reason = pm.validate_submission(pm.read_label_csv(csv_path), rows)
        out.append(("format", reason is None, False,
                    "{} rows expected: {}".format(rows, reason or "passes")))
    recorded = manifest["config_sha256"]
    out.append(("config_sha256", config.sha256(manifest["config"]) == recorded, False,
                "recorded config {} its hash".format(
                    "matches" if config.sha256(manifest["config"]) == recorded else "does not match")))
    out.append(("git_dirty", manifest.get("git_dirty") is False, True,
                "run made from a {} tree".format("clean" if manifest.get("git_dirty") is False
                                                 else "dirty or unknown")))
    head = _head()
    out.append(("git_commit", head is not None and manifest.get("git_commit") == head, True,
                "run at {}, checked out {}".format(manifest.get("git_commit"), head)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", help="manifest.json written by scripts/run_v1.py")
    parser.add_argument("--rows", type=int, default=source_frames("tactical", GAME),
                        help="expected row count (default: the tactical video length)")
    parser.add_argument("--strict", action="store_true",
                        help="treat provenance (clean tree, current commit) as required")
    args = parser.parse_args(argv)

    failed = False
    for name, passed, provenance, detail in checks(args.manifest, args.rows):
        if passed:
            status = "OK  "
        elif provenance and not args.strict:
            status = "WARN"
        else:
            status, failed = "FAIL", True
        print("{} {:<14} {}".format(status, name, detail))
    print("\n{}".format("FAILED" if failed else "OK -- provenance and format check out "
                        "(this says nothing about accuracy)"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
