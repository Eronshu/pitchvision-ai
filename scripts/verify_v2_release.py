"""Re-verify a V2 run NOW before upload: clean tree at the recorded commit, unchanged release
config, V1 config, model and sources, and a CSV that still matches and validates.

    python scripts/verify_v2_release.py "/path/pv_v2/manifest.json"

Exits 0 only if every check passes (pvext/v2/release_check.py). The CSV is checked against the
official 199,319 tactical frames, and the release config and model must be the git-tracked
configs/v2.json and models/m1_final.joblib inside the repository. Complements, and does not
replace, scripts/verify_submission.py --strict.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext.v2 import release_check  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--rows", type=int, help=argparse.SUPPRESS)   # synthetic tests only
    parser.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    results = release_check.check(args.manifest, args.root, args.rows)
    for name, ok, detail in results:
        print("{:4} {:15} {}".format("OK" if ok else "FAIL", name, detail))
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
