"""Re-verify an HA3 run NOW before upload: every M1 release check plus the HA3 release config,
the HA3 model and the frozen HA3 prediction code.

    python scripts/verify_ha3_release.py "/path/pv_ha3/manifest.json"

Exits 0 only if every check passes (pvext/ha/release_check.py). Complements, and does not replace,
scripts/verify_submission.py --strict.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext.ha import release_check  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--rows", type=int, help=argparse.SUPPRESS)   # synthetic tests only
    parser.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    results = release_check.check(args.manifest, args.root, args.rows)
    for name, ok, detail in results:
        print("{:4} {:18} {}".format("OK" if ok else "FAIL", name, detail))
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
