"""Download IDSSE / DFL match files, verified against the pinned manifest in pvext/idsse.py.

    python scripts/fetch_idsse.py --match J03WPY
    python scripts/fetch_idsse.py --match J03WPY --parts meta,positions --root /path/outside/repo

The files go to --root (default $IDSSE_DIR, else ~/Projects/PitchVisionAI/External/idsse), never
into the repository. A file that is already there and verifies is not downloaded again. Licence:
CC BY 4.0 -- see docs/EXTERNAL_DATA.md for the attribution the data requires.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext import idsse  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog=idsse.ATTRIBUTION + " See docs/EXTERNAL_DATA.md.")
    parser.add_argument("--match", required=True, help="one of: " + ", ".join(sorted(idsse.MATCHES)))
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT, help="download directory")
    parser.add_argument("--parts", default="meta,events,positions",
                        help="comma-separated subset of meta,events,positions")
    args = parser.parse_args(argv)

    if args.match not in idsse.MATCHES:
        print("unknown match {!r}; the published ids are {}".format(
            args.match, ", ".join(sorted(idsse.MATCHES))), file=sys.stderr)
        return 2
    try:
        idsse.check_root(args.root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    print(idsse.ATTRIBUTION + "\n")
    files = idsse.MATCHES[args.match]["files"]
    for part in args.parts.split(","):
        entry = files[part]
        print("{:<9} {} ({:,} bytes) ...".format(part, entry["name"], entry["size"]), flush=True)
        try:
            path = idsse.fetch_one(entry, args.root)
        except (ValueError, OSError) as error:
            print("FAIL {}: {}".format(entry["name"], error), file=sys.stderr)
            return 1
        print("          verified -> {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
