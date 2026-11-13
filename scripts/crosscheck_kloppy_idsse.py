"""Cross-check pvext.dfl_parse against kloppy (only_alive=False) on one IDSSE match.

    python scripts/crosscheck_kloppy_idsse.py --match J03WPY --out /outside/repo/J03WPY_kloppy.json
    python scripts/crosscheck_kloppy_idsse.py --meta META.xml --positions POS.xml --out OUT.json

Exit 0 when both parsers agree on every frame and player observation, 1 otherwise. It needs
kloppy (requirements-starter.txt) and about 4 GB of memory for a full match. The report holds
aggregates only. Data: see docs/EXTERNAL_DATA.md (CC BY 4.0).
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext import crosscheck, idsse  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", help="an IDSSE match id; files read from --root")
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--meta")
    parser.add_argument("--positions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.match:
        paths = idsse.local_paths(args.match, args.root)
        meta_xml, positions_xml = paths["meta"], paths["positions"]
    else:
        meta_xml, positions_xml = args.meta, args.positions
    result = crosscheck.compare(meta_xml, positions_xml)
    result["attribution"] = idsse.ATTRIBUTION
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("{} -> {}".format("AGREE" if result["agrees"] else "DISAGREE", args.out))
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
