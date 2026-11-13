"""Train the final M1 deployment model on all seven IDSSE matches (pvext/v2/final.py).

    python scripts/v2_train_final.py --out $IDSSE_DIR/v2/m1_final

Reads the frozen M1 training caches in $IDSSE_DIR/v2/m1/train, verified against
$IDSSE_DIR/v2/m1/MANIFEST.json, and never writes there. Writes into --out, which must be new or
empty and outside the repository:
  m1_final.joblib       the model bundle (the only file ever loaded by the submission generator)
  MANIFEST.json         threshold and curve, model sha256, training matches/rows, cache and code
                        hashes, dependencies, git commit, training-set sanity scores, attribution
  release_config.json   the proposed configs/v2.json (model sha256 kept outside the artifact)
Copying the model into models/ and the release config into configs/v2.json is a separate,
approved step.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pvext import idsse  # noqa: E402
from pvext.v2 import final  # noqa: E402


def inside(path, directory):
    """True if `path` is `directory` or anywhere below it (symlinks resolved)."""
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return os.path.commonpath([path, directory]) == directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=idsse.DEFAULT_ROOT)
    parser.add_argument("--out", required=True, help="new, empty directory outside the repository")
    args = parser.parse_args(argv)
    src = idsse.check_root(os.path.join(args.root, "v2", "m1"))
    out = os.path.abspath(args.out)
    for frozen in ("m1", "m1_robust"):
        if inside(out, os.path.join(args.root, "v2", frozen)):
            raise SystemExit("refusing to write into or under the frozen {} directory".format(frozen))
    try:
        manifest = final.train_final(src, os.path.join(src, "MANIFEST.json"), out)
    except (ValueError, FileExistsError) as error:
        raise SystemExit("final training refused: {}".format(error))
    print("threshold", manifest["threshold"], "model_sha256", manifest["model_sha256"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
