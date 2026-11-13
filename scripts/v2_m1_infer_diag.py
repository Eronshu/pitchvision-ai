"""BAR-ATH inference check for M1: diagnostics only, no ground truth (design spec, robustness C).

For the project lead to run locally, never in a Claude session:

    python scripts/v2_m1_infer_diag.py --out /path/outside/repo/m1_barath_diag

It runs frozen V1 and the seven leave-one-match-out M1 models (trained on IDSSE only) on the
competition ATD. It writes one file, diagnostics.json, with aggregates only:
  - per method: label shares inside the graded blocks, D-run count and length quantiles, frame
    agreement with V1 and D-mask Jaccard against V1;
  - across the seven M1 models: the spread of D share and the minimum pairwise D Jaccard;
  - execution checks: length, label set, finite features.
No ground truth or public GT is read. Nothing is scored, selected or tuned. No per-frame
predictions are written.

PROTOCOL: this is an inference and execution sanity check only. Its D share, transitions and
agreement statistics must NOT be used to choose between models, select thresholds, tune
features or calibrate predictions (design spec, "Competition-side check").
"""

import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from pitchvision import atd, config, pipeline  # noqa: E402
from pvext.v2 import features, m1  # noqa: E402

LABELS = ("H", "A", "D", "X")


def _graded(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def _dead_runs_s(pred, blocks, fps):
    lengths = []
    for _, first, last in blocks:
        dead = np.concatenate(([0], (pred[first:last + 1] == "D").astype(np.int8), [0]))
        edges = np.diff(dead)
        lengths += ((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / fps).tolist()
    return lengths


def _jaccard(a, b):
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else 1.0


def describe(pred, v1, blocks, fps):
    graded = _graded(len(pred), blocks)
    p, ref = pred[graded], v1[graded]
    runs = _dead_runs_s(pred, blocks, fps)
    q = np.percentile(runs, [10, 50, 90]).tolist() if runs else [None] * 3
    return {"share": {c: float(np.mean(p == c)) for c in LABELS},
            "d_runs": len(runs), "d_run_s_p10_p50_p90": q,
            "d_runs_per_graded_minute": len(runs) / (graded.sum() / fps / 60.0),
            "agreement_with_v1": float(np.mean(p == ref)),
            "d_jaccard_with_v1": _jaccard(p == "D", ref == "D")}


def diagnose(tr, blocks, pitch, fps, bundles, cfg):
    """Pure core: V1 and every M1 bundle on one Tracking; returns the aggregate dict."""
    n = len(tr.ball_xy)
    names, X = features.compute(tr, pitch, blocks, fps)
    checks = {"n_frames": n, "features_finite": bool(np.isfinite(X).all()),
              "n_features": len(names)}
    v1 = pipeline.run(tr, cfg, fps, pitch, blocks)
    preds = {"v1": v1}
    for name, bundle in sorted(bundles.items()):
        if X.shape[1] != bundle["model"].n_features_in_:
            raise ValueError("{}: feature count {} != model's {}".format(
                name, X.shape[1], bundle["model"].n_features_in_))
        prob = bundle["model"].predict_proba(X)[:, 1]
        dead = m1.persist(prob, blocks, cfg["alive"], fps, bundle["threshold"])
        preds[name] = m1.run_with_dead(tr, cfg, fps, pitch, blocks, dead)
    for name, pred in preds.items():
        if len(pred) != n or not set(np.unique(pred)) <= set(LABELS):
            raise ValueError("{}: invalid prediction".format(name))
    out = {"checks": checks, "methods": {k: describe(p, v1, blocks, fps) for k, p in preds.items()}}
    m1_names = [k for k in preds if k != "v1"]
    if m1_names:
        graded = _graded(n, blocks)
        shares = [out["methods"][k]["share"]["D"] for k in m1_names]
        pairs = [_jaccard(preds[a][graded] == "D", preds[b][graded] == "D")
                 for i, a in enumerate(m1_names) for b in m1_names[i + 1:]]
        out["m1_spread"] = {"d_share_min": min(shares), "d_share_max": max(shares),
                            "min_pairwise_d_jaccard": min(pairs) if pairs else None}
    return out


def expected_models():
    """{filename: (train, test)} for the seven leave-one-match-out models, exactly."""
    return {"lomo_{}.joblib".format(test[0]): (sorted(train), list(test))
            for train, test in m1.folds("lomo")}


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_models(models_dir, manifest_path=None, allow_unverified=False):
    """Load exactly the seven LOMO models, each validated before use.

    - The *.joblib files in `models_dir` that start with "lomo_" must be exactly the expected
      names; a missing or unexpected file raises.
    - Each bundle's train/test membership must equal its fold (via experiment.check_held_out,
      plus an exact match of the training set).
    - Each file's sha256 and threshold must match the preserved experiment MANIFEST.json. Without a
      manifest this raises unless `allow_unverified` is set; the result then records that.
    Returns (bundles keyed "m1_lomo_<match>", provenance dict)."""
    from pvext.v2 import experiment
    expected = expected_models()
    present = {os.path.basename(p) for p in glob.glob(os.path.join(models_dir, "lomo_*.joblib"))}
    missing, unexpected = set(expected) - present, present - set(expected)
    if missing or unexpected:
        raise ValueError("LOMO model set mismatch: missing {}, unexpected {}".format(
            sorted(missing), sorted(unexpected)))
    manifest = None
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle).get("models")
        if not isinstance(manifest, dict):
            raise ValueError("manifest {} has no models section".format(manifest_path))
    elif not allow_unverified:
        raise ValueError("no experiment manifest at {!r}; pass --allow-unverified to run "
                         "without hash verification".format(manifest_path))
    bundles, hashes = {}, {}
    for name, (train, test) in sorted(expected.items()):
        path = os.path.join(models_dir, name)
        digest = _sha256(path)
        if manifest is not None:
            entry = manifest.get(name)
            if entry is None:
                raise ValueError("{} is not in the experiment manifest".format(name))
            if entry.get("sha256") != digest:
                raise ValueError("{}: sha256 differs from the experiment manifest".format(name))
        bundle = joblib.load(path)
        experiment.check_held_out(bundle, test[0], name)
        if sorted(bundle["train"]) != train or list(bundle["test"]) != test:
            raise ValueError("{}: train/test {} / {} do not match the fold".format(
                name, bundle["train"], bundle["test"]))
        if manifest is not None and manifest[name].get("threshold") != bundle["threshold"]:
            raise ValueError("{}: threshold differs from the experiment manifest".format(name))
        bundles["m1_" + name[:-len(".joblib")]] = bundle
        hashes[name] = digest
    return bundles, {"models_sha256": hashes, "manifest": manifest_path,
                     "manifest_verified": manifest is not None}


def main(argv=None):
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    from pvext import idsse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output directory, outside the repository")
    parser.add_argument("--models", default=os.path.join(idsse.DEFAULT_ROOT, "v2", "m1", "models"))
    parser.add_argument("--manifest", help="experiment MANIFEST.json (default: next to --models)")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="run without a manifest (recorded in the output)")
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    out = os.path.realpath(args.out)
    if out == ROOT or out.startswith(ROOT + os.sep):
        raise SystemExit("--out must be outside the repository")
    manifest = args.manifest or os.path.join(os.path.dirname(os.path.abspath(args.models)),
                                             "MANIFEST.json")
    try:
        bundles, provenance = load_models(args.models, manifest, args.allow_unverified)
    except ValueError as error:
        raise SystemExit("model check failed: {}".format(error))
    cfg = config.load(m1.V1_CONFIG)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    tr = atd.load_arrays(meta_path, raw_path, n, pitch)
    result = diagnose(tr, blocks, pitch, fps, bundles, cfg)
    result.update(provenance)
    result["note"] = ("Inference/execution sanity check only: no ground truth read, nothing scored. "
                      "Must not be used to choose models, thresholds or features, or to calibrate.")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "diagnostics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", os.path.join(out, "diagnostics.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
