"""Synthetic final-model artifacts for the training, release and submission tests.

Training caches with the real 59-feature schema but random rows (a signal in two columns), a frozen
M1-style manifest for them, and a release config pointing at a model trained by final.train_final.
Nothing here touches IDSSE or competition data.
"""

import hashlib
import json
import os

import numpy as np

from pvext.v2 import final, m1

ROWS = 300


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_caches(src, names=None, matches=m1.MATCHES, seed=0):
    """<src>/train/*.npz for every match x preset x seed, and <src>/MANIFEST.json describing them."""
    names = list(names or final.feature_schema())
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(src, "train"), exist_ok=True)
    hashes = {}
    for match in matches:
        for preset in m1.TRAIN_PRESETS:
            for s in m1.TRAIN_SEEDS:
                y = (np.arange(ROWS) // 30) % 2 == 1
                X = rng.normal(0, 1, (ROWS, len(names))).astype(np.float32)
                X[:, 0] += 2.5 * y
                X[:, 1] -= 1.5 * y
                name = "{}_{}_{}.npz".format(match, preset, s)
                path = os.path.join(src, "train", name)
                np.savez(path, X=X, y=y, frames=np.arange(ROWS) * 5,
                         blocks=json.dumps([(1, 0, ROWS // 2 - 1), (2, ROWS // 2, ROWS - 1)]),
                         names=json.dumps(names))
                hashes[name] = sha(path)
    manifest = {"config": {"train_presets": list(m1.TRAIN_PRESETS),
                           "train_seeds": list(m1.TRAIN_SEEDS), "stride": m1.STRIDE,
                           "thresholds_grid": list(m1.THRESHOLDS)},
                "train_cache_sha256": hashes}
    path = os.path.join(src, "MANIFEST.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle)
    return path


def rewrite_manifest(path, **changes):
    with open(path) as handle:
        manifest = json.load(handle)
    manifest.update(changes)
    with open(path, "w") as handle:
        json.dump(manifest, handle)


def build_release(base):
    """Train a synthetic final model under <base>/final and write <base>/v2.json pointing at it
    (absolute model path). Returns (release_config_path, final_manifest)."""
    src = os.path.join(str(base), "src")
    manifest_path = write_caches(src)
    out = os.path.join(str(base), "final")
    manifest = final.train_final(src, manifest_path, out)
    with open(os.path.join(out, final.RELEASE_CONFIG_FILE)) as handle:
        release = json.load(handle)
    release["model"] = os.path.join(out, final.MODEL_FILE)
    path = os.path.join(str(base), "v2.json")
    with open(path, "w") as handle:
        json.dump(release, handle)
    return path, manifest


def edit_release(src_path, dst_path, **changes):
    with open(src_path) as handle:
        release = json.load(handle)
    release.update(changes)
    with open(dst_path, "w") as handle:
        json.dump(release, handle)
    return str(dst_path)
