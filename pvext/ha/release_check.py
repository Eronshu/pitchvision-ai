"""HA3 release integrity: re-check, NOW, everything a scripts/run_ha3.py manifest recorded.

Every M1 check of pvext/v2/release_check.py (git clean and at the recorded commit, configs/v2.json
and models/m1_final.joblib tracked inside the repository and unchanged, V1 config, all recorded
sources, the CSV's hash and official validation at 199,319 rows) is applied unchanged to the shared
part of the manifest. On top of that:

  ha3_release_config  configs/ha3.json is byte-identical to the one used and equals the recorded
                      HA3 release; the run recorded configs/ha3.json, and it names configs/v2.json
  ha3_model           models/ha3_final.joblib has the sha256 of the HA3 release and the manifest
  ha3_release_files   configs/ha3.json and models/ha3_final.joblib are git-tracked files inside
                      the repository (no symlink escape); the model entry is that relative path
  ha3_dead_model      models/ha3_dead_m1_league.joblib is git-tracked inside the repository, the
                      release's relative dead_model entry, and has the release's and manifest's sha256
  ha3_frozen_code     the HA3 prediction sources are byte-identical to the frozen HA3 candidate
                      (FROZEN_HA3 below, from the HA3 freeze) and to the release's record
Every check must pass; there is no warning mode.
"""

import json
import os
import shutil
import tempfile

from pvext.v2 import release_check as m1_check

ROOT = m1_check.ROOT
HA3_CONFIG = "configs/ha3.json"
HA3_MODEL = "models/ha3_final.joblib"
HA3_DEAD_MODEL = "models/ha3_dead_m1_league.joblib"
# The frozen HA3 candidate (v3/ha/frozen_ha3/SOURCES.sha256, FREEZE.json; lockbox run ha3_004).
FROZEN_HA3 = {
    "pvext/ha/features.py": "c078865d3f61b1327ec7bde76bec570319d46a95c5dda928648c650eac32be5c",
    "pvext/ha/decode.py": "400b1e3e18f2b6cbb8e42456d970ba2d20befb767d47a15ce475c47f9fc74331",
    "pvext/ha/experiment.py": "61a0eb9983a46455713ce7627b66e520eb23e0cf1346257bb3b7ae10c411b8d2",
}


def _m1_part(manifest_path, manifest, root, rows):
    """Run the unchanged M1 verifier on the shared part: a temporary run_v2-style manifest next to
    a symlink to the same CSV (the original output directory is never written)."""
    tmp = tempfile.mkdtemp(prefix="ha3_verify_")
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), manifest["output"])
        if os.path.exists(csv_path):
            os.symlink(os.path.realpath(csv_path), os.path.join(tmp, manifest["output"]))
        shared = dict(manifest, mode="v2_m1")
        path = os.path.join(tmp, "manifest.json")
        with open(path, "w") as handle:
            json.dump(shared, handle)
        return m1_check.check(path, root, rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _inside(root, rel):
    real_root = os.path.realpath(root)
    full = os.path.join(root, rel)
    return os.path.realpath(full) == os.path.join(real_root, rel) and os.path.isfile(full)


def check(manifest_path, root=ROOT, rows=None):
    """[(name, ok, detail)] for every check; never raises on a failed check."""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("mode") != "v2_m1_ha3":
        return [("mode", False, "not a run_ha3 manifest: {!r}".format(manifest.get("mode")))]
    out = _m1_part(manifest_path, manifest, root, rows)

    path = os.path.join(root, HA3_CONFIG)
    try:
        with open(path) as handle:
            current = json.load(handle)
        same = (m1_check._sha256(path) == manifest.get("ha3_release_config_sha256")
                and current == manifest["config"]["ha3_release"]
                and manifest.get("ha3_release_config_path") == HA3_CONFIG
                and current.get("m1_release_config") == m1_check.RELEASE_CONFIG)
        out.append(("ha3_release_config", same, path))
    except (OSError, ValueError, KeyError) as error:
        current = None
        out.append(("ha3_release_config", False, "cannot read: {}".format(error)))

    model = os.path.join(root, HA3_MODEL)
    actual = m1_check._sha256(model) if os.path.isfile(model) else None
    out.append(("ha3_model", actual is not None and current is not None
                and actual == current.get("model_sha256") == manifest.get("ha3_model_sha256"),
                "{} {}".format(model, actual)))

    problems = []
    for rel in (HA3_CONFIG, HA3_MODEL):
        if not _inside(root, rel):
            problems.append("{} is missing, a symlink or escapes the repository".format(rel))
        if m1_check._git(root, "ls-files", "--error-unmatch", "--", rel) is None:
            problems.append("{} is not tracked by git".format(rel))
    if current is None or current.get("model") != HA3_MODEL:
        problems.append("HA3 release model entry is not the relative {}".format(HA3_MODEL))
    out.append(("ha3_release_files", not problems, "; ".join(problems) or
                "{} and {} tracked inside the repository".format(HA3_CONFIG, HA3_MODEL)))

    dead_rel = (current or {}).get("dead_model") or {}
    dead = os.path.join(root, HA3_DEAD_MODEL)
    dead_sha = m1_check._sha256(dead) if os.path.isfile(dead) else None
    dead_problems = []
    if not _inside(root, HA3_DEAD_MODEL):
        dead_problems.append("{} is missing, a symlink or escapes the repository".format(HA3_DEAD_MODEL))
    if m1_check._git(root, "ls-files", "--error-unmatch", "--", HA3_DEAD_MODEL) is None:
        dead_problems.append("{} is not tracked by git".format(HA3_DEAD_MODEL))
    if dead_rel.get("model") != HA3_DEAD_MODEL:
        dead_problems.append("dead_model entry is not the relative {}".format(HA3_DEAD_MODEL))
    if dead_sha is None or not (dead_sha == dead_rel.get("model_sha256")
                                == manifest.get("dead_model_sha256")):
        dead_problems.append("dead-ball model sha256 {} differs from the release/manifest".format(dead_sha))
    out.append(("ha3_dead_model", not dead_problems, "; ".join(dead_problems) or
                "{} {}".format(HA3_DEAD_MODEL, dead_sha)))

    changed = []
    for rel, frozen in sorted(FROZEN_HA3.items()):
        full = os.path.join(root, rel)
        now = m1_check._sha256(full) if os.path.isfile(full) else None
        recorded = (current or {}).get("frozen_sources", {}).get(rel)
        if now != frozen or recorded != frozen:
            changed.append(rel)
    out.append(("ha3_frozen_code", not changed,
                "differs from the frozen HA3: {}".format(changed) if changed
                else "HA3 prediction code is the frozen candidate"))
    return out
