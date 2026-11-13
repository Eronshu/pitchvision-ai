"""V2 release integrity: re-check, NOW, everything a run_v2.py manifest recorded.

scripts/verify_submission.py checks what the run recorded (provenance and format). This module
re-derives the current state and compares it with the record:

  git_clean        the working tree at `root` is clean now (git status --porcelain is empty)
  git_head         HEAD now is the commit the manifest recorded, and the run itself was clean
  release_config   the release config file now is byte-identical to the one used (sha256), and
                   its content equals the configuration recorded in the manifest
  v1_config        the V1 config now loads to the configuration recorded in the manifest
  model            the model file now has the sha256 of the release config and of the manifest
  sources          every source file recorded in the manifest has the same sha256 now
  release_files    the release config is <repo>/configs/v2.json and its model entry is the relative
                   path models/m1_final.joblib; both resolve inside the repository with no symlink
                   (no escape), both are tracked by git, and the run recorded configs/v2.json
  output           the CSV next to the manifest has the recorded sha256 and passes the official
                   validation now (possession_metrics.validate_submission) with the OFFICIAL row
                   count (games.source_frames("tactical") = 199,319), not the manifest's count;
                   `rows` overrides it only for synthetic tests
Every check must pass; there is no warning mode. Development loading (pvext/v2/deploy.py) still
accepts absolute model paths; only release verification requires the in-repository layout.
"""

import hashlib
import json
import os
import subprocess
import sys

from pitchvision import config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RELEASE_CONFIG = "configs/v2.json"
RELEASE_MODEL = "models/m1_final.joblib"


def official_rows():
    sys.path.insert(0, ROOT)
    from games import GAME, source_frames
    return source_frames("tactical", GAME)


def release_files(root, manifest, current):
    """(ok, detail) for the in-repository, git-tracked release layout."""
    real_root = os.path.realpath(root)
    problems = []
    for rel in (RELEASE_CONFIG, RELEASE_MODEL):
        if os.path.realpath(os.path.join(root, rel)) != os.path.join(real_root, rel):
            problems.append("{} is a symlink or escapes the repository".format(rel))
        elif not os.path.isfile(os.path.join(root, rel)):
            problems.append("{} is missing".format(rel))
    if manifest.get("release_config_path") != RELEASE_CONFIG:
        problems.append("run used release config {!r}, not {}".format(
            manifest.get("release_config_path"), RELEASE_CONFIG))
    if current is None or current.get("model") != RELEASE_MODEL:
        problems.append("release config model is {!r}, not the relative {}".format(
            None if current is None else current.get("model"), RELEASE_MODEL))
    for rel in (RELEASE_CONFIG, RELEASE_MODEL):
        if _git(root, "ls-files", "--error-unmatch", "--", rel) is None:
            problems.append("{} is not tracked by git".format(rel))
    return not problems, "; ".join(problems) or "{} and {} tracked inside the repository".format(
        RELEASE_CONFIG, RELEASE_MODEL)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *args):
    try:
        return subprocess.run(["git"] + list(args), cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def check(manifest_path, root=ROOT, rows=None):
    """[(name, ok, detail)] for every check; never raises on a failed check."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    out = []
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("mode") != "v2_m1":
        return [("mode", False, "not a run_v2 manifest: {!r}".format(manifest.get("mode")))]

    status = _git(root, "status", "--porcelain")
    out.append(("git_clean", status == "", "working tree {}".format(
        "clean" if status == "" else "dirty or unknown")))
    head = _git(root, "rev-parse", "HEAD")
    out.append(("git_head", head is not None and head == manifest.get("git_commit")
                and manifest.get("git_dirty") is False,
                "HEAD {}, run at {} (dirty={})".format(head, manifest.get("git_commit"),
                                                       manifest.get("git_dirty"))))

    rel_path = os.path.join(root, RELEASE_CONFIG)
    try:
        with open(rel_path) as handle:
            current = json.load(handle)
        same = (_sha256(rel_path) == manifest.get("release_config_sha256")
                and current == manifest["config"]["release"])
        out.append(("release_config", same, rel_path))
    except (OSError, ValueError, KeyError) as error:
        current = None
        out.append(("release_config", False, "cannot read: {}".format(error)))

    try:
        v1_now = config.load(_resolve(root, manifest["config"]["release"]["v1_config"]))
        out.append(("v1_config", v1_now == manifest["config"]["v1"]
                    and config.sha256(manifest["config"]) == manifest.get("config_sha256"),
                    "V1 config and recorded config hash"))
    except (OSError, ValueError, KeyError) as error:
        out.append(("v1_config", False, "cannot check: {}".format(error)))

    if current is not None and "model" in current:
        model_path = _resolve(root, current["model"])
        actual = _sha256(model_path) if os.path.isfile(model_path) else None
        out.append(("model", actual is not None and actual == current.get("model_sha256")
                    == manifest.get("model_sha256"), "{} {}".format(model_path, actual)))
    else:
        out.append(("model", False, "no current release config"))

    out.append(("release_files",) + release_files(root, manifest, current))

    changed = []
    for path, recorded in sorted(manifest.get("source_sha256", {}).items()):
        full = _resolve(root, path)
        if not os.path.isfile(full) or _sha256(full) != recorded:
            changed.append(path)
    out.append(("sources", bool(manifest.get("source_sha256")) and not changed,
                "changed: {}".format(changed) if changed else "{} files match".format(
                    len(manifest.get("source_sha256", {})))))

    csv_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), manifest["output"])
    ok = os.path.isfile(csv_path) and _sha256(csv_path) == manifest.get("output_sha256")
    n_rows = rows if rows is not None else official_rows()
    reason = pm.validate_submission(pm.read_label_csv(csv_path), n_rows) \
        if os.path.isfile(csv_path) else "missing"
    out.append(("output", ok and reason is None and manifest.get("valid") is True
                and manifest.get("n_rows") == n_rows,
                "sha256 {}, format {} at {} rows (manifest {})".format(
                    "matches" if ok else "differs", "valid" if reason is None else reason, n_rows,
                    manifest.get("n_rows"))))
    return out
