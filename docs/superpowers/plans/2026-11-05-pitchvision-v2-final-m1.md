# PitchVision V2 — final M1 model and competition release: implementation plan

**Goal:** One final M1 model trained on all seven IDSSE matches, and a submission generator that runs
it inside the frozen V1 pipeline. The CSV must be reproducible byte for byte from a clean clone with
the competition data and the packaged model. Frozen V1 remains the fallback.

**Decisions (project lead):**

| | decision |
|---|---|
| D1 | One model on all 7 matches. Threshold from the pre-declared inner LOMO over all 7, used as returned. |
| D2 | Package the joblib artifact in the repository. Its sha256 lives in the external manifest and in `configs/v2.json`. |
| D3 | Option A: scikit-learn and joblib at runtime, pinned, with no inference rewrite. `pvext/v2/m1.py` is not modified. |
| D4 | M1 is the provisional primary candidate and V1 the fallback. Final approval depends on the engineering and reproducibility checks. |
| D5 | The release branch and public commit are decided after verification. PRs stay Draft. |
| D6 | The file is named after the team: `PitchVision AI.csv` (SUBMISSION.md). |
| D7 | Byte-identical CSV on macOS and Linux clean environments. |

**Deployment option.** Option A was chosen over B (NumPy/JSON re-implementation):
- **Code.** A reuses the tested model, `m1.persist` and `m1.run_with_dead`, so no new inference code
  can drift from the validated path. B would add an exporter, a re-implementation and equivalence
  tests, with no rule requiring it.
- **Reproducibility.** Both depend on floating-point behaviour across platforms. A pins exact
  versions and records the decision margin, and the D7 cross-platform byte comparison is the real
  test in either case.

## Tasks

- [x] T1 `pvext/v2/final.py` and `scripts/v2_train_final.py`:
  - verify the 42 frozen caches (set, sha256, feature names, order and width, M1 training
    configuration);
  - select the threshold with `m1.select_threshold` over all 7 matches, then fit;
  - save `m1_final.joblib`, `MANIFEST.json` and `release_config.json` into a new external
    directory; never overwrite; refuse the frozen and in-repository directories.
- [x] T2 `pvext/v2/deploy.py` and `scripts/run_v2.py`:
  - pre-data validation (config, V1 hash, persistence, FPS, dependency versions, sha256 before
    unpickling, bundle type and metadata);
  - post-feature validation (schema, shape, finiteness, labels);
  - run_v1-compatible manifest, official validation, team file name; `--out` outside the
    repository.
- [x] T3 Synthetic tests:
  - `tests/ext/v2/test_final.py`, `test_deploy.py`, `test_run_v2.py`;
  - fixtures in `tests/ext/v2/release_fixture.py`.
- [x] T4 Draft technical document `docs/TECHNICAL_DOCUMENT.md`.
- [x] T5a Provisional final training: run once with commit 7028d81's code (threshold 0.40, model
  sha256 `4b2b59d0…`). A later output-directory guard change to `scripts/v2_train_final.py` is
  recorded in `PROVENANCE_ADDENDUM_1.json`. No retraining was done.
- [ ] T5 (original wording) run the final training on the real IDSSE caches into
  `$IDSSE_DIR/v2/m1_final/`, then review the manifest (threshold, curve, sanity scores).
- [ ] T6 **Needs approval:**
  - copy `m1_final.joblib` to `models/` and `release_config.json` to `configs/v2.json`;
  - commit (the model artifact plus config).
- [ ] T7 **Needs approval:** the project lead runs `run_v2.py` on BAR-ATH locally, then the official
  validator and `verify_submission.py --strict` at the committed, clean tree.
- [ ] T8 **Needs approval:** clean-clone reproduction on macOS and Linux:
  - `cmp` and sha256 of the CSV against the submitted file;
  - measured runtime and peak memory.
- [ ] T9 Finalise the technical document (TBD items), choose the public branch and commit, and
  publish the repository.

## Release checklist

1. Full test suite green under `-W error`.
2. `pitchvision/`, `configs/v1.json` and the official files unchanged against `a8ee2b7`.
3. Frozen M1 and robustness artifacts unchanged, and the 189 frozen hashes reproduced.
4. `models/m1_final.joblib` sha256 equals `configs/v2.json` and the external manifest.
5. The run is made from a clean tree at the release commit, and `verify_submission.py --strict`
   passes.
6. The official `validate_submission.py` passes on the exact file to upload.
7. macOS and Linux clean clones produce byte-identical CSVs (`cmp`, sha256).
8. No competition data, IDSSE raw data, caches or outputs are in the repository.
9. V1 fallback regenerated and verified.
