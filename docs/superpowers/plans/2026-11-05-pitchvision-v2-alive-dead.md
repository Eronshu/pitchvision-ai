# PitchVision V2 Alive/Dead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A learned dead-ball decision (M1 logistic regression, then M2 gradient boosting) that replaces only `alive.dead_mask`. It is evaluated under whole-match-held-out folds against frozen V1 and the fallback on IDSSE.

**Architecture:** `pvext/v2/` holds ATD-only, slot-invariant features computed from a `pitchvision.atd.Tracking`; a dataset builder (degraded copies with training seeds, features and targets cached outside the repository); fold-aware training; and an evaluation harness that reuses `pvext.benchmark.score`. The possession state machine and everything in `pitchvision/` stay frozen.

**Tech Stack:** Python 3.13, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1 (new, pinned), pytest 9.2.0.

**Spec:** `docs/superpowers/specs/2026-11-05-pitchvision-v2-alive-dead-design.md` (with the P3 protocol in `docs/superpowers/specs/2026-11-04-pitchvision-v2-external-benchmark-design.md`)

## Global Constraints

- `pitchvision/`, `configs/v1.json`, the official files, and the presets and frozen evaluation seeds 0, 1, 2 are unchanged.
- Inputs are only what `atd.load_arrays` reads from an ATD file. There is never `z`, DFL `S`/`D`/`A`/`M`, events or labels, and never the ATD ball-speed field.
- Training seeds are 100–104. Evaluation uses only the frozen presets and seeds 0, 1, 2 on held-out matches.
- No BAR-ATH data and no public GT in any fit, selection or threshold.
- Caches, models and reports live outside the repository (`$IDSSE_DIR/v2/`).
- Every commit needs approval; never push without approval. Tests: `.venv/bin/python -m pytest -W error`.

## Review Focus

- Slot permutation: shuffling slot order within a team must leave every feature unchanged (Task 2).
- An all-missing ball over a long stretch: features are defined, not NaN-propagating into the model (Task 2).
- Fold leakage: a match appears in both the training and the test rows of a fold (Task 3).
- A held-out match with no stoppage starts inside a block (start F1 undefined) is reported as undefined, not 0 or 1 (Task 4).
- Integration: with the learned model switched off, the pipeline output is byte-identical to frozen V1 (Task 6).

---

### Task 1: Fix the two recorded known issues

**Files:** Modify `pvext/degrade.py` and `pvext/convert.py`. Test: `tests/ext/test_degrade.py`, `tests/ext/test_convert.py`.
- [ ] Failing tests:
  - a preset with `crowd_radius_m=6.0` marks more frames as crowded than one with 3.0;
  - the four frozen presets give **byte-identical** `apply()` output before and after the fix (hashes recorded first);
  - the written ball group's third field is the position-derived speed in m/s (or `NaN` when unknown);
  - V1's and the fallback's predictions on a fixture are unchanged.
- [ ] Implement; run the full suite; commit after approval.

### Task 2: ATD-only, slot-invariant features

**Files:** Create `pvext/v2/__init__.py`, `pvext/v2/features.py`, `tests/ext/v2/test_features.py`.

**Interfaces:** `features.compute(tr: atd.Tracking, pitch, fps=25, stride=5) -> (names: list[str], X: float32 (T, F))`, with the groups of the design spec (ball, field motion, shape, ±1/3/10 s windows, frozen-V1 signals via `pitchvision.features`/`alive.dead_score`).
- [ ] Failing tests:
  - permuting slots within each team gives an identical `X`;
  - the signature has no label parameter;
  - **deceleration onset** is positive just before a scripted field-wide slowdown;
  - a kick-off formation (both teams in their own halves) sets that feature;
  - a long missing-ball stretch gives finite features;
  - no feature name references `z`, `ball_speed_field`, `S`, `BallStatus` or `possession`;
  - the result is deterministic.
- [ ] Implement with numpy only; run; commit after approval.

### Task 3: Dataset builder and folds

**Files:** Create `pvext/v2/dataset.py`, `scripts/v2_build_dataset.py`, `tests/ext/v2/test_dataset.py`.

**Interfaces:** `dataset.build(match_id, presets, seeds, root) -> path` writes `features.npy`, `target.npy` and `meta.json` (match, preset, seed, blocks, feature names) outside the repo. The path goes through `degrade.apply`, then `convert.write_atd`, then `atd.load_arrays`, then `features.compute`. `dataset.folds(kind="lomo"|"league") -> list[(train_ids, test_id(s))]`.
- [ ] Failing tests:
  - features are byte-identical when the targets are permuted;
  - no fold has overlapping training and test match ids;
  - the training seed set is disjoint from {0, 1, 2};
  - X frames inside graded blocks only;
  - the league folds put {J03WMX, J03WN1} against the five Düsseldorf matches.
- [ ] Implement; build the training copies for all 7 matches (local compute only, outside the repo); commit the code after approval.

### Task 4: Evaluation harness (shared by every model)

**Files:** Create `pvext/v2/evaluate.py`, `scripts/v2_evaluate.py`, `tests/ext/v2/test_evaluate.py`.

**Interfaces:** `evaluate.run(model_factory, folds, presets=("mild","doc","severe"), seeds=(0,1,2))`. For each held-out match it rebuilds the frozen evaluation ATD, predicts dead frames, injects them as the dead mask into the frozen possession machine, and scores the candidate, V1 and the fallback with `pvext.benchmark.score`. Output: per-match and per-league tables, and a `go_criteria` dict evaluated mechanically from the design spec.
- [ ] Failing tests:
  - with an oracle model (dead = target) the D F1 is 1;
  - with `model_factory` returning V1's mask, the candidate's scores equal V1's exactly;
  - an undefined start F1 is reported as `None`;
  - the go criteria evaluate on a hand-built table.
- [ ] Implement; commit after approval.

### Task 5: M1 logistic regression, then M2 gradient boosting

**Files:** Create `pvext/v2/models.py`, `tests/ext/v2/test_models.py`. Modify `requirements-pv.txt` to add `scikit-learn==1.9.1`.
- [ ] Failing tests:
  - deterministic fit and predict for a fixed seed;
  - the model is fit only on the training-fold rows (checked through row ids);
  - V1's persistence post-processing is applied.
- [ ] Implement M1 and evaluate under leave-one-match-out and the league folds. **Report and stop for review.**
- [ ] Only after approval: implement M2, evaluate it the same way, report and stop.

### Task 6: Optional pipeline integration (only after a GO)

**Files:** Create `pvext/v2/pipeline.py`, `scripts/run_v2.py` and tests.
- [ ] Failing tests:
  - with the model off, the output is byte-identical to frozen V1;
  - the model file's hash is recorded in the manifest;
  - `verify_submission.py --strict` passes on a synthetic run.
- [ ] The project lead runs it on BAR-ATH locally for L1 diagnostics only.

**Decision points:** after Task 5/M1, after M2, and (if proposed) before M3. Time box: no GO by 2026-11-12 means V1 stays the submission candidate.
