# PitchVision V2 — focused Alive/Dead model: design

Date: 2026-11-05 · Branch: `pv/v2` · Status: **finalised for M1 (approved 2026-11-05). M2 and M3 not authorised.**

## Why, and what is (and is not) established

Frozen-V1 benchmark on IDSSE, on DFL labels (these figures do not estimate competition
performance).

**Confirmed across J03WPY, J03WMX and J03WN1 (both leagues):**
- **D recall is low.** It is 0.30–0.72 depending on the preset, and the predicted dead share is far
  below the true share. D precision is high (≥ 0.90).
- **Stoppage starts are weak.** F1 is 0.12–0.32 at ±1 s, against stoppage ends at 0.84–0.89 on
  clean data.
- **mild scores below doc** on D recall in every match, with non-overlapping seed ranges.
- **V1 beats the official fallback** on D F1 and alive/dead F1 in every match × preset cell.

**Not established**: *why* mild underperforms doc. One hypothesis is that tracking noise
suppresses the static-ball cue while doc's missing-ball cue compensates. It is untested, and it is
not used as a design premise.

**Goal**: replace only V1's dead-ball decision (`alive.dead_mask`) with a learned one that raises D
recall and improves stoppage-start detection without losing much precision. The possession state
machine, the restart handling downstream of the dead mask and the submission format stay as in
V1. V1 and the fallback remain available unchanged.

## Inputs: only what the competition ATD provides

The model must run on the competition's ATD exactly as it runs on converted IDSSE. Every feature is
computed from a `pitchvision.atd.Tracking`, i.e. from what `atd.load_arrays` reads, and nothing else.

| source | in competition ATD (DATA.md) | used |
|---|---|---|
| player slot `x`, `y` (metres after scaling) | yes | yes |
| player slot speed (m/s) | yes | yes, but only through team-level aggregates (its smoothing may differ between Metrica's tracker and DFL; position-derived speeds are computed as well) |
| slot → team membership | yes | yes |
| slot identity over time | **no** (tracklets) | no: every feature is invariant to slot order |
| ball `x`, `y`, or missing | yes | yes |
| ball speed | not documented (the third raw ball field is undefined) | **only derived from positions**. `convert.write_atd`'s constant-0 field is a known issue and must never be read |
| ball height `z` | no | **never** (it is in IDSSE, but the degrader is its only user) |
| DFL `S`, `D`, `A`, `M`, events, BallStatus, BallPossession | no | **never** as inputs; BallStatus is the target only |
| Smart Tagging | yes (competition), no (IDSSE) | no: it cannot be trained or validated on IDSSE |
| homography, video | yes | out of scope |

## Features (per frame, computed at a 5-frame stride, then broadcast)

All features are aggregates over slots, so slot order and identity cannot matter.

- **Ball**: present; length of the current missing-ball gap; position-derived speed and
  acceleration (smoothed); a static-run flag; out of bounds, and `exit_plausible`; distance to the
  nearest touchline, to the goal line, to a corner arc and to the centre spot; inside a goal area.
- **Field motion**: median and 90th-percentile player speed (position-derived, and from the speed
  field); share of players above 2 m/s; **deceleration onset** (the median speed in the past 1 s
  and 3 s against the next 1 s and 3 s, which is the whistle signature behind stoppage starts).
- **Shape**: each team's centroid and spread; players per team in each penalty area; players within
  10 m of the ball (wall or set-piece cluster); nearest player's distance to the ball; whether both
  teams are in their own halves (kick-off); players near the touchline next to the ball (throw-in).
- **Temporal context**: means and maxima of the above over centred windows of ±1, ±3 and ±10 s.
  Labelling is offline, so future frames are allowed.
- **V1 signals**: V1's soft `dead_score` components and V1's own dead mask (frozen V1 is a fixed
  feature extractor here, never retrained).

## Model ladder: simplest first, each step gated

| step | model | decoding |
|---|---|---|
| M0 | frozen V1 dead mask (reference), plus the fallback | — |
| **M1** | **logistic regression** (standardised features; L2; fixed seed) | threshold chosen on the training matches (see the M1 specification), then V1's own persistence post-processing (minimum DEAD/ALIVE durations) |
| M2 | `HistGradientBoostingClassifier` (scikit-learn, pinned, fixed `random_state`) | same as M1 |
| M3 | M2 + **two-state HSMM / Viterbi** with duration distributions estimated from the training folds | Viterbi |

M2 runs only if M1 is in place and evaluated. **M3 needs its own approval**, and only if M2 passes
the go criteria and its remaining errors are segment-level (fragmentation or boundary jitter) rather
than missed stoppages. Each step's output plugs into the V1 pipeline in place of `alive.dead_mask`.

## M1 specification (fixed before training)

- **Model**: `StandardScaler` → `LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
  max_iter=2000, random_state=0)`. There is no hyperparameter search; the only fitted choice is the
  decision threshold.
- **Training rows**: every 5th frame inside graded blocks, from the training matches only.
- **Persistence (V1-style, V1 values from `configs/v1.json`)**: the probability is averaged over a
  centred `alive.smooth_s` window (1 s) and thresholded. DEAD runs shorter than `min_dead_s`
  (2 s) are dropped, and ALIVE runs shorter than `min_alive_s` (1 s) between two DEAD runs are
  absorbed. This is applied per graded block.
- **Threshold (training-only)**: chosen from the grid 0.30, 0.35, …, 0.70 to maximise the mean
  per-match D F1 of inner leave-one-match-out predictions over the fold's training matches, after
  persistence, at the 5-frame stride. Ties go to the threshold closest to 0.5. The held-out match is
  never seen.
- **Into the pipeline**: the M1 dead mask replaces `alive.dead_mask` per block, and the frozen
  `possession.label_block` does the rest. V1's own code is called, never edited.

## Data, folds and leakage prevention

- **Matches**: 7 IDSSE matches; the 2 Bundesliga matches share no team or venue with the 5
  Düsseldorf home matches.
- **Training copies**: the presets `mild`, `doc` and `severe` with **training seeds 100 and 101**
  (6 copies per match; fixed in advance for compute reasons), disjoint from the frozen evaluation
  seeds 0, 1 and 2. Evaluation always uses the frozen presets and
  seeds, on the held-out match only.
- **Whole-match held out**:
  - primary: leave-one-match-out (7 folds);
  - secondary: leave-competition-out (train on the 5 2. Bundesliga matches, test on the 2
    Bundesliga matches, and the reverse).
  
  The feature scaler, the model, the threshold, and the HSMM durations (M3) are all fit inside the
  training matches of a fold. Hyperparameters, if any, are chosen by inner leave-one-match-out within
  those training matches.
- **Leakage guards (tested)**:
  1. Features are built from the `Tracking` that `atd.load_arrays` returns for a `write_atd` file, so
     they cannot see labels.
  2. The feature matrix is byte-identical when the targets are permuted.
  3. No training row comes from the held-out match, under any preset or seed.
  4. Evaluation seeds never appear in training.
  5. No BAR-ATH data and no public GT are used for fitting, selection or thresholds.

## Evaluation and pre-registered go criteria

For each held-out match, reported per match and grouped by league, never pooled across matches:
D precision, recall and F1; stoppage-start and stoppage-end F1 at ±1 s and ±2 s; alive/dead F1;
predicted against true D share; and possession spellF1 (secondary). All are computed for the
candidate, frozen V1 and the fallback on the identical ATD files. `doc` is the primary preset;
`mild` and `severe` are reported next to it.

**GO** for a candidate over V1 requires **all** of the following, on `doc` (mean over seeds 0–2 per match),
under leave-one-match-out. These thresholds are fixed before any training and are not revised afterwards:
1. D F1 is higher in at least 6 of 7 matches, including both Bundesliga matches;
2. stoppage-start F1 at ±1 s is higher in at least 5 of 7 matches;
3. D precision is at most 0.05 lower than V1's in the median match;
4. spellF1 is not lower by more than 0.02 in more than 2 of 7 matches;
5. leave-competition-out shows the test matches' mean D F1 above V1's in both directions;
6. `mild` and `severe`: D F1 above V1 in at least 4 of 7 matches for each.

Stoppage-end F1 and alive/dead F1 are reported and monitored, without thresholds.
Possible outcomes: **GO** (all six), **partial** (1 and 3 pass, others fail: candidate for the
constrained hybrid), **NO-GO** (otherwise: retain V1).

## Competition-side check (no ground truth)

After a GO, the project lead runs the candidate on BAR-ATH locally. This is an **inference and
execution sanity check only**. It shows that the pipeline runs end to end on competition-format
ATD and produces a valid, finite output of the right length and label set. The aggregate outputs
are reported for information:
- D share;
- D-run counts and lengths;
- predicted transitions;
- agreement with V1;
- spread across the seven models.

**They must not be used** to choose between models or variants, select thresholds, tune or
select features, or calibrate predictions. Model and threshold choices come only from the IDSSE
protocol above. The numbers stay local. The public 5-minute GT is only a pipeline-correctness
check and is never used for selection.

## Fallback plan

- **No GO for M1 or M2**: keep frozen V1's dead-ball logic. The submission decision stays between V1
  and the official fallback, as before.
- **Partial gain** (more stoppage starts found, but precision or spellF1 lost): evaluate a constrained
  hybrid, *V1 + backward start extension*. The learned model may only move a V1 stoppage's start
  earlier (it never adds or removes stoppages), and the same go criteria apply.
- **Time box**: if no candidate passes by **2026-11-12**, V2 stops and V1 is frozen for the submission
  (deadline 2026-11-17), leaving time for the reproducibility checks and the technical document.

## Known issues addressed first

1. `degrade._crowd()` must honour the preset's `crowd_radius_m`. Every current preset must stay
   byte-identical.
2. `convert.write_atd()`'s ball-speed field is **left as it is**, so the frozen benchmark's input
   encoding and hashes stay unchanged. It cannot reach a feature: `atd.load_arrays` does not
   keep it, and a test shows that the features are identical whatever that field holds. Ball speed
   is always derived from positions.

## As run: M1 experiment (commit 0cdf4ed) and reconciliation with this spec

- **Fallback.** The M1 run scored V1, M1 (leave-one-match-out) and M1 (league folds) only. The
  official fallback was not re-scored there. Its scores on the identical frozen evaluation inputs
  are in the Phase A benchmark (`benchmark_<match>/per_run.csv`): the M1 run's V1 `input_sha256` and
  `prediction_sha256` match it for all 63 V1 runs. The robustness study below scores the fallback
  directly.
- **Boundary tolerances.** `per_run.csv` holds stoppage start and end precision, recall and F1 at
  ±0.5 s, ±1 s and ±2 s (12, 25 and 50 frames). The M1 summary tables reported only ±1 s and
  start ±2 s. The robustness report includes ±1 s and ±2 s for both starts and ends.
- **Start-error records.** The missed-start analysis counts 2,130 records on `doc`. These are
  **710 distinct true stoppage starts**, each counted once per evaluation seed (0, 1, 2). Its
  matched / late / early / missed categories come from overlap between true and predicted DEAD
  runs, a diagnostic. They are not the one-to-one boundary matching behind the start/end F1.
- **Evaluation grid.** `go_criteria` now validates the grid first. It requires exactly 7 matches ×
  {v1, m1_lomo, m1_league} × 3 presets × 3 seeds, no duplicates, and finite required metrics, and
  raises otherwise. The held-out membership check is an explicit `ValueError`, not an `assert`.

## Robustness study (fixed 2026-11-05, before any robustness model is fit or evaluated)

Purpose: test whether M1's gain over V1 depends on cues that may be specific to DFL data or to our
degradation model. Outputs go to `$IDSSE_DIR/v2/m1_robust/`. The frozen M1 artifacts in
`$IDSSE_DIR/v2/m1/` are only read.

**A. Feature-group ablations.** Separate models are refit without the group, not zeroed at
inference. They use the same training caches (presets mild/doc/severe, seeds 100 and 101), the same
leave-one-match-out and league folds, the threshold chosen by inner leave-one-match-out on training
matches only, and the frozen evaluation presets with seeds 0, 1, 2.

| variant | removed features |
|---|---|
| `no_ball_presence` | `ball_present`, `ball_gap_s`, `ball_present_mean_{1,3,10}s` |
| `no_slot_speed` | `players_speed_slot_p50`, `players_speed_slot_p90` |
| `no_both` | both groups |

Known limits, stated in advance:
- Other ball features are 0 when the ball is missing, so presence can still be partly inferred.
- The frozen-V1 signals (`v1_dead_score`, `v1_dead_mask`) use the ATD speed field, just as V1
  itself does.

Each variant gets the six GO criteria mechanically, for reference.

**B. Alternative label-agnostic degradations.** Each is built only from tracking (no BallStatus or
BallPossession) and starts from `doc`, changing one aspect. Evaluation seeds are 0, 1, 2 on
held-out matches only. Frozen V1, the official fallback and every M1 model above are compared on
identical inputs.

| distribution | change from `doc` | question |
|---|---|---|
| `alt_uniform_dropout` | `hard_weight` 1 (ball loss independent of play context) | does M1 depend on our context-weighted ball loss? |
| `alt_long_gaps` | `ball_gap_s` 6 (same share, fewer longer gaps) | gap-length shape |
| `alt_speed_from_positions` | slot speed field replaced by speed from the degraded slot positions (central difference over ±2 frames) | a tracker-computed speed field instead of DFL's |
| `alt_noisy_ball` | `false_ball_rate` 0.02, `noise_ball_m` 1.0 | noisier ball detections |
| `alt_sparse_players` | `player_missing` 0.25, `player_gap_s` 4, `tracklet_mean_s` 8 | sparser, shorter player tracklets |

Pre-declared reading for each distribution, per M1 variant (descriptive, not a new GO):
- **robust** if D F1 is above V1 in at least 6 of 7 matches and the median D precision change is
  at least −0.05;
- **fragile** otherwise.

These results are reported separately from the frozen benchmark.

**C. BAR-ATH inference check.** The project lead runs it locally; it is never run in a Claude
session. A script runs frozen V1 and the seven leave-one-match-out M1 models on the competition ATD
and writes only aggregate diagnostics:
- D share, D-run count and length quantiles;
- agreement with V1;
- spread across the seven models;
- execution checks (length, label set, no NaN).

It uses no ground truth, scores nothing and selects nothing. As in "Competition-side check", none
of its outputs may be used to choose models, thresholds or features, or to calibrate predictions.

## Final deployment model and release (decisions D1–D7, 2026-11-05)

- **One final M1 model** is trained on all seven IDSSE matches. Architecture, 59 features,
  training presets and seeds, hyperparameters and persistence are unchanged.
- **Threshold.** It comes from the pre-declared `m1.select_threshold` procedure (inner
  leave-one-match-out over the seven training matches) and is used as returned.
- **Labels.** The model and its threshold use IDSSE's DFL `BallStatus` labels. No BAR-ATH ground
  truth (public or private) and no BAR-ATH output plays any part.
- **Performance evidence** remains the LOMO, league-out and robustness evaluation of the validation
  models. The final model's training-set score is an implementation sanity check only.
- **Deployment, Option A.** The scikit-learn pipeline is saved with joblib, with dependencies
  pinned. Its sha256 is kept outside the artifact (external manifest and `configs/v2.json`) and
  verified before unpickling. Inference reuses `m1.persist` and `m1.run_with_dead`.
- **Implementation.** `pvext/v2/final.py`, `scripts/v2_train_final.py`, `pvext/v2/deploy.py` and
  `scripts/run_v2.py`. The plan is
  `docs/superpowers/plans/2026-11-05-pitchvision-v2-final-m1.md`.
