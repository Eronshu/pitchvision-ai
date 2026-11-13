# PitchVision AI — M1 technical document (historical)

> **Historical document.** It describes M1, the dead-ball model that is now the fallback. The
> submitted method is HA3, described in [`docs/submission/PitchVision AI.md`](submission/PitchVision%20AI.md).
> Items marked TBD below were never filled in; the release measurements are in
> [`docs/HA3_RELEASE.md`](HA3_RELEASE.md).

Team **PitchVision AI**, More than a Hack 2027 Phase 1 (FC Barcelona vs Athletic Club).
This draft becomes the submitted `PitchVision AI.md` / `.pdf`. Items marked **TBD** are filled in only
after the approved final training and release runs.

## 1. Architecture

```
competition ATD ─► atd.py (frame_id − 1 → video frame) ─► V2 features (59, pvext/v2/features.py)
   ─► final M1 logistic regression ─► dead probability ─► V1 persistence (1 s smoothing,
      DEAD ≥ 2 s, ALIVE gaps < 1 s absorbed) ─► dead mask
   ─► frozen V1 possession machine (pitchvision/possession.py) ─► H / A / D / X ─► CSV + manifest
```

Each graded half is labelled independently. Video frames 11,185–90,651 and 110,808–190,197 are
graded (DATA.md). Every other frame is `X`. The output is `frame,label` with 199,319 rows.

## 2. Frozen V1: the rule-based possession algorithm

V1 (`pitchvision/`, `configs/v1.json`, frozen at commit `a8ee2b7`) is described in
[`docs/PITCHVISION.md`](PITCHVISION.md). It has three parts:

1. **Alive/dead.** A trigger → persist → restart state machine. Triggers are an out-of-play ball,
   or several soft cues together: a resting ball, a slow field, a missing ball.
2. **Possession.** An H/A/UNKNOWN state machine with hysteresis. Control evidence is distance ×
   margin over the other team × relative motion. Loose, contested, in-flight or missing balls keep
   the last possession, and each live stretch is back-filled to its first clear controller.
3. **No learning.** All thresholds are stated priors.

V1 stays the **fallback** submission, generated with `scripts/run_v1.py`, which is unchanged.

## 3. M1: the learned alive/dead classifier, and why only this module changed

**Why only this module.** On an external benchmark (section 5), V1's dead-ball detection was its
weakest part on all seven matches:
- D recall 0.29–0.54 on the documented-degradation preset, against D precision around 0.95;
- stoppage-start F1 at ±1 s of 0.13–0.22.

M1 replaces only `alive.dead_mask`. Possession logic is unchanged (section 12).

**Model.** `StandardScaler` → `LogisticRegression(C=1, L2, lbfgs, max_iter=2000, random_state=0)`.

**Features.** 59 per-frame, slot-invariant features computed only from the ATD:
- ball presence and gap length;
- ball speed and acceleration derived from positions (the ATD ball-speed field is never used);
- out-of-play and exit plausibility, distances to lines, goal area and corners;
- field motion: quantiles of position speed and of the ATD speed field, the share of players
  moving, deceleration;
- shape: team centroids and spread, the own-half split at kick-offs, players in the boxes, near
  the ball and near the touchline;
- frozen V1's own dead score and mask;
- ±1/3/10 s centred window means and maxima.

No targets, DFL-only fields or competition labels are used as inputs.

**Decoding.** The model's probability goes through V1's own persistence: 1 s centred mean, then
the threshold, DEAD runs shorter than 2 s dropped, ALIVE gaps shorter than 1 s absorbed. It then
replaces V1's dead mask in the frozen possession machine.

## 4. Final training and threshold selection

- **One model, all seven IDSSE matches.** Training uses the frozen M1 caches:
  - presets mild/doc/severe × training seeds 100 and 101;
  - every 5th frame inside graded blocks;
  - **TBD** rows (about 1.2 million expected).

  Each of the 42 caches is checked against the frozen M1 manifest's sha256. Feature names and order
  must equal the extractor's 59 features.
- **Threshold.** The pre-declared procedure `m1.select_threshold`:
  - inner leave-one-match-out over the seven training matches;
  - the threshold from {0.30, 0.35, …, 0.70} that maximises the mean per-match D F1 after
    persistence;
  - ties go to the value closest to 0.5.

  The returned value is used as it is: **TBD**. It is not replaced by the 0.40 that all nine
  validation folds selected.
- **Labels used.** The classifier **is trained on IDSSE labels**: DFL `BallStatus` (dead vs
  alive) of the seven IDSSE matches is the training target, and the threshold is chosen on those
  same IDSSE labels.
- **Labels not used.** No BAR-ATH ground truth, neither the public 5-minute window nor the private
  full-match file, and no BAR-ATH output or diagnostic is used for training, threshold selection or
  calibration.
- `scripts/v2_train_final.py` writes `m1_final.joblib`, `MANIFEST.json` and `release_config.json`.
  The manifest records:
  - the threshold and its curve, the model sha256, training matches and rows;
  - cache and code hashes, dependency versions, the git commit;
  - IDSSE attribution;
  - training-set scores, labelled as a sanity check only.

## 5. External training data: IDSSE

- **Source.** Seven matches of the open DFL/Sportec tracking-and-event dataset (IDSSE):
  - Bundesliga: J03WMX, J03WN1;
  - 2. Bundesliga: J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9, all Düsseldorf home matches.
- **Licence.** CC BY 4.0. Attribution, exactly as recorded with every IDSSE-derived artifact
  (`pvext/idsse.ATTRIBUTION`):

  > IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB), published
  > by Bassek, M. et al. (2025), 'An integrated dataset of spatiotemporal and event data in elite
  > soccer', Scientific Data 12:195, doi:10.1038/s41597-025-04505-y; dataset
  > doi:10.6084/m9.figshare.28196177 (v1). Licensed CC BY 4.0. Changes: converted from DFL XML to
  > arrays and to the competition's ATD text format, and synthetically degraded for benchmarking. No
  > endorsement by the DFL, Sportec Solutions or the authors is implied.

  The raw data and caches are not redistributed. Only the trained model parameters are published.
- **Target.** DFL `BallStatus`, dead vs alive. `BallPossession` was a diagnostic only.
- **Conversion.** Frames are converted to the competition's ATD format (`pvext/convert.py`) and
  re-read by the same loader as the competition data. The conversion was cross-checked against
  kloppy, with agreement on all seven matches.

## 6. Synthetic degradation and its limits

IDSSE tracking is near perfect, while the competition ATD has the ball located on 59.3% of graded
frames, about 20.5 tracked objects and 46 slots (DATA.md).

**How the ATD is imitated** (`pvext/degrade.py`, seeded, with no access to labels):
- occlusion gaps for players;
- smooth position noise;
- tracklets cut into slots;
- bursty ball dropout, more likely when the ball is high or crowded;
- false ball detections.

**Presets.**
- `doc` reproduces the documented figures.
- `mild` and `severe` bracket it.
- Gap lengths, noise and tracklet lengths are assumptions.

**Limits.** The real ATD's error structure is unknown. Robustness to five further, pre-declared
degradations was tested (section 7), but that does not prove the competition data lies inside the
tested range.

## 7. Validation evidence (IDSSE labels, not competition accuracy)

**Leave-one-match-out**, 7 folds, `doc` preset, evaluation seeds 0–2, held-out matches only:

| | D F1 | D precision | D recall | start F1 ±1 s | spellF1 |
|---|---|---|---|---|---|
| frozen V1 | 0.44–0.69 | 0.92–0.97 | 0.29–0.54 | 0.13–0.22 | 0.27–0.35 |
| M1 (LOMO) | 0.89–0.93 | 0.89–0.94 | 0.88–0.94 | 0.36–0.46 | 0.32–0.41 |

**Pre-declared GO criteria.** All six passed: D F1 higher in 7/7 matches, start F1 higher in 7/7,
median precision change −0.023, no spellF1 drops, league-out higher in both directions (+0.24 and
+0.32), and mild and severe both higher in 7/7.

**Robustness study.**

| test | result |
|---|---|
| ablations without ball-presence features, without speed-field features, and without both | each still passes all six criteria |
| five alternative degradations: uniform ball dropout, long gaps, speed field from positions, noisy ball, sparse players | M1 is above V1 on D F1 in 7/7 matches under every one, and meets the pre-declared robustness reading |
| reproducibility | the frozen experiment reproduces exactly: 189 of 189 input and prediction hashes |

**Where M1 is weaker.**
- D precision is about 0.02–0.06 lower than V1's.
- More stoppages are detected early.
- Short stoppages (under 5 s) are still often missed.

## 8. BAR-ATH diagnostic: an inference check only

The seven validation models were run locally on BAR-ATH by the project lead, to show that the
pipeline executes:
- 199,319 frames;
- 59 finite features;
- manifest-verified models.

This is **not accuracy evidence**. Its outputs (D share, transitions, agreement with V1) were not
used, and must not be used, to choose models, thresholds or features, or to calibrate anything.
The public 5-minute ground truth is used only to check that the pipeline is correct, never to
compare or tune.

## 9. Validation models and the deployment model

| | 7 LOMO validation models | final deployment model |
|---|---|---|
| training matches | 6 each, 1 held out | all 7 |
| threshold | inner LOMO over its 6 training matches | inner LOMO over all 7 |
| purpose | honest held-out performance estimates | the submitted predictor |
| performance evidence | yes (section 7) | none of its own: its training-set score is a sanity check only |

## 10. Packaging, dependencies and reproduction

**Packaging.**
- The model ships as `models/m1_final.joblib`: a scikit-learn pipeline plus its metadata.
- Its sha256 is recorded outside the file, in `configs/v2.json`.

**Pinned dependencies** (`requirements-v2.txt`): Python 3.13.2, numpy 2.5.4, pandas 3.0.6,
scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0.

**Checks in `scripts/run_v2.py`.**

Before any data is read, it refuses to run on any mismatch:
- the V1 config hash;
- persistence and FPS;
- installed dependency versions;
- the model file's sha256, checked before unpickling, so only our verified artifact is ever loaded;
- the model type, metadata, threshold and feature schema.

After feature extraction, it checks:
- feature names and order, width and finiteness;
- prediction length and label set, and `X` outside the graded halves.

**Manifest.** It records the config, model, feature-schema, source and output hashes, the git
commit and tree state, and the smallest decision margin (how close any smoothed probability came
to the threshold). It keeps `run_v1.py`'s keys, so `scripts/verify_submission.py` checks it.

**Estimates, to be measured.** Runtime is roughly a minute, and peak memory is well under 1 GB
(199,319 × 59 float32 features are about 47 MB). Actual runtime and peak memory: **TBD**, measured
with `/usr/bin/time -l` on macOS and Linux.

## 11. Label-definition mismatch

- **IDSSE's dead ball is DFL's `BallStatus`.** On these matches it covers about 45% of graded
  frames. M1 learns that rate: its predicted D share is about 0.45, against V1's about 0.22.
- **The competition's `D` may be defined more narrowly** (issue #31). If so, M1's extra and earlier
  stoppages become false D labels, and they cost spellF1, which splits possession spells at D.
- **This cannot be measured before scoring.** There is no competition ground truth to use, and the
  public 5 minutes may not be used for tuning.

## 12. Unchanged H/A possession logic

H/A decisions are frozen V1, so M1 improves only where the dead mask was wrong. On IDSSE, spellF1
rose only by +0.02 to +0.09. Possession transitions remain a limitation: changes in a contested
ball, and the team after a restart. Improving them is separate future work and was not attempted.

## 13. Fallback

`scripts/run_v1.py --config configs/v1.json` regenerates the frozen V1 file, with its own manifest
and strict verification. The official `baseline_smooth50` is available through `--fallback-baseline`.

## 14. Reproduction from a clean clone

```bash
git clone <public-repository-url> pv && cd pv
```

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-v2.txt
```

```bash
.venv/bin/python -m pytest -W error
```

```bash
export MTAH_DATA_DIR=/path/to/competition/bundle
```

```bash
.venv/bin/python scripts/run_v2.py --config configs/v2.json --out /tmp/pv_v2
```

```bash
.venv/bin/python evaluation/validate_submission.py "/tmp/pv_v2/PitchVision AI.csv"
```

```bash
.venv/bin/python scripts/verify_submission.py /tmp/pv_v2/manifest.json --strict
```

```bash
.venv/bin/python scripts/verify_v2_release.py /tmp/pv_v2/manifest.json
```

```bash
cmp "/tmp/pv_v2/PitchVision AI.csv" "/path/to/submitted/PitchVision AI.csv" && echo identical
```

### Two different reproducibility objectives

1. **Reproducing the submission** (the competition requirement, RULES §3). Running this repository
   at the release commit, with the competition data and the **released model file**
   (`models/m1_final.joblib`, sha256 in `configs/v2.json`), regenerates the exact submitted CSV, byte
   for byte. This does not involve retraining. `scripts/verify_submission.py --strict` and
   `scripts/verify_v2_release.py` check it.
2. **Reproducing the training** (a separate provenance objective). The final model was trained
   once, from the IDSSE caches, with the training code of commit
   `7028d81c12eb4b2a009822a356f7391804b0ec21`. All 13 training-relevant source files recorded at
   training time are byte-identical to that commit. `scripts/v2_train_final.py` can retrain from
   IDSSE, but a retrained model is not guaranteed to be byte-identical across platforms or library
   builds, and it is not what produces the submission.

### Provenance record

- **The training run is traceable to commit `7028d81`.** It is recorded in the run's external
  manifest and source snapshot (`MANIFEST.json`, `PROVENANCE.json`).
- **A later change to `scripts/v2_train_final.py`** (sha256 `09ae9f28…` → `fcb3531a…`) only widened
  the output-directory guard. It now also refuses paths below the frozen experiment directories.
  - It runs before training, and it leaves the arguments passed to training unchanged.
  - The recorded run's destination is accepted by both the old and the new guard.
  - It is **not** part of the original training run and is not claimed to be.
  - The current training script is therefore not byte-identical to the training-time snapshot. All
    other training-relevant sources are.
  - The change is documented in `PROVENANCE_ADDENDUM_1.json`, next to the model's training records.
- **The model was not retrained.** Its sha256 remains
  `4b2b59d0879f484eba334ccb5a1ceac1531736d34f7624dba2513a1a64788bf8`.
