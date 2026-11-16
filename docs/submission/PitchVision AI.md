# PitchVision AI — Technical Document

More than a Hack 2027, Phase 1 · Team: **PitchVision AI**

Code: [github.com/Eronshu/pitchvision-ai](https://github.com/Eronshu/pitchvision-ai), tag **`submission-v2`**.
Output: **`PitchVision AI.csv`**, 199,319 data rows, indexed by tactical-video frame.

## 1. Method

PitchVision AI is an offline possession labeller using the supplied tactical automatic tracking data (ATD). Two histogram gradient-boosting models estimate dead-ball probability and the team in possession. A joint three-state Viterbi decoder converts these estimates into a continuous sequence of home possession (H), away possession (A), and dead ball (D).

The dead-ball model uses **59 features** describing ball visibility, motion, distance to pitch boundaries, player movement, team shape, and temporal summaries. The team model uses **38 features** describing player–ball proximity, relative motion, touch evidence, sustained proximity to the freshly observed ball, and the reliability and age of those observations. Team-swap augmentation and symmetric prediction make the team-probability estimator equivariant to swapping home and away.

This design combines nonlinear tracking cues with temporal consistency. Observation-reliability features distinguish fresh ball evidence from carried or missing positions. Joint decoding selects stoppages and team possession together, avoiding an irreversible dead-ball decision before team evidence is considered.

Let `d` estimate P(dead) and `h` estimate P(home | live). The decoder uses:

```
P(D) = d
P(H) = (1 − d) h
P(A) = (1 − d) (1 − h)
```

It maximises the sum of log state probabilities, scaled by `1/25` per frame, minus transition costs: **0.5** for H↔A, **0.25** for H/A→D, and **4.0** for D→H/A. Probabilities are clipped to avoid logarithms of zero. Each graded half is decoded independently, with no preferred initial state. Frames outside the organisers' graded halves are X. There is no additional minimum-duration rule or output smoothing.

## 2. Possession conventions and missing observations

- **Loose balls, challenges and deflections:** these have no separate output class. Temporal evidence and transition costs determine whether possession stays with the previous team or switches. A brief touch does not automatically transfer possession.
- **Stoppages:** D represents the ball being out of play or play being stopped. The dead-ball model learns from the external dataset's `BallStatus` labels.
- **Restarts:** D has no team owner. The team after a stoppage is selected from the evidence after the decoder leaves D; possession is not automatically carried across the stoppage.
- **Missing ball:** positions are carried for up to one second; after that, ball-relative features become neutral. Reliability features describe whether the ball is currently observed and how old the available evidence is.

The system is offline, not causal: dead-ball features use centred windows up to ±10 seconds; team windows extend up to three seconds ahead; next-touch features can look further ahead within the same half. The released implementation requires 25 Hz inputs.

## 3. Input assets and training data

**Prediction input.** The final pipeline reads tactical ATD metadata and player/ball tracking. Positions are represented in pitch coordinates, so it does not require video-to-pitch homography. Tactical video, broadcast video and broadcast ATD are not integrated. Broadcast tracking would additionally require alignment to the tactical-video timeline. Smart Tagging is not used: earlier external-data experiments with simulated restart tags did not establish a benefit for the tested integration. Those experiments do not establish that the supplied Smart Tagging or a different integration would be ineffective.

**Training.** Both models use five other matches from the IDSSE positional dataset: J03WPY, J03WOH, J03WOY, J03WQQ and J03WR9 (German 2. Bundesliga), totalling **715,116 graded frames at 25 Hz**. Targets are the provider-supplied `BallStatus` and `BallPossession` fields distributed with the positional data. The source record does not establish whether those fields were originally assigned automatically or manually.

IDSSE data were collected by Sportec Solutions for DFL and released under **CC BY 4.0**: Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12:195 (2025), [paper](https://doi.org/10.1038/s41597-025-04505-y), [dataset](https://doi.org/10.6084/m9.figshare.28196177). Training labels come from these external matches.

Tracking is converted to the ATD representation and degraded with seeded player occlusions, coordinate noise, fragmented tracks, bursty ball dropout and false ball detections. Three severity presets and two training seeds are used, with a five-frame sampling stride.

| Model | Training samples | Main settings |
|---|---|---|
| Dead-ball classifier | 300,000 rows, equally sampled across matches with natural within-match class balance | 200 iterations, 15 leaves, learning rate 0.05, L2 1 |
| Team classifier | 300,000 live-play rows sampled from 479,304 eligible rows, plus team-swapped copies | 200 iterations, 31 leaves, learning rate 0.1 |

Both models disable early stopping. Decoder costs were selected using match-level cross-validation on the five development matches. The repository distributes trained models and inference code; it does not reproduce the research training pipeline.

## 4. Evaluation and limitations

On IDSSE, leave-one-match-out development evaluation excludes the outer held-out match from training and uses inner folds for parameter selection. Mean spellF1 is **0.528** under the standard degradation preset, **0.642** under mild degradation and **0.421** under severe degradation. Each result averages three evaluation seeds per match and then the five matches.

These five matches were repeatedly used to develop the approach: the results are **development evidence, not independent confirmation**. The evaluated fold models train on four matches; the released models train on all five. These figures are not scores for the competition match.

A separate check of the released models uses ten matches from [SkillCorner Open Data](https://github.com/SkillCorner/opendata), under its MIT licence. On samples where the provider names a team, mean agreement is **0.800**: **0.154** name the wrong team and **0.046** are predicted D. Unknown provider labels are excluded. These are silver labels without dead-ball truth, so this check cannot measure missed stoppages or official spellF1; prior team exposure to all ten matches is not established.

The main limitations are:

- **Domain shift:** tracking errors and possession definitions differ between datasets. Aggregate competition tracking gap statistics informed an exploratory long-gap stress test; this was not independent validation.
- **Missing ball near transitions:** long gaps and gaps around switches or restarts can cause missed possessions or incorrect stoppages. Fully absent ball tracking is particularly difficult.
- **Short possessions:** development recall is approximately 3% for possessions below one second and 19% for one to three seconds.
- **Coupled uncertainty:** uncertain team evidence can favour D even when its probability is below 0.5. Transition costs also affect boundary placement.
- **Validation coverage:** the released models lack independent full-match H/A/D validation. Reproduction has been checked on macOS; other platforms are not verified.

With more time, priorities would be independent full-label evaluation on other matches, better modelling of missing-ball patterns around transitions, and separately validated fusion of tagging or visual evidence.

## 5. Reproduction

Use Python **3.13** (verified with 3.13.2) and the exact dependencies in `requirements-release.txt`: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0 and threadpoolctl 3.8.0.

```bash
git clone https://github.com/Eronshu/pitchvision-ai
cd pitchvision-ai
git checkout submission-v2
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-release.txt
export MTAH_DATA_DIR=/path/to/the/competition/data
.venv/bin/python scripts/run_astra5_combo.py \
  --config candidate/combo.json --out /tmp/pitchvision_submission
shasum -a 256 "/tmp/pitchvision_submission/PitchVision AI.csv"
```

The data directory must contain the organisers' tactical assets in the layout described in `DATA.md`. Output must be outside the repository. The runner writes the CSV and `manifest.json` with integrity and validation metadata. Before reading match data, it checks the configuration, model and source hashes and the required runtime versions.

Expected CSV SHA-256:

```
86b73f4199b47a392d68fa51127ee0b96d14541e0a12e4017552907255938e00
```

The exact submission file passes the official format validator. Synthetic end-to-end, determinism and integrity-rejection checks pass **24/24** and can be run without match data:

```bash
PYTHONPATH=evaluation .venv/bin/python -m pytest -q tests/test_release_c.py
```

The team's code is MIT-licensed; organiser files retain their original licence. Attribution is recorded in `NOTICE.md`.
