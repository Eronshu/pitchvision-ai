# PitchVision HA1 — learned H/A possession with switch-penalty decoding: experiment protocol

Fixed on 2026-11-06, before any HA1 model is fitted or evaluated. Approved by the project lead
("Approve experiment"). Running each stage needs its own approval.

## Why this experiment

This comes from a read-only analysis of the frozen doc evaluation: V1 and M1 regenerated and
hash-verified, 7 matches × seeds 0–2.

| configuration | spellF1 |
|---|---|
| M1 | 0.370 |
| perfect dead mask with V1's H/A | 0.463 |
| M1's dead mask with true H/A | 0.840 |

H/A assignment, not dead-ball detection, is the dominant loss:
- 54% of true H↔A switches are missed;
- 37% of predicted switches are spurious;
- predicted spells number only 0.68× the true ones;
- team accuracy is 0.72 in the 3 s after restarts and 0.77 with the ball missing.

## Candidate HA1

**Dead mask.** Frozen M1, unchanged. HA1 replaces only the H/A decision inside M1's alive frames.

**Per-frame model.** P(home in possession) from ATD-only features (`pvext/ha/features.py`). Every
feature is **antisymmetric**: home−away differences, or values signed by each team's attacking
direction. With `StandardScaler(with_mean=False)` and `LogisticRegression(C=1, fit_intercept=False,
lbfgs, max_iter=2000, random_state=0)`, swapping the two teams maps P to exactly 1−P. A home bias,
and with it team identity, cannot be learned.

**Feature groups.**
- per-team nearest and second-nearest player distance to the ball, and players within 3, 5 and 10 m;
- V1's ball–player relative speed;
- ball velocity and position signed by the attacking direction (from team centroids, per half);
- a sticky last-close-team memory and its decayed version;
- centred ±0.5 s, ±1 s and ±3 s means, and past and future 1 s means.

The ball position is carried forward for at most 1 s.

**Decoding** (`pvext/ha/decode.py`).
- **Segments.** A two-state Viterbi runs on each alive segment, a maximal non-dead run inside a
  graded block.
- **Score.** Each frame's evidence is `dt · log p` for H and `dt · log(1−p)` for A, where `dt` is
  1/fps in seconds. Each switch costs λ nats.
- **Start.** Each segment starts from a uniform prior.
- **Units.** Because the evidence is scaled by `dt`, λ means the same at 5 fps (training rows) and
  25 fps (evaluation).

**Training data.**
- training copies: presets mild/doc/severe × seeds 100, 101, as for M1;
- every 5th frame inside graded blocks;
- target: IDSSE DFL `BallPossession` on true-alive rows only.

## Splits, leakage and the single held-out stage

| stage | matches | use |
|---|---|---|
| **dev** | the five 2. Bundesliga matches (J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9) | leave-one-match-out: 5 folds, each trained on the other 4 |
| **lockbox** | the two Bundesliga matches (J03WMX, J03WN1) | never read during dev; evaluated **once**, with a model trained on all 5 dev matches |

**λ selection, training data only.**
- Grid: λ ∈ {0.5, 1, 2, 4, 8, 16}.
- Inner leave-one-match-out over the fold's training matches, scored with the official
  `spell_f1_mean` on the strided training series.
- The inner folds decode within the training matches' own true dead mask.
- Ties go to the larger λ.

**Dead masks at evaluation.**
- dev match m: frozen M1 `lomo_m`;
- lockbox: frozen M1 `league_J03WMX-J03WN1`, trained on the 2. Bundesliga matches only.

**Other guards.**
- Evaluation uses the frozen evaluation copies only: presets mild/doc/severe, seeds 0, 1, 2.
- V1 and M1 predictions are regenerated and must match the frozen hashes.
- No BAR-ATH data, public GT or BAR-ATH diagnostics are used.
- Dev never reads lockbox data, which is enforced in code.
- The lockbox stage refuses to run if its report exists, or if the dev report does not.

## Comparison and GO criteria (doc, mean over seeds 0–2)

The candidate is **M1+HA1** (M1's dead mask + HA1). The reference is **M1** with the same dead mask
(dev: `m1_lomo`; lockbox: `m1_league`). Frozen V1 is reported as well.

GO requires all four:

1. **Lockbox.** spellF1 above M1 in both matches, and mean gain ≥ +0.04.
2. **Dev.** spellF1 above M1 in ≥ 4 of 5 matches, and mean gain ≥ +0.04.
3. **Other presets.** Lockbox mean spellF1 gain > 0 on mild and on severe.
4. **Spell count.** On doc, the ratio of predicted to true spells lies in [0.7, 1.5] in every
   lockbox and dev match.

**Otherwise:**
- **within noise:** gains are positive but below the thresholds; keep M1;
- **NO-GO:** keep M1.

There is no second lockbox attempt.

**Secondary diagnostics, reported only:**
- macroF1;
- team accuracy on true-alive frames;
- the H↔A switch on-time, late, early and missed shares;
- the spurious switch share;
- team accuracy after restarts and with the ball missing;
- mild and severe dev results.

## Known risks, stated in advance

1. **Label definition.** λ is selected on DFL's possession granularity (median spell 3.7 s). The
   competition GT may count fewer, longer possessions, and this cannot be checked without the
   prohibited public GT.
2. **Degradation realism.** The degraded copies are synthetic.
3. **Sample size.** 5 dev matches and a 2-match lockbox. ±0.04 is the official full-match noise
   band.

## Budget

| date | step |
|---|---|
| 2026-11-06/07 | implementation and synthetic tests |
| 2026-11-08 | dev run, after approval |
| then | lockbox run, after a separate approval |

If there is no GO by **2026-11-11**, stop and submit M1.
