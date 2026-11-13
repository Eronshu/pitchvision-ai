# PitchVision AI — HA3 technical document (release candidate)

**Release status.** HA3 is the primary release candidate by an **explicit project-lead override**
of the pre-registered GO verdict.

HA3 did **not** pass the pre-registered criteria. Criterion 4 (spell-count ratio) fails both in
its original form and in a post-hoc amended form, and both verdicts are `WITHIN_NOISE_KEEP_M1`
(§5). The frozen M1 is the fallback (§7; `docs/TECHNICAL_DOCUMENT.md`).

**No competition accuracy has been measured.** Every number below comes from IDSSE Bundesliga
labels, which use a different possession definition from the competition GT.

## 1. Pipeline

```
ATD ─► M1 features (59) ─► M1 dead-ball model ─► V1 persistence ─► dead mask
     │                                                                  │
     │                                         frozen V1 machine ◄──────┤   (= M1 labels; D, X)
     └► HA3 features (26) ─► symmetric HGB P(home) ─► Viterbi per alive segment ─► H/A
```

HA3 changes only the H/A labels of alive frames. D and X are exactly those of M1 under the same
dead-ball model, and this is checked at run time.

### Why H/A

Oracle experiments in the M1 phase showed that most of the remaining spellF1 loss comes from the
team assignment, not from the alive/dead boundary.

### HA3 features (`pvext/ha/features.py`, frozen)

There are 26 features in all:
- 21 team-antisymmetric features (home minus away): nearest and second-nearest distance,
  player counts within 3, 5 and 10 m, relative speed, ball velocity in the home attack direction,
  and similar;
- 5 touch features computed on the raw ball:
  - a touch is a velocity change of at least 3 m/s over ±3 frames with the nearest player
    within 2 m;
  - the features are the last and next touch team, each also with a 2 s decay, plus a 3 s touch
    balance.

The touch windows are fixed frame counts, so HA3 is validated at 25 fps only, and the loader
refuses any other rate.

### Classifier

Scikit-learn `HistGradientBoostingClassifier` trained on (X, y) ∪ (−X, ¬y), on at most 300k
strided rows (seed 0). Its prediction is p = ½(g(x) + 1 − g(−x)), so swapping the teams swaps
P(home) exactly.

### Decoder (`pvext/ha/decode.py`, frozen)

A two-state Viterbi per alive segment with a uniform start:
- the evidence is dt·log p;
- every team switch costs λ.

λ = 0.5 was chosen from {1/32 … 2} by inner leave-one-match-out on the training matches only.
Ties go to the larger λ.

## 2. Models and training data (provenance and disclosure)

| artifact | sha256 | trained on | role |
|---|---|---|---|
| `models/ha3_final.joblib` | db988179bdf367c8556f884d5a16636296e5b7297d4b6d195d8013ffc7ba955b | IDSSE dev matches J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9 (presets mild/doc/severe, seeds 100/101) | HA3 classifier, λ = 0.5; the model evaluated on the lockbox |
| `models/ha3_dead_m1_league.joblib` | 850c4daad35b4db3e0d862bbccfc9f49faff17110c247c93734fad32a52d1290 | the same 5 dev matches | M1 dead-ball model used in the HA3 lockbox evaluation |
| `models/m1_final.joblib` | 4b2b59d0879f484eba334ccb5a1ceac1531736d34f7624dba2513a1a64788bf8 | all 7 IDSSE matches | M1 fallback only; not used for the HA3 labels |

### Training data

All three models are trained only on IDSSE open data (Bassek et al. 2025, *Scientific Data*
12:195; CC BY 4.0). The IDSSE data was converted to the competition's ATD format and
synthetically degraded.

No competition data, no BAR-ATH ground truth and no public five-minute labels were used to
train, tune or select any HA model, the dead-ball pairing, or HA3 over M1.

HA targets are the IDSSE BallPossession team. During training, HA decodes with the true IDSSE
dead mask. BallStatus and BallPossession are targets only, never inputs.

### Dead-ball pairing

The release uses the dev-trained "league" model, not the all-seven M1 final. That makes the
deployed pipeline exactly the one evaluated on the lockbox:
- `deploy.label` is bit-identical to the evaluation code on IDSSE dev runs;
- the two dead masks disagree on 0.55–0.76% of graded frames on the dev matches.

## 3. Evaluation protocol

### Split

- **Dev:** 5 matches, used for leave-one-match-out evaluation and every model choice.
- **Lockbox:** J03WMX and J03WN1. It was evaluated once, with a model trained on the dev matches
  only.

### Evaluation settings

- Metric: the official spellF1.
- Presets: mild, doc and severe degradation, with 3 evaluation seeds each.

### Pre-registered GO criteria

1. Lockbox: both matches improve over M1, mean gain ≥ +0.04.
2. Dev: at least 4 of 5 matches improve, mean gain ≥ +0.04.
3. Lockbox: mild and severe mean gains > 0.
4. Spell-count ratio (predicted/true) within [0.7, 1.5] on every match.

### Research path

| step | result | decision |
|---|---|---|
| HA1 | linear classifier | accepted |
| HA1b | | rejected |
| HA2 | symmetric gradient boosting; dev +0.067 vs M1 | accepted |
| HA3 | HA2 + touch features; dev +0.079 vs M1, +0.012 vs HA2 | frozen as the candidate |

The dev results carry selection bias from these choices.

## 4. Results (IDSSE, spellF1, doc preset unless stated)

| set | match | V1 | M1 | HA3 | HA3 − M1 | mild Δ | severe Δ |
|---|---|---|---|---|---|---|---|
| dev | J03WPY | 0.306 | 0.375 | 0.457 | +0.082 | +0.113 | +0.073 |
| dev | J03WOH | 0.321 | 0.381 | 0.472 | +0.090 | +0.126 | +0.082 |
| dev | J03WOY | 0.314 | 0.368 | 0.436 | +0.068 | +0.108 | +0.079 |
| dev | J03WQQ | 0.337 | 0.409 | 0.484 | +0.075 | +0.076 | +0.077 |
| dev | J03WR9 | 0.352 | 0.373 | 0.452 | +0.079 | +0.133 | +0.065 |
| lockbox | J03WMX | 0.325 | 0.359 | 0.419 | +0.060 | +0.075 | +0.067 |
| lockbox | J03WN1 | 0.274 | 0.321 | 0.401 | +0.080 | +0.111 | +0.071 |

- Dev M1 is the leave-one-match-out model. Lockbox M1 is the dev-trained league model.
- HA3 is above M1 in all 21 match × preset cells.

**Lockbox, M1 → HA3**

| | J03WMX | J03WN1 |
|---|---|---|
| macroF1 | 0.832 → 0.849 | 0.782 → 0.807 |
| team accuracy | 0.843 → 0.871 | 0.775 → 0.815 |
| spurious switch share | 0.34 → 0.26 | 0.32 → 0.24 |

## 5. GO verdict and the override

| criterion | result |
|---|---|
| 1 lockbox | PASS (mean +0.070) |
| 2 dev | PASS (5/5, mean +0.079) |
| 3 lockbox mild / severe | PASS (+0.093 / +0.069) |
| 4 original: ratio in [0.7, 1.5] | **FAIL**: lockbox 0.564 and 0.594; dev J03WPY 0.665, J03WOH 0.696, J03WOY 0.679 |
| 4 amended, post hoc: \|r_HA − 1\| ≤ \|r_M1 − 1\| + 0.02 and r_HA ≤ 1.5 | **FAIL** on J03WMX (0.4364 > 0.4336); passes on the other 6 matches |
| verdict, original | `WITHIN_NOISE_KEEP_M1` |
| verdict, amended | `WITHIN_NOISE_KEEP_M1` |

The verdict label means "every lockbox match improved, but not GO". It does not mean the spellF1
gains are within noise.

M1 and HA3 have similar spell-count ratios:

| | J03WMX | J03WN1 |
|---|---|---|
| M1 spell-count ratio | 0.586 | 0.589 |
| HA3 spell-count ratio | 0.564 | 0.594 |

On 6 of the 7 matches the HA3 and M1 ratios are within 0.025 of each other.

**Correction (2026-11-13).** An earlier version of this document said that the shortfall was
inherited from M1's dead-ball segmentation. A later read-only loss attribution, on the frozen
held-out predictions of the 5 dev matches, does not support that claim:
- The official spell count splits into live-ball intervals plus direct H/A switches inside them.
- Per match, the truth has about 100 intervals and 310 switches. HA3 predicts about 111
  intervals, slightly too many, and only about 189 switches.
- The shortfall is therefore in direct team switches, not in the dead-ball segmentation. See
  `docs/submission/PitchVision AI.md` §6.

**Project-lead override.** HA3 is selected because spellF1 is the competition metric and HA3
improved it on every dev and lockbox match. Both failed verdicts are kept in the record unchanged.

## 6. Limitations

- **Transfer.** The competition GT definition, tracking quality and league differ from IDSSE.
  The degradation presets only approximate the competition tracking. The touch features depend
  on the raw ball quality, and under the severe preset HA3 gains nothing over HA2.
- **Too few spells.** Both M1 and HA3 predict roughly 0.56–0.87 of the true spell count. The
  deficit is in direct H/A switches. Possessions shorter than 3 s are rarely recovered (IDSSE dev
  observations; see the correction in §5).
- **Small, partly seen lockbox.** The lockbox has only 2 matches, and it was not untouched:
  - pre-HA aggregate analyses used its BallPossession labels;
  - the dev M1 leave-one-match-out models used its BallStatus labels;
  - its M1 and V1 scores were seen in the M1 phase.
- **Version-bound pickles.** The models are pickles bound to scikit-learn 1.9.1 and the pinned
  stack (`requirements-ha3.txt`); the loaders refuse any other version.

## 7. Fallback

M1 (`configs/v2.json` → `models/m1_final.joblib`, `scripts/run_v2.py`) is unchanged and is
checked every time HA3 loads. V1 (`scripts/run_v1.py`) stays the last-resort fallback. See
`docs/HA3_RELEASE.md`.
