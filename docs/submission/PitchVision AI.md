# PitchVision AI — technical document

More than a Hack 2027, Phase 1: FC Barcelona vs Athletic Club (BAR-ATH).

> This document describes **release C** (repository tag `submission-v2`). The team's earlier
> entry, the HA3 release (`submission-v1`), stays in the repository as a fallback; its document
> is `docs/submission/submission-v1.md`. "This candidate" below means release C.

- **Team:** PitchVision AI
- **Possession file:** `PitchVision AI.csv`: 199,319 rows, `frame,label`, labels H/A/D/X.
  sha256 `86b73f4199b47a392d68fa51127ee0b96d14541e0a12e4017552907255938e00`.
- **Repository:** https://github.com/Eronshu/pitchvision-ai, tag `submission-v2`.

> **Status of the evidence.** Every accuracy number in this document comes from external open
> data (IDSSE: German 2. Bundesliga; SkillCorner: A-League), whose possession and dead-ball definitions differ from the
> competition's. **No accuracy figure for BAR-ATH is reported or relied on here.** That is not the
> same as no contact with BAR-ATH labels: one script computed the official metrics on the public
> five-minute ground truth for `submission-v1` and both candidates, stored them locally and
> displayed some results derived from them (§10.5). All five IDSSE development matches were used
> repeatedly to design and compare methods, so the figures are development results, not an
> independent validation. The released models have a silver-label team-agreement check on other
> open data (§6.3) and **no** independent validation with full H/A/D labels.

## 1. Approach

The system is an offline labeller. It reads only the provided tactical tracking (ATD) and labels
every frame of the tactical video:

```
ATD ─► 59 dead-ball features ─► gradient boosting ─► d = P(dead)
  └──► 38 team features ──────► symmetric gradient boosting ─► h = P(home | live)
                 d, h ─► one three-state Viterbi decoder (H / A / D) per graded half ─► H / A / D
                 frames outside the graded halves ─► X
```

1. **Graded halves.** Frames outside the two graded halves (video frames 11,185–90,651 and
   110,808–190,197) are `X`. Each half is decoded independently.
2. **Dead-ball evidence.** A gradient-boosted classifier over 59 ATD features gives a per-frame
   probability `d` that the ball is dead.
3. **Team evidence.** A team-swap-symmetric gradient-boosted classifier over 38 ATD features
   gives a per-frame probability `h` that the home team has the ball, given that play is live.
4. **Joint decision.** One Viterbi decoder chooses the H/A/D sequence. Per second, state D scores
   `log d`, H scores `log(1−d) + log h`, A scores `log(1−d) + log(1−h)`. A change of state costs
   0.5 (H↔A), 0.25 (into D) or 4.0 (out of D). There is no separate smoothing or minimum-duration
   rule.

### What changed from the earlier HA3 release, and why

| stage | HA3 (`submission-v1`) | this candidate |
|---|---|---|
| dead-ball classifier | logistic regression, then 1 s smoothing, threshold and minimum-duration rules | gradient boosting on the same 59 features |
| team features | 26 | the same 26 plus 12 observation-reliability and fresh-ball control features (HA5a) |
| decision | dead mask first, then a two-state decoder inside each live stretch | one joint three-state decoder |

Each change was tested on its own before the combination (§6, §7):

- **Dead-ball classifier.** Most dead-ball frame errors were already wrong in the raw
  probability, before any post-processing. A non-linear classifier on the same features halved
  the raw log-loss on held-out matches.
- **Team features.** The extra features tell the classifier whether its ball evidence is fresh.
- **Joint decoding.** Deciding the dead mask first is irreversible. A joint decoder removed false
  stoppages and placed restarts earlier, because the 1 s smoothing of the old pipeline delays the
  end of a stoppage.

## 2. Possession rules as implemented

- **Who has the ball.** Decided from evidence over time, not frame by frame. The decoder picks
  the sequence that maximises the summed per-second scores minus the transition costs. A change
  of team is placed only where the evidence for the other team outweighs the cost of 0.5.
- **Loose balls, deflections and challenges.** There is no explicit rule. A brief touch by the
  other team rarely carries enough evidence to pay for two changes, so possession stays with the
  previous team. Sustained control by the other team does pay for the change.
- **Dead ball.** The ball is out of play or play is stopped. The model was trained on DFL
  `BallStatus`. A stoppage is entered where the accumulated dead-ball evidence pays for the costs
  of entering and leaving D (4.25 in total). There is no minimum length; in practice the costs
  remove very short stoppages.
- **Uncertain team and dead ball.** When the team evidence is uncertain (`h` near 0.5), the live
  probability is split between two states, so D can win with `d` as low as 1/3. This is a
  property of the joint decision, not an error. On the development data an uncertain team is
  itself associated with a dead ball. We tested a variant that weakens this coupling; it was not
  adopted (§7).
- **Restarts.** A dead ball has no owner (`D`). Possession is not carried across a stoppage: the
  team after a restart is whichever team the evidence supports once the decoder leaves D.
- **Half edges.** Each half starts with no preferred state. Because entering D is cheap and
  leaving it is expensive, the decoder treats the start and the end of a half differently. We
  analysed this only on synthetic sequences.
- **Missing ball.** The last observed ball position is carried for up to 1 s. After that,
  ball-relative features become neutral, and the reliability features tell the classifier how
  old its ball evidence is.
- **No home bias.** Team features are either antisymmetric under a team swap or team-invariant.
  The classifier is trained on each example and its team-swapped mirror and predicts
  p = ½(g(x) + 1 − g(x̃)), where x̃ is the mirrored example. Swapping the teams swaps P(home)
  exactly.

## 3. Features and handling of missing and noisy tracking

**Dead-ball features (59, unchanged).** Per-frame, slot-invariant, from the ATD only: ball
presence and gap length; ball speed and acceleration from positions; out-of-play, line, box and
corner distances; field motion; team shape; V1's own dead-ball score; ±1, 3 and 10 s window
statistics.

**Team features (38).**
- The 26 HA3 features, all home-minus-away: nearest and second-nearest player distance to the
  ball; player counts within 3, 5 and 10 m; relative speed; ball position and velocity in the
  home attacking direction; the last team close to the ball; window means; five touch features.
- Six team-invariant reliability features: whether the ball is seen now; time since it was last
  seen; share of missing ball in the last second; distance of the nearest player of either team;
  time since the last close-control event; time since the last touch.
- Six antisymmetric features computed from the freshly observed ball only: a soft control score
  and its 0.4 s past and future means; the last team close to the fresh ball with a 0.5 s decay;
  the team nearest to the ball when it reappears after a gap; fresh-ball velocity in the home
  attacking direction.

The labeller is offline and uses future frames: dead-ball features use centred windows of up to
±10 s, team window features look up to 3 s ahead, and the next-touch features look ahead to the
next detected touch, however far, within the same half. Windows are fixed in frames, so the
pipeline accepts only 25 fps.

**Missing and noisy data.**
- The ATD locates the ball on only about 59% of graded frames and splits players into short
  tracklets. All features are defined on tracking slots and treat a missing value as neutral.
- Every model was trained on open data that we **degraded synthetically**: player occlusion gaps,
  position noise, tracklets cut into slots, bursty ball dropout, false ball detections. The
  degradation is seeded and independent of the labels. We report three severities: `mild`, `doc`
  (matched to the documented ATD statistics) and `severe`.
- **A known mismatch.** In the training degradation the ball goes missing in short gaps (about
  2 s on average) that are unrelated to play. This is an assumption. Aggregate gap statistics of
  the competition tracking, computed without labels (§10.4), show much longer gaps (median about
  10 s). We did not retrain or change any parameter in response; we added a long-gap stress
  scenario on external data (§6.2).

## 4. Assets used, and assets dropped

| asset | use |
|---|---|
| tactical ATD (players, ball, slots) | **used**, the only input at prediction time |
| IDSSE open data, 5 matches of other games (DFL/Sportec, CC BY 4.0): positions with `BallStatus`/`BallPossession` | **used** for training and development evaluation only (§5), converted to the ATD format and degraded (§3) |
| IDSSE Bundesliga matches J03WMX, J03WN1 | not read for this candidate (they were the held-out matches of the HA3 release) |
| SkillCorner open data (A-League broadcast tracking, MIT licence) | **evaluation only**: a silver-label check of team agreement (§6). Never used for training or tuning |
| public 5-minute ground truth | read by one script, once, for `submission-v1` and both candidates. The script computed the official metrics and a time-shift diagnostic, stored them locally and displayed some results derived from them; see §10.5. Not used to train any model or to set any parameter |
| Smart Tagging, homography, tactical video, broadcast video, broadcast ATD | **not used** |
| HA3 two-stage pipeline (`submission-v1`) | not used to produce this file; kept in the repository as the fallback |
| trusted-contact and sustained-control features (S1) | **rejected** (§7) |
| decoder variant C2 | **rejected** (§7) |

## 5. Training data

**No label of the provided game was used to train any model or to set any parameter, threshold
or decoder cost**; both packages were fixed before any run on competition data (§10.4). The
public five-minute ground truth was read once by a script whose partial results were displayed;
what that could and could not have influenced is stated in §10.5. We did not hand-label any data.

| dataset | source and licence | labels used | quantity |
|---|---|---|---|
| IDSSE positional data | DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB). Bassek et al. (2025), Scientific Data 12:195; dataset doi:10.6084/m9.figshare.28196177 (v1). CC BY 4.0 | per-frame `BallStatus` (dead/alive) and `BallPossession` (home/away), delivered with the data. Used as training and evaluation targets, never as inputs. We did not relabel them | 5 matches (2. Bundesliga J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9); 715,116 graded frames at 25 fps |

**Training data of each model in this candidate.** Both were trained on synthetically degraded
copies (presets mild, doc and severe; seeds 100 and 101; 5-frame stride).

| model | file | trained on | rows |
|---|---|---|---|
| dead-ball classifier | `candidate/models/combo_dead_hgb.joblib` | the 5 development matches | 300,000 rows, the same number from each match, natural class balance |
| team classifier (HA5a features) | `candidate/models/combo_ha5a.joblib` | the 5 development matches | 300,000 rows sampled (seed 0) from the 479,304 H/A rows of the degraded copies, plus team-mirrored copies. The `n_rows` field stored in the model file records the 479,304 |

Both are histogram gradient boosting without early stopping (dead ball: 200 iterations, 15
leaves, learning rate 0.05, L2 1; team: 200 iterations, 31 leaves, learning rate 0.1). The
decoder's dead-ball costs were chosen by leave-one-match-out over the five matches from a grid
fixed in the round's configuration file before the run (§10.3); they lie on a plateau (§8).

**External attribution.** IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec
Solutions (TRACAB). Bassek, M. et al. (2025), *An integrated dataset of spatiotemporal and event
data in elite soccer*, Scientific Data 12:195, doi:10.1038/s41597-025-04505-y. CC BY 4.0. We
converted it to the ATD format and degraded it. No endorsement is implied. The raw data is not
redistributed. SkillCorner Open Data: https://github.com/SkillCorner/opendata, MIT licence.

## 6. Validation (external data, not BAR-ATH)

### 6.1 Development results on IDSSE

**Protocol.** Leave-one-match-out over the five development matches. For each held-out match,
every model, threshold and decoder cost was fitted or chosen on the other four matches only, with
inner models that exclude the inner validation match as well. Metric: the official spellF1,
mean of 3 evaluation seeds per match, then the mean over matches.

**These fold models are not the released models.** The released models are trained on all five
matches and use one fixed pair of decoder costs. The released models themselves have no
evaluation against full H/A/D labels, because no further fully labelled match was available to
us. Their only labelled evaluation is the silver-label team-agreement check in §6.3, which has no
dead-ball labels and is not spellF1.

| `doc` preset | mean of 5 | per-match gain over the HA3 recipe |
|---|---|---|
| HA3 recipe (re-fitted under this protocol) | 0.459 | — |
| + gradient-boosted dead-ball classifier | 0.508 | +0.042 to +0.059 |
| + HA5a team features | 0.517 | +0.046 to +0.064 |
| + joint decoding (no HA5a) | 0.518 | +0.048 to +0.070 |
| **all three (this candidate)** | **0.528** | **+0.053 to +0.084** |

- Under `mild` and `severe` degradation the candidate scores 0.642 and 0.421, against 0.569 and
  0.370 for the HA3 recipe.
- Raw dead-ball log-loss on held-out matches fell from 0.210 to 0.103 with the new classifier.
- Against the dead-ball change alone, the full candidate gains +0.020, in all five matches. It
  matches only about 3 more true possessions per match (19 gained, 16 lost, of about 410); most of
  the gain is fewer false possessions (103 → 86 per match) and better stoppage boundaries.
- The two held-out matches of the HA3 release (J03WMX, J03WN1) were not read for this candidate.

### 6.2 Stress tests on IDSSE (fold models, `doc` copy, 5 matches)

Inputs were altered without retraining. Scenarios marked LC used the IDSSE labels to decide where
to alter the input; the labels never reached the features or the decoder. They are synthetic and
do not reproduce the competition's real conditions. These tests were exploratory, not
pre-registered. The long-gap scenario was designed after, and because of, the aggregate gap
statistics of the competition tracking (§10.4).

| scenario | dead-ball change only | this candidate | matches where the candidate is higher |
|---|---|---|---|
| unchanged `doc` copy | 0.513 | 0.534 | 5 of 5 |
| long ball gaps (median 10 s) | 0.466 | 0.480 | 5 of 5 |
| 1 s false ball once a minute | 0.506 | 0.528 | 5 of 5 |
| 25 → 10 → 25 Hz resampling | 0.499 | 0.519 | 5 of 5 |
| LC: ball missing 15 s inside long live intervals | 0.430 | 0.454 | 5 of 5 |
| LC: ball always visible while dead | 0.524 | 0.544 | 5 of 5 |
| **LC: ball missing ±4 s around every switch and restart** | 0.294 | **0.278** | **2 of 5** |
| **no ball at all** | 0.114 | **0.097** | **1 of 5** |

The last two rows are measured regressions of this candidate against the simpler change.

### 6.3 Cross-vendor silver-label check (SkillCorner)

SkillCorner broadcast tracking carries a provider flag for the team in possession and nothing
else: **no dead-ball labels**. We measure, on the frames where the provider names a team, how
often the prediction names the same team; a predicted D on such a frame counts as an error.
Frames with no provider team are unknown and excluded; they are never treated as dead ball. This
is not spellF1 and cannot detect missed stoppages.

Ten matches, the released models, protocol and match list committed before download, scored once.
Whether any team member had seen these matches before is not established by our records.

| mean of 10 matches | HA3 (`submission-v1`) | dead-ball change only | this candidate |
|---|---|---|---|
| agreement on known-team frames | 0.633 | 0.798 | 0.800 |
| predicted D on known-team frames | 0.241 | 0.047 | 0.046 |
| team accuracy when predicted live | 0.833 | 0.838 | 0.839 |
| wrong team on known-team frames | 0.126 | 0.155 | 0.154 |

- The candidate against the dead-ball change alone: +0.002, 95% interval −0.000 to +0.004 by a
  paired bootstrap over matches. The interval includes zero: this check does not separate them.
- Both candidates name the wrong team more often than HA3 (0.154 against 0.126 of known-team
  frames). HA3 calls D on many of those frames instead, which this measure also counts as an
  error; the two error types are not separable into "better" and "worse" without dead-ball
  labels.
- An earlier batch of three matches gave the same picture; it was used to compare versions and is
  a selection set.

## 7. Development history: what was tried and dropped

Every step used only IDSSE data for training and scoring. What was fixed before results, and
with what record, differs by step and is listed in §10.3; the stress tests were exploratory.

- **HA1 → HA3, HA4.** As in the `submission-v1` document: HA3 was released; HA4 (set-piece tags)
  was rejected.
- **S3: dead-ball classifier.** Logistic regression against gradient boosting, each with and
  without 21 extra validity features. The classifier change gained +0.049; the extra features
  gained nothing and were dropped.
- **S2: joint decoding** on frozen probabilities: +0.014 with the old classifier. Calibration
  made no difference.
- **S1: trusted contact and sustained control.** Validity masks, fresh-only memories,
  identity-safe velocities, a trusted-contact detector and a control score were added step by
  step. HA5a's group gain reproduced (+0.010); none of the added steps improved on it, and
  resetting memories at predicted stoppages was worse. **Rejected.**
- **Combination.** The three kept components were combined in a second round, with its grid and
  gate fixed in a configuration file before the run. The
  gains were roughly additive (0.049 + 0.009 + 0.010 against 0.069 measured). The combination was
  designed after the single-component results were seen.
- **C2: weaker coupling between team uncertainty and dead ball** (protocol committed before any
  run). One extra parameter reduces
  how much an uncertain team pushes the decoder towards D. Nested selection chose "no change" in
  all five folds. With the parameter forced on, spellF1 fell slightly under normal and long-gap
  conditions; it rose materially only with no ball at all (+0.017) and by at most 0.002 in two
  other scenarios. **Rejected.** The decoder is unchanged.

## 8. Limitations

- **Transfer is unmeasured.** The competition's definitions, league and tracking errors differ
  from the training data. The true BAR-ATH score could be lower than every figure here.
- **No independent full-label validation.** The five IDSSE matches shaped every decision. The
  SkillCorner check has no dead-ball labels, and our records do not establish that its matches
  were unseen by the team.
- **The released models are not the models of the IDSSE table** (§6.1). The released models were
  scored only by the SkillCorner silver-label check.
- **Competition-derived information in the process.** Aggregate statistics of the competition
  tracking and of the predictions, and partial results of one public-ground-truth script, were
  seen during development (§10.4, §10.5).
- **Dead-ball transfer.** The new classifier's gain is concentrated on frames where the ball is
  seen. It relies on how the ball looks during stoppages in the training data. If the
  competition tracking differs there, the classifier may miss real stoppages. The silver-label
  check cannot see this.
- **Measured regressions.** With no ball at all, and when the ball is missing around every
  switch and restart, the candidate scores below the simpler dead-ball change (§6.2). With long
  gaps the extra team features add almost nothing.
- **Short possessions.** Possessions under 1 s are almost never recovered (recall about 3%);
  1–3 s recall is about 19%.
- **Decoder costs on a plateau.** Any total dead-ball cost between 2 and 8 gives almost the same
  development score, so the chosen pair (0.25, 4.0) is not a sharp optimum.
- **Version-bound models.** The models are scikit-learn pickles bound to the pinned versions
  below. Reproduction has been verified on macOS only.

## 9. Reproduction

```bash
git clone https://github.com/Eronshu/pitchvision-ai && cd pitchvision-ai && git checkout submission-v2
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-release.txt
export MTAH_DATA_DIR=/path/to/the/competition/data
.venv/bin/python scripts/run_astra5_combo.py --config candidate/combo.json --out /tmp/pv_c
shasum -a 256 "/tmp/pv_c/PitchVision AI.csv"          # 86b73f41…8e00
```

- **Pinned stack:** Python 3.13 (verified with 3.13.2), numpy 2.5.4, pandas 3.0.6, scikit-learn
  1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0.
- **Integrity checks, all before any match data is read.** `candidate/RELEASE.lock` holds the
  sha256 of the configuration and of both model files; they are checked first, so an edited
  decoder cost or model is refused before anything is unpickled. The configuration then pins the
  Python minor version, the six library versions and the sha256 of 20 files: the run script,
  `games.py` (graded frame ranges), the dependency list and every module on the prediction path.
  The feature schema is checked once features exist. The entry loads no model other than the two
  above and reads no ground truth.
- **Limit of these checks.** The lock and configuration live in the same repository, so someone
  who edits both consistently passes them. The fixed reference is the git tag and the lock's
  sha256, `3489f36a…7c41`, which each run records in `manifest.json`. The Python patch version
  and the operating system are not pinned.
- **What was verified, and how.** Three different checks:
  - *clean source tree, fresh dependency environment, real input:* before tagging, a fresh
    `git clone` of the release commit from a local repository, with a newly created virtual
    environment installed from `requirements-release.txt`, reproduced the file above byte for
    byte and passed the official format validator (199,319 rows);
  - *tests without match data:* `PYTHONPATH=evaluation .venv/bin/python -m pytest
    tests/test_release_c.py` (24 tests: synthetic end-to-end, determinism, equality with the
    research implementation, refusal of modified models, costs, versions and source files);
  - *clone of the public GitHub remote:* done after tagging, so its result cannot be recorded in
    this document.
- **Relation to the research candidate.** Models, features and decoder costs are byte-identical to
  the research candidate evaluated in §6. The release differs only in metadata, integrity checks
  and the removal of a label-free comparison step from the run script; the output file is
  identical.
- **Fallback.** `scripts/run_ha3.py --config configs/ha3.json` still regenerates the
  `submission-v1` file (sha256 `a512018d…34d29`).
- **Reproduced on macOS only.** Training cannot be re-run from this repository: the release
  ships trained models and prediction code, not the research training pipeline.
- **Licences.** The team's code is MIT (`LICENSE-PITCHVISION-AI`). The organisers' files keep
  their licence (`LICENSE`). `NOTICE.md` lists which files are whose.

## 10. Use of AI assistance

> This section records what happened, from session and repository records. It is **not** a
> statement that these actions comply with the competition terms; that has not been confirmed
> with the organisers. Items we could not verify are marked **not verified**.

### 10.1 Tools and services

- **Claude (Anthropic), through Claude Code**, running on the project lead's machine. It wrote
  most of the code, ran the experiments on external data and drafted these documents. Whatever a
  command prints in an assistant session is sent to Anthropic's hosted service.
- **Other AI tools — not verified.** The research plan for this candidate was supplied to the
  assistant by the project lead as a written report; its origin is not recorded here. Local
  traces on the development machine indicate that an OpenAI tool (Codex / ChatGPT) was also used
  on this project, including a review dated 2026-11-14 and an audit of open SkillCorner data. The
  Claude sessions did not open those materials. What was provided to that service, and whether
  any of it derived from competition material, is not verified.

### 10.2 Who did what

- The project lead set the goals, the constraints and the stop rules, and decided which candidate
  to prepare. The assistant executed multi-step work under those written instructions, in part
  autonomously (several experiments in sequence without a check-in between them).
- The project lead reviewed the assistant's summary reports. Our records do not show a
  line-by-line human review of all code and experiment outputs.
- **Review.** No human independent review took place. On 2026-11-16 a fresh-context AI reviewer
  (the same model family, given only source, configuration, synthetic tests and external-data
  reports) reviewed code and documents; the integrity gaps it found were fixed in this release
  (§9) and its document findings were applied.
- **Publication.** At the project lead's instruction the assistant also pushed this release to
  the public repository and prepared the submission files.

### 10.3 Pre-registration: what the records support

| work | record |
|---|---|
| HA1–HA3 | design documents written before results (`docs/superpowers/specs/`) |
| HA4, HA5a | pre-registration commits that precede the result commits, in the team's private development repository (not in this repository) |
| S1, S2, S3 and the combination round | the acceptance gate was fixed in the written research plan before the experiments, and the combination grid was fixed in a configuration file before its run. Neither was committed to version control before the results existed; code and results were committed together afterwards |
| decoder variant C2 | protocol and input hashes committed before any run |
| SkillCorner check of the released models (10 matches) | protocol and match list committed before download; scored once |
| stress tests, the B-versus-C analysis, the first SkillCorner batch (3 matches) | **not pre-registered**; exploratory |

### 10.4 Competition material and the assistant

**What `submission-v1` said, and what changed.** The `submission-v1` document says that scripts
processing the competition match were run by the team and not from an assistant session. For this
candidate that is no longer true.

**Actions started from an assistant session, on the local machine, at the project lead's
instruction** (first on 2026-11-14):

1. Label-free inference on the competition tracking for the `submission-v1` version, this
   candidate and a single-change candidate, repeated for reproducibility checks up to and
   including this release, and the official format validator. From 2026-11-15 these runs
   returned only file hashes, row counts and validation status to the session.
2. A label-free comparison script that computed aggregate statistics of the predictions and of
   the tracking input.
3. One run of a script on the public five-minute ground truth (§10.5).

The assistant also works with file-system access on the machine that holds the competition data
and the organisers' repository, including the public ground-truth file. Our session records show
it reading the organisers' code and documentation; they do not show it opening the tracking,
video, Smart Tagging, homography or ground-truth files directly.

**Categories of competition-derived information that entered the hosted service** (as command
output shown in the session):

- file hashes, row counts, format-validation status, run times;
- aggregates of the predictions: label shares per half; counts and duration distributions of
  dead-ball segments, possessions and live intervals, including the longest ones and the half in
  which they fall; agreement and disagreement rates between versions, overall and split by
  whether the ball is observed;
- aggregates of the tracking input: ball-presence rate; number, median and maximum length of
  ball gaps per half; visible-player counts;
- from the public ground-truth script: the items listed in §10.5.

**Not shown in the session, according to our records:** tracking coordinates, frame-level
predictions, video, Smart Tagging or homography content, ground-truth label rows, and numeric
metric values on the competition match. No script uploaded a file. That does not mean that no
competition-derived information left the machine: the aggregates above did.

**How this information was used.** We state each use separately.

- *Model parameters and thresholds.* Both candidate packages were built before the first run on
  competition data (file timestamps 2026-11-14 00:44 and 13:57:13; first run 13:57:58), and their
  hashes have not changed since. No parameter, threshold or decoder cost was changed afterwards.
- *Design of a stress test.* The gap statistics of the competition tracking (median gap of about
  10 s, against about 2 s in our training degradation) were used to design an additional
  long-gap stress scenario on external data (§6.2 for candidate C; it is also the source of the
  "known mismatch" statement in §3).
- *Choice of candidate.* The aggregates appear in the internal reports that compared the
  `submission-v1` version and the two candidates. They were used there to screen for anomalies and to
  describe transfer risk (for example the dead-ball share of each version, and the mismatch in
  gap structure), and the long-gap scenario above is one of the results cited when the
  three-component candidate was recommended over the single-change candidate. Competition-derived
  aggregates were therefore part of the information considered when choosing which candidate to
  prepare. They were not used as accuracy estimates.
- *Labels.* The possession file is the deterministic output of the package fixed as described
  above.

### 10.5 The public five-minute ground truth

One script was run once, on 2026-11-14, from an assistant session, on the public five-minute
ground truth and the output files of three versions (`submission-v1`, this candidate and the
single-change candidate).

- **What it computed.** For each version: the official spellF1 and macroF1 against the public
  slice; spellF1 at nine time shifts between −50 and +50 frames; the number of predicted and
  true possessions. It also ran the organisers' evaluation command-line tool on each file and
  checked that it exits without error.
- **What it stored.** The metric values, the shift curves and the possession counts were written
  to a file on the local machine.
- **What it displayed, and therefore sent to the hosted service.** Facts about the public slice
  (number of graded frames, number of labelled runs, position of the slice within its half), and
  for each version: format-validation status; whether the score is a finite number in [0, 1];
  **the time shift at which spellF1 peaked**; whether the unshifted score is within 0.01 of that
  peak; whether the ratio of predicted to true possession counts lies between 0.5 and 2.
- **What this means.** The metric values were not displayed and, according to our records, the
  stored file has not been opened since by the assistant. But the metrics were computed, and the
  best time shift and the two threshold checks are feedback derived from the ground truth. We do
  not describe this as having had no contact with ground-truth feedback.
- **Use.** The displayed results were reported as a pipeline check in the internal report that
  also recommended a candidate. They were not cited there as a reason to rank the versions, and
  no time shift or other adjustment was applied to any version. We cannot demonstrate that they
  had no influence on any judgement.
- **Measures taken afterwards.** On the same day the project lead prohibited any further use of
  the competition ground truth: no reading of the public or private ground truth, of the stored
  score file or of any output of that script; no re-run of it or of the organisers' evaluation
  tool on competition data; scorer correctness is tested with synthetic labels only. Later work
  was done in checkouts that exclude the ground-truth and results directories, and integrity
  manifests exclude those files by name. The stored score file still exists on the local machine,
  unopened by the assistant since. It is not in the repository.

### 10.6 The labels

The labels in the possession file are produced entirely by the algorithm described here. No
person labelled or corrected any frame of the competition match.
