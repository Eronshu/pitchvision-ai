# PitchVision AI — technical document

More than a Hack 2027, Phase 1: FC Barcelona vs Athletic Club (BAR-ATH).

- **Team:** PitchVision AI
- **Possession file:** `PitchVision AI.csv`: 199,319 rows, `frame,label`, labels H/A/D/X.
  sha256 `a512018d99108e8e97b8a1d53d66bebd4fd6c4c86f7478170335721e81d34d29`.
- **Repository:** https://github.com/Eronshu/pitchvision-ai, tag `submission-v1`.

> **Status of the evidence.** Every accuracy number in this document comes from external open
> data (IDSSE, Bundesliga), whose possession and dead-ball definitions differ from the
> competition's. **We have not measured accuracy on BAR-ATH.** The public five-minute ground truth
> was used only to check that the pipeline is wired correctly, never to tune, compare or choose.

## 1. Approach

The system is an offline labeller. It reads only the provided tactical tracking (ATD) and
labels every frame of the tactical video in three stages:

```
ATD ─► 59 dead-ball features ─► logistic regression ─► persistence ─► dead mask ─► D
  │                                                                       │
  │                         V1 possession machine (frames outside play) ◄─┤ ─► X
  └► 26 team features ─► symmetric gradient boosting P(home) ─► Viterbi per live stretch ─► H / A
```

1. **Graded halves.** Frames outside the two graded halves (video frames 11,185–90,651 and
   110,808–190,197) are `X`. Each half is labelled independently.
2. **Dead ball (D).** A logistic regression over 59 ATD features gives a per-frame probability
   that the ball is dead. The probability then goes through fixed persistence rules:
   - a 1 s centred mean, then a threshold of 0.4;
   - dead runs shorter than 2 s are dropped;
   - live gaps shorter than 1 s are absorbed.
3. **Team in possession (H/A)** on the remaining live frames:
   - a gradient-boosted classifier gives P(home has the ball) per frame;
   - a two-state Viterbi decoder smooths it over each live stretch.

### Why this design

Our first version (V1) was a hand-written rule system with no learning. We kept it as a fallback.
Testing V1 on seven fully labelled open matches located its two weaknesses:

- **It found too few stoppages.** Dead-ball recall was 0.29–0.54 at about 0.95 precision. We
  replaced its dead-ball detector with a learned one ("M1").
- **Team assignment caused most of the remaining spellF1 loss.** Oracle experiments showed that
  most of what was left came from the team assignment, not from the alive/dead boundary: flicker
  between teams, and late or spurious changes of possession. We replaced the H/A logic with a
  learned per-frame probability plus a decoder that charges a cost for every change ("HA3").

We chose sequence decoding over per-frame decisions because spellF1 penalises spurious short
possessions: a flicker splits one possession into three. A Viterbi decoder with a fixed switch
cost λ is the simplest model that trades per-frame evidence against the number of possession
changes, and it has a single parameter.

## 2. Possession rules as implemented

- **Who has the ball.** It is decided from evidence over time, not frame by frame. Each live frame
  carries a probability that the home team has the ball. The decoder picks the H/A sequence that
  maximises the summed log-probability minus λ = 0.5 per change of team.
  - A change of possession is therefore placed only where the evidence for the other team
    outweighs the cost of switching.
  - It is placed at the frame where the decoded evidence changes sides.
- **Loose balls, deflections and challenges.** There is no explicit rule. These situations are
  handled by the cost of a switch:
  - a brief touch by the other team, a deflection or a contested ball rarely carries enough
    evidence to pay for two changes, so possession stays with the previous team;
  - sustained control by the other team does pay for the change.
- **Missing ball.** The last observed ball position is carried for up to 1 s. After that,
  ball-relative features become neutral (0), and the decoder keeps the current team unless other
  evidence, such as team shape or the last touch, points the other way.
- **Restarts.** A dead ball has no owner (`D`). Each live stretch is decoded on its own, with no
  preference for either team at its start, so the team after a restart is whichever team the
  evidence in that stretch supports from its first frame. Possession is never carried across a
  stoppage.
- **Dead ball.** The ball is out of play or play is stopped. The model was trained on DFL
  `BallStatus`. Its boundaries are where the smoothed probability crosses the threshold.
- **No home bias.** The team features are antisymmetric (home minus away, or signed by the
  attacking direction). The classifier is trained on each example and its team-swapped mirror,
  and it predicts p = ½(g(x) + 1 − g(−x)). Swapping the two teams therefore swaps P(home) exactly.

## 3. Features and handling of missing and noisy tracking

**Dead-ball features.** There are 59 per-frame, slot-invariant features, computed from the ATD
only:
- ball presence and gap length;
- ball speed and acceleration derived from positions;
- out-of-play and line, box and corner distances;
- field motion, as speed quantiles and the share of players moving;
- team shape and kick-off formation;
- V1's own dead-ball score;
- ±1, 3 and 10 s window statistics.

**Team features.** There are 26, all home-minus-away:
- nearest and second-nearest player distance to the ball;
- player counts within 3, 5 and 10 m;
- relative speed;
- ball position and velocity in the home attacking direction;
- the last team close to the ball;
- window means over 0.5, 1 and 3 s, and 1 s past and future means;
- 5 **touch** features: a touch is a ball velocity change of at least 3 m/s over ±3 frames with
  the nearest player within 2 m. The features are the last and next touching team, with 2 s
  decays, and a 3 s touch balance.

The labeller is offline, so features may look up to 3 s ahead. The touch windows are fixed in
frames, so the pipeline accepts only 25 fps.

**Missing and noisy data.**
- The ATD locates the ball on only about 59% of graded frames and splits players into short
  tracklets. All features are therefore defined on tracking slots (no player identities) and
  treat a missing value as neutral.
- Every model was trained on open data that we **degraded synthetically** to look like the
  competition ATD. The degradation is seeded and independent of the labels:
  - player occlusion gaps, position noise, tracklets cut into slots;
  - bursty ball dropout, more likely when the ball is high or crowded;
  - false ball detections.
- We tested three severities, `mild`, `doc` (matched to the documented ATD statistics) and
  `severe`. Results are reported for all three.

## 4. Assets used, and assets dropped

| asset | use |
|---|---|
| tactical ATD (players, ball, slots) | **used**, the only input at prediction time |
| IDSSE open data, 7 matches of other games (DFL/Sportec, CC BY 4.0): positions with `BallStatus`/`BallPossession` | **used** for training and validation only (§5), converted to the ATD format and degraded (§3) |
| IDSSE DFL event data of the same 7 matches | used **only in the rejected HA4 experiment** (§7), never in the release |
| public 5-minute ground truth | pipeline-correctness check only (frames aligned, file well-formed). Never used for training, tuning or choosing |
| Smart Tagging (SET PIECES, phases) | available in V1 as an optional dead-ball cue and restarting-team seed; **dropped**: off by default, not used by the release. HA4 (§7) tested set-piece tags on IDSSE and was rejected |
| homography, tactical video, broadcast video and ATD | **not used** |
| V1's rule-based H/A machine | **replaced** by HA3 on live frames; kept in the V1/M1 fallbacks |
| HA1 (linear H/A) | superseded by HA2 (symmetric gradient boosting) |
| HA1b | **rejected** on the development matches |
| HA2 | superseded by HA3 (HA2 + touch features) |
| HA4 (set-piece tags + joint retune) | **rejected** by its pre-registered rule (§7) |
| HA5a (reliability-aware H/A features) | development experiment after the release was frozen (§7); not part of the release |

## 5. Training data

**No labels of any kind on the provided game were used, by the team or anyone else.** The full-match
ground truth was never available to us. The 5 public minutes were used only to check the pipeline.
We did not hand-label any data.

The only labelled data we used is the IDSSE open dataset, which covers seven other professional
matches.

| dataset | source and licence | labels used | who produced the labels | quantity |
|---|---|---|---|---|
| IDSSE positional data | DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB). Bassek et al. (2025), Scientific Data 12:195; dataset doi:10.6084/m9.figshare.28196177 (v1). CC BY 4.0 | per-frame `BallStatus` (dead/alive) and `BallPossession` (home/away). They were used as training and evaluation targets, never as inputs | Delivered with the official DFL positional data. Positions come from TRACAB's video-based tracking. The paper defines both fields but does not say whether they are produced automatically or by people. We did not relabel or correct them | 7 matches (Bundesliga J03WMX, J03WN1; 2. Bundesliga J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9); 1,002,644 graded frames at 25 fps, of which 715,116 are from the 5 development matches |
| IDSSE event data (same matches) | same source and licence | restart events (kick-off, throw-in, corner, goal kick, free kick) with time and team | Per the paper, collected by trained human operators, with post-match quality control | 7 matches. Used only in the rejected HA4 experiment, to simulate set-piece tags |

**Training data of each released model.**

Every model was trained on synthetically degraded copies: presets mild, doc and severe, with
seeds 100 and 101, at a 5-frame stride.

| model | file | trained on | rows |
|---|---|---|---|
| H/A classifier (HA3) | `models/ha3_final.joblib` | 5 development matches: J03WPY, J03WOH, J03WOY, J03WQQ, J03WR9 | 300,000 rows sampled (seed 0) from their H/A frames, plus team-mirrored copies |
| dead-ball model (used with HA3) | `models/ha3_dead_m1_league.joblib` | the same 5 development matches | 858,162 rows |
| M1 dead-ball model (fallback only) | `models/m1_final.joblib` | all 7 matches | 1,203,204 rows |

The first two are the exact models evaluated on the held-out Bundesliga matches. Nothing from the
competition match was used to train, tune or select anything.

**External attribution.** IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec
Solutions (TRACAB). Published by Bassek, M. et al. (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12:195,
doi:10.1038/s41597-025-04505-y; dataset doi:10.6084/m9.figshare.28196177 (v1). Licensed CC BY
4.0.

We converted it to the competition's ATD format and degraded it synthetically. No endorsement is
implied. The raw data is not redistributed; the repository contains only trained model
parameters.

## 6. Validation (IDSSE, not BAR-ATH)

**Protocol.**
- **Development:** five matches, used for leave-one-match-out comparison and every design
  choice.
- **Held out:** two matches (J03WMX, J03WN1). They were evaluated once, with the models above.
- **Metric:** the official spellF1, on the `doc` preset, mean of 3 evaluation seeds.

| | J03WMX (held out) | J03WN1 (held out) | development, mean of 5 |
|---|---|---|---|
| V1 (rules) | 0.325 | 0.274 | 0.326 |
| M1 (learned dead ball + V1 H/A) | 0.359 | 0.321 | 0.381 |
| **HA3 (release)** | **0.419** | **0.401** | **0.460** |

- HA3 scored above M1 in all 21 match × severity cells, by +0.06 to +0.13.
- On the held-out matches HA3 also raised:
  - team accuracy, from 0.84 to 0.87 and from 0.78 to 0.82;
  - macroF1, from 0.83 to 0.85 and from 0.78 to 0.81.

**What failed.** We pre-registered four acceptance criteria. HA3 passed three:
- gains on both held-out matches;
- gains on at least 4 of 5 development matches;
- gains under mild and severe degradation.

It failed the fourth: the number of predicted possessions should be within 0.7–1.5× of the true
number.
- HA3 predicts too few possessions: 0.56× and 0.59× of the true count on the held-out matches.
- M1 has almost the same shortfall (0.59× on both).
- An earlier draft attributed the shortfall to the dead-ball segmentation. A later measurement
  does not support that; see §8.

A post-hoc relaxation also failed by a small margin on J03WMX. We release HA3 by an explicit
decision of the project lead, because spellF1 is the competition metric and HA3 improved it on
every match tested. The failure is reported here rather than hidden.

## 7. Development history: what was tried and dropped

Every step below was pre-registered before its results were seen. Every step used only IDSSE data,
and each was judged against the model before it.

- **HA1 → HA2 → HA3.**
  - HA1: a linear H/A model.
  - HA1b: a lower λ grid, rejected.
  - HA2: symmetric gradient boosting.
  - HA3: HA2 plus touch features. HA3 is the release (§6).
- **HA4: set-piece tags and a joint retune. Rejected.** It was tested after HA3 was frozen, on the
  5 development matches only. Its three parts:
  - A1 moved the end of a predicted stoppage to a set-piece tag.
  - A2 gave the tagged team a prior at a restart.
  - A4 retuned the dead threshold, the minimum stoppage length and λ jointly.

  The competition's Smart Tagging has no equivalent in IDSSE, so tags were simulated from the DFL
  restart events, with realistic misses, timing noise and no team on throw-ins.

  Results, by nested leave-one-match-out:
  - No fold chose to use the tags.
  - The retune alone gained only +0.004 in mean doc spellF1, below the pre-registered +0.010.
  - Even with perfect (noise-free) tags, A1 lowered spellF1 by about 0.008 and A2 by about
    0.002. The DFL event time and the labelled restart differ by about ±1–2 s, and HA3 already
    finds the restarting team.

  HA4 was not adopted, and Smart Tagging remains unused.
- **HA5a: reliability-aware H/A features.** A single pre-registered challenger, evaluated after
  the release was frozen. It is not part of the release.

  - **What changed.** HA5a adds 12 features to HA3's 26:
    - team-invariant observation-reliability features: is the ball seen, how long ago it was last
      seen, and how recently a close-control event or a touch happened;
    - short-horizon control features computed from the fresh ball only, with no reliance on
      tracking-slot identities.

    Everything else was kept as in HA3: the dead-ball model, the decoder and λ, the model family
    and the training data.
  - **Result**, against its pre-registered criteria, by match-held-out evaluation on the 5
    development matches:
    - mean doc spellF1 +0.010, with 4 of 5 matches improved (−0.004 to +0.025);
    - mild +0.014 and severe +0.009, each improving on all 5 matches;
    - macroF1 +0.004;
    - direct-switch precision +0.016;
    - the gain is mostly on possessions of 1–10 s. Possessions under 1 s and the possession-count
      deficit are essentially unchanged.
  - **Status.** It passed its criteria only narrowly. The design was informed by analyses of these
    same matches, so the estimate is optimistic. The only other external matches had already been
    seen, so no independent held-out confirmation was possible.
  - **Decision.** HA5a is therefore **not** part of this release, and its code is not included
    here. The submitted file is HA3.

## 8. Limitations, and what the development data shows

- **Transfer is unmeasured.** The competition's possession and dead-ball definitions, its league
  and its tracking errors all differ from our training data. The synthetic degradation is an
  approximation, and the true BAR-ATH score could be lower than the IDSSE figures.
- **Too few possessions: where the deficit really is.** These are observations on the IDSSE
  development matches, from read-only analyses of frozen held-out predictions. They are **not**
  competition results.
  - The official possession count splits into live-ball intervals plus direct H/A switches inside
    them.
  - Per match, the truth has about 100 intervals and 310 switches. HA3 predicts about 111
    intervals and about 189 switches. The deficit is in team switches, not in the dead-ball
    segmentation, as an earlier draft wrongly claimed.
  - Possessions shorter than 3 s are about 40% of true possessions. They are rarely recovered:
    mean recall is about 2% under 1 s, 15% for 1–3 s, 44% for 3–10 s and 82% from 10 s.
  - Short-possession misses are associated with an unobserved ball:
    - recall drops to about 4–5% when the ball is seen in less than a quarter of the possession,
      or was last seen over 1 s before it starts;
    - ball-dependent touch and control-memory features update late around missed switches;
    - the classifier's evidence there is near 0.5, so it is uncertain rather than confidently
      wrong.
  - The Viterbi decoder suppresses only about 4% of the missed short possessions. Neither the λ
    choice nor the minimum stoppage lengths explain the deficit.
  - These are associations, not proven causes.
- **Ball-tracking dependence.** The touch features need a usable raw ball trajectory. Under
  severe degradation their gain over HA2 disappears.
- **Small held-out set.** Only 2 matches were held out, and they had been partly seen in earlier
  aggregate analyses. The development results carry selection bias.
- **Version-bound models.** The models are scikit-learn pickles bound to the pinned versions
  below, and the loaders refuse any other version. Reproduction has been verified on macOS only.
- **With more time** we would:
  - test ball-observation-aware evidence and boundary detection for short possessions on more
    external matches;
  - check whether the competition's ground truth contains as many sub-second possessions as
    IDSSE.

## 9. Reproduction

The command below regenerates the exact submitted file from the provided data bundle. The models
ship in the repository, so no training data is needed.

```bash
git clone https://github.com/Eronshu/pitchvision-ai && cd pitchvision-ai && git checkout submission-v1
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-ha3.txt
export MTAH_DATA_DIR=/path/to/the/competition/data   # or unpack it into ./data
.venv/bin/python scripts/run_ha3.py --config configs/ha3.json --out /tmp/pv_ha3
shasum -a 256 "/tmp/pv_ha3/PitchVision AI.csv"        # a512018d…34d29
```

- **Pinned stack** (`requirements-v2.txt`, loaded by `requirements-ha3.txt`): Python 3.13.2,
  numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl
  3.8.0.
- **Checks before any data is read.** The run refuses to start on any of these mismatches:
  - a model or config hash (each model's sha256 is checked before it is unpickled);
  - the frozen prediction-code hashes;
  - the installed versions;
  - the feature schema.
- **Run cost.** About 9 s and 1.7 GB peak memory on a laptop.
- **Determinism.** Two runs and a run from a fresh clone gave byte-identical files.
- **Full check.** `bash scripts/local_release_e2e.sh <out>` runs every check: generation twice,
  a clean-clone regeneration, byte comparison, the official validator at 199,319 rows, and the
  release verifiers.
- **Licences.** The team's code is MIT (`LICENSE-PITCHVISION-AI`). The organisers' files keep
  their licence (`LICENSE`). `NOTICE.md` lists which files are whose.
- **Other documentation.**
  - Full method: `docs/HA3_TECHNICAL_DOCUMENT.md`.
  - Release and fallback: `docs/HA3_RELEASE.md`.
  - Research reproduction on IDSSE: `docs/HA3_RELEASE.md` §6.

## 10. Use of AI assistance

- **What the assistant did.** The code, the experiments on the external IDSSE data and these
  documents were developed with the help of an AI coding assistant (Claude, by Anthropic, used
  through Claude Code). Team members directed the work, reviewed it and made every decision.
- **Pre-registration.** Protocols and acceptance rules were written down before the results they
  governed were seen. The pre-registered designs are in `docs/superpowers/specs/`.
- **Competition data policy.** The team's rule, written into the development plans from the first
  version onward, was that no competition material would be given to a hosted AI service.
  Competition material means the videos, tracking, Smart Tagging, homography, ground truth and any
  extracts of them.
  - Scripts that process the competition match are written to run locally and to print summaries.
    Their headers say they are not to be run from an assistant session.
  - The team ran those scripts locally on its own machine.
  - What our development notes record being relayed to the assistant from those runs is summary
    output: check results, file hashes, run times, memory use and aggregate diagnostic statistics.
    The notes record no raw data, frame-level labels or data extracts being relayed.
  - We have not audited every assistant interaction exhaustively. This statement describes our
    policy and our records. It is not a guarantee beyond them.
- **The labels.** The submitted labels are produced entirely by the algorithm described here. No
  person labelled or corrected any frame of the competition match.
