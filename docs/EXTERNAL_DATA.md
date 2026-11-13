# External data: IDSSE / DFL open dataset

PitchVision V2 uses one external dataset, for **independent validation only**: the IDSSE open
data. It has seven German Bundesliga and 2. Bundesliga matches from 2022/23, with 25 fps TRACAB
positions for every player and the ball, events, and match information. Its per-frame
`BallStatus` (alive/dead) and `BallPossession` (home/away) are used as evaluation **targets**.
They are never used as model inputs.

No competition (BAR-ATH) data, and nothing derived from it, is used together with this dataset for
fitting or calibration.

## Attribution (required by CC BY 4.0)

Use this text wherever the data, or results computed from it, appear: this repository's docs, the
technical document, and benchmark reports. It lives in code as `pvext.idsse.ATTRIBUTION`, and
`scripts/fetch_idsse.py` prints it.

> IDSSE open data: DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB), published by
> Bassek, M. et al. (2025), "An integrated dataset of spatiotemporal and event data in elite
> soccer", *Scientific Data* 12:195, doi:[10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y);
> dataset doi:[10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) (v1).
> Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: converted from DFL
> XML to arrays and to the competition's ATD text format, and synthetically degraded (ball dropout,
> tracklets, noise) for benchmarking. No endorsement by the DFL, Sportec Solutions or the authors is
> implied. Provided as-is, without warranties.

### What CC BY 4.0 asks of us, and how we meet it

| licence condition (Section 3(a)) | how it is met |
|---|---|
| identify the creator(s) and the attribution parties designated | DFL (owner), Sportec Solutions (collector), Bassek et al. (publishers) |
| retain a copyright notice, a licence notice and the no-warranty notice | the attribution text above |
| give a link (URI) to the material | dataset DOI and paper DOI |
| indicate whether the material was modified, and how | the "Changes:" sentence; every derived file is labelled as converted or degraded |
| link to the licence | https://creativecommons.org/licenses/by/4.0/ |
| no additional restrictions or technological measures on recipients (Section 2(a)(5)(B)) | we add none; we do not redistribute the files either (see Storage) |
| no implied endorsement (Section 2(a)(6)) | stated explicitly |

The paper itself says: "The present data set is released under a CC-BY 4.0 license with the
authorization from the DFL". The match videos are not part of the dataset. The competition's
rules allow external data if the organisers can access it and the technical document describes
it. This dataset is public, and its use is described here and in `docs/PITCHVISION.md`.

## Storage and integrity

- The files are downloaded with `scripts/fetch_idsse.py` into `$IDSSE_DIR`, by default
  `$IDSSE_DIR`, which is **outside the repository**. The script
  refuses any directory that resolves inside the repository, whether through a symlink, `..`, a relative path or a
  case-only difference.
- Every file is pinned in `pvext/idsse.py` by figshare URL, size and MD5, generated from the figshare
  files API on 2026-11-04. Supplied and computed MD5 agree for all 21 files. A download is written
  to a temporary file that `tempfile.mkstemp` creates exclusively (`O_CREAT|O_EXCL`, mode 0600)
  inside the verified directory, so no existing path or symlink is ever opened for writing. It is
  moved into place with `os.replace` only if both the size and the MD5 match, and a target that
  is itself a symlink is refused.
- The data is never committed. Derived arrays and reports stay outside the repository unless they
  are aggregate tables, and those carry the attribution.

## Verified facts (public sources, before any download)

- Match ids (file names and kloppy): J03WMX and J03WN1 (DFL-COM-000001), and J03WOH, J03WOY,
  J03WPY, J03WQQ and J03WR9 (DFL-COM-000002). The Hugging Face README lists J03WPF and J03WQF,
  but **those ids do not exist**.
- Teams, home–away (paper, Table 1): J03WMX Köln–Bayern; J03WN1 Bochum–Leverkusen; J03WPY, J03WOH,
  J03WQQ, J03WOY and J03WR9 all have Fortuna Düsseldorf at home, against Nürnberg, Regensburg,
  St. Pauli, Rostock and Kaiserslautern respectively. **Five of the seven matches share a home team
  and a venue**, so evaluation is grouped by match and reported in strata.
- Label semantics (paper): `BallPossession` 1 = home, 2 = away, "defined by the player in control of
  the ball". `BallStatus` 0 = inactive, 1 = active, "inactive when the match is interrupted by the
  referee". These may not match the competition's definitions, so benchmark scores compare methods
  with each other and do not predict competition scores.
- Frame numbering (kloppy): the first half starts at N = 10000 and the second at N = 100000.
- **kloppy's `sportec.load_tracking` defaults to `only_alive=True` and silently drops every
  dead-ball frame.** Any alive/dead work through kloppy must pass `only_alive=False`. Our parser
  (`pvext/dfl_parse.py`) keeps every frame, and maps a missing `BallStatus`/`BallPossession` to
  unknown (−1) instead of kloppy's dead/away defaults. On a synthetic fixture both parsers agree,
  frame by frame, on ball and player positions, state and possession
  (`test_agrees_with_kloppy_on_fixture`).

## Benchmark decisions (project lead, 2026-11-04)

1. **Primary target**: the original DFL `BallStatus`. It is used to evaluate alive/dead
   classification (D precision, recall and F1), alive/dead F1 and the accuracy of transition
   boundaries.
2. **Secondary diagnostic**: the original DFL `BallPossession`, unmodified. No smoothing or
   minimum-spell rule is applied. External spellF1 alone must not decide the final model.
3. **Positions `MetaData`**: may be absent (the absence is recorded), and is strictly validated
   whenever present. It needs a `MatchId` equal to the match information's `General MatchId`, and
   `PitchSize X/Y` must be finite, positive, and equal to `Environment PitchX/PitchY` within 0.01 m.
   The match-information dimensions themselves must also be finite and positive.

## P0 findings on J03WPY (2026-11-04)

The download was verified by size and MD5, both by `fetch_one` and by the system `md5` tool. The
parser reads the file in 7.8 s with a 228 MB peak RSS. kloppy reads it in 20.6 s and 4.15 GB
(the full cross-check, with both kloppy loads run one after the other, takes 58 s and 4.3 GB).

**Format.**
- The structure is `PutDataRequest > Positions > FrameSet > Frame`, plus one
  `Positions/MetaData[Type="pitch-size"]/PitchSize`, which the synthetic fixture originally lacked.
  The parser now validates it: it must agree with `Environment PitchX/PitchY`, 105 × 68 m, within
  0.01 m.
- Frames carry the documented `N T X Y S` and, on the ball, `Z BallStatus BallPossession`. They
  also carry `D`, `A` and `M`, which are ignored and never used as features.
- There are 55 FrameSets, all of them the ball or listed players: 1 ball + 22 players in the first
  half, 1 ball + 31 in the second (substitutes). There are no referee or staff FrameSets. In the
  match information the `guest` team is listed first.

**Timeline and frame rate.**
- The first half runs N 10000–79130 (69,131 frames), the second N 100000–177079 (77,080 frames).
- The ball stream is contiguous in both halves, and no frame is dropped.
- Timestamps give exactly **25.000 fps**, and the frame counts agree with
  `TotalTimeFirstHalf/SecondHalf` (2,765,200 / 3,083,160 ms).
- Each half starts at the kickoff: the ball is already moving at 12–15 m/s and the status is
  alive. **There is no pre-kickoff dead period**, unlike BAR-ATH's second half.

**Physical units** (also in code as `pvext.dfl_parse.UNITS`):

| attribute | unit | status | used |
|---|---|---|---|
| `N` | frame number, 25 fps | measured from `T` | yes |
| `T` | ISO 8601 timestamp | documented | yes (frame rate, `t_range`) |
| `X`, `Y` | m, origin at the centre spot, x along the length, y bottom to top | documented; origin and axes measured, and confirmed by kloppy | yes |
| `Z` | m (ball height) | documented | only for degradation (P1), never as a labeller input |
| `S` | **km/h** | documented (Bassek et al.); **confirmed on J03WPY**: S / position-derived speed has a median of 3.57 (players) and 3.75 (ball) | parsed as `players_speed_kmh` |
| `D` | about cm per frame | inferred (about 100 × the per-frame displacement) | no |
| `A` | about m/s² | inferred (±8 for players, 0 for the ball) | no |
| `M` | match minute | inferred (1–47 per half) | no |

V1 and the competition ATD use **m/s** for speed. Conversion is explicit, through
`dfl_parse.kmh_to_mps()`, and the parser exposes no unit-less speed field. Speeds in the P0 report
(`ball_speed_median_dead_alive_mps`, kickoff ball speed) are derived from positions in m/s; they do
not come from `S`.

**Coordinates.**
- Metres, with the **origin at the centre spot** and x along the length. The ball's x stays within
  [-54.5, 53.7] and its y within [-37, 37]. Only 0.3–0.7 % of observations fall outside a centred
  pitch plus 1 m, against 72–77 % outside a corner-origin pitch.
- kloppy reports `SportecTrackingDataCoordinateSystem`, origin `center`, y `BOTTOM_TO_TOP`.
  Conversion to the ATD corner convention is therefore `x + L/2`, `y + W/2`.

**Completeness.**
- All 22 players are present on every frame, and the ball on 100 % of frames, including dead and
  out-of-play frames. The data is far cleaner than the competition ATD (about 59 % ball presence),
  **so degradation is essential** for any comparison.

**Labels.**
- `BallStatus` and `BallPossession` have 0 missing and 0 unexpected values. Possession is always
  set, including on dead frames, so the target maps dead → `D` regardless of it.
- Dead share is 33 % in the first half and 54 % in the second, 44 % overall. That fits
  league-level effective playing time, as a qualitative check only.
- There are 38 and 59 dead runs, with a median of about 22.5 s and a 10th percentile of 5–6 s.
- The median ball speed is 0.0 m/s on dead frames and 6.3–6.6 m/s on alive ones. Only 12–14 % of
  dead frames have the ball outside the pitch, which fits "interrupted by the referee".

**Possession definition gap: an open uncertainty, not a conclusion.**
- Alive possession spells, split by dead runs, number 204 + 217, with a mean of 9.1 s and 6.5 s and
  a 10th percentile of 0.7–1.0 s. Possession switches 5.4–6.7 times per alive minute.
- The competition's public `DATA.md` documents 17 possessions in its first 5 minutes, with a mean
  of 14.2 s.
- The two sources differ in league, match, sample length and possibly labelling procedure, so
  these figures **do not show why** the DFL spells are shorter. They are consistent with a
  different possession definition, but they do not establish one. In particular they do not show
  that DFL changes possession on every touch.
- Consequence: possession-spell scores against DFL targets are a secondary diagnostic that
  compares methods with each other. They neither predict competition scores nor decide the final
  model on their own.

**Independent cross-check.** kloppy (`only_alive=False`) and `pvext.dfl_parse` agree on every
frame (146,211) and every player observation (3,216,642), with 0 status and 0 possession
mismatches. The maximum position difference is 2e-6 m, which is float32 representation error
(we store float32, kloppy float64), well inside the 1e-4 m verdict tolerance. kloppy's default
`only_alive=True` keeps 81,460 frames and **drops 64,751 dead frames (44 %)**, exactly the dead
count from `BallStatus`.

Reproduce it (needs kloppy and about 4.3 GB of memory; about 1 minute; the report holds
aggregates only):

```bash
.venv/bin/python scripts/crosscheck_kloppy_idsse.py --match J03WPY --out $IDSSE_DIR/J03WPY_kloppy_crosscheck.json
```

```bash
.venv/bin/python scripts/inspect_idsse.py --match J03WPY --out $IDSSE_DIR/J03WPY_probe.json
```

The verdict is `AGREE` only if every frame and every player observation is matched, there are
no status or possession mismatches, there are **no unknown (−1) labels** on our side (kloppy's
defaults, missing status → dead and missing possession → away, could otherwise line up with them
by accident), there are **no non-finite coordinates on either side**, and the position differences
are within 1e-4 m. The scripts exit non-zero on any disagreement. The downloaded files and both reports stay in the
external directory and are never committed.

**Remaining uncertainty.**
- Whether DFL's dead-ball boundaries (start at the referee's interruption, end at the restart)
  align with the competition's (issue #31).
- How to treat DFL's touch-level possession spells (raw, or a rule fixed in advance before any
  method is scored).
- Whether the other six files share this exact structure; each is checked by the strict parser
  when it is first read.
- The five Düsseldorf home matches are not independent samples.

## Exploratory benchmark on J03WPY (T6, 2026-11-04)

> **One match: exploratory diagnostics only.** These figures do not estimate competition
> performance and must not be used to select or tune parameters.

Frozen V1 (`configs/v1.json`) and the official fallback (`smooth(…, 50)`) read the same converted
ATD file in every run. The degraded presets use seeds 0, 1 and 2, and the table gives the mean.
Targets are DFL `BallStatus` and `BallPossession`; the true dead share is 44.3 %.

| metric | clean V1 / FB | mild V1 / FB | doc V1 / FB | severe V1 / FB |
|---|---|---|---|---|
| alive/dead F1 | 0.891 / 0.716 | 0.780 / 0.715 | 0.820 / 0.698 | 0.781 / 0.668 |
| D precision | 0.999 / 0.000 | 0.988 / 0.355 | 0.942 / 0.398 | 0.911 / 0.414 |
| **D recall** | **0.694** / 0.000 | **0.296** / 0.002 | **0.493** / 0.048 | **0.350** / 0.138 |
| D F1 | 0.819 / 0.000 | 0.455 / 0.003 | 0.647 / 0.085 | 0.505 / 0.207 |
| predicted D share | 0.308 / 0.000 | 0.133 / 0.002 | 0.232 / 0.053 | 0.170 / 0.147 |
| dead **start** F1 ±1 s | 0.200 / 0.000 | 0.301 / 0.006 | 0.175 / 0.074 | 0.123 / 0.052 |
| dead **end** F1 ±1 s | 0.843 / 0.000 | 0.450 / 0.012 | 0.447 / 0.048 | 0.153 / 0.058 |
| possession spellF1 (secondary) | 0.447 / 0.317 | 0.365 / 0.276 | 0.306 / 0.194 | 0.217 / 0.142 |
| predicted spells (true 421) | 415 / 375 | 340 / 456 | 290 / 693 | 292 / 819 |

V1's D F1 range across seeds: mild 0.418–0.477, doc 0.614–0.667, severe 0.496–0.510.

Reproduce it, with outputs outside the repository (about 2 minutes, 1.3 GB):

```bash
.venv/bin/python scripts/run_benchmark.py --match J03WPY --presets clean,mild,doc,severe --seeds 0,1,2
```

## Phase A: frozen cross-match benchmark (2026-11-05)

J03WMX (Köln–Bayern) and J03WN1 (Bochum–Leverkusen) are 1. Bundesliga matches that share no team,
venue or competition with J03WPY. Only their match information and positions were downloaded.
Each passed G1–G6:
- G1: MD5 matches;
- G2: the strict parser reads the file;
- G3: kloppy reports AGREE, with 0 mismatches, unknown labels or non-finite values;
- G4: 25 fps, centred coordinates, no missing labels;
- G5: two `clean` runs give identical prediction hashes;
- G6: the V1 config hash, presets, seeds and metrics are identical to the J03WPY run.

A J03WPY `clean` re-run at the current commit reproduces its original prediction hashes.

Data notes:
- J03WMX and J03WN1 include referee FrameSets (3 per half). They are skipped as designed.
- J03WN1 is 11 v 10 from about minute 7 (Leverkusen), and its dead share is 51.7 %.
- J03WN1's second half has one stale first frame: the ball is 29 m from the centre with status
  dead, and at the centre spot and alive from the next frame. It affects 1 of 141,561 frames.

**Frozen V1, mean over seeds, per match (J03WPY / J03WMX / J03WN1). DFL labels: not an estimate of
competition performance.**

| metric | clean | mild | doc | severe |
|---|---|---|---|---|
| D recall | 0.69 / 0.69 / 0.72 | 0.30 / 0.34 / 0.30 | 0.49 / 0.54 / 0.52 | 0.35 / 0.42 / 0.37 |
| D precision | 1.00 / 1.00 / 1.00 | 0.99 / 0.99 / 0.98 | 0.94 / 0.96 / 0.95 | 0.91 / 0.90 / 0.91 |
| predicted / true D share | 0.31/0.44 · 0.29/0.42 · 0.37/0.52 | 0.13 · 0.14 · 0.16 | 0.23 · 0.23 · 0.28 | 0.17 · 0.19 · 0.21 |
| dead **start** F1 ±1 s | 0.20 / 0.24 / 0.28 | 0.30 / 0.26 / 0.32 | 0.18 / 0.22 / 0.22 | 0.12 / 0.14 / 0.12 |
| dead **end** F1 ±1 s | 0.84 / 0.85 / 0.89 | 0.45 / 0.46 / 0.44 | 0.45 / 0.46 / 0.46 | 0.15 / 0.19 / 0.16 |
| predicted / true stoppage starts | 73/97 · 77/103 · 86/106 | 36 · 37 · 36 | 70 · 69 · 76 | 119 · 111 · 127 |
| alive/dead F1 | 0.89 / 0.90 / 0.87 | 0.78 / 0.81 / 0.73 | 0.82 / 0.85 / 0.78 | 0.78 / 0.81 / 0.73 |
| spellF1 (secondary) | 0.45 / 0.45 / 0.46 | 0.37 / 0.40 / 0.36 | 0.31 / 0.33 / 0.27 | 0.22 / 0.21 / 0.18 |

The seed ranges of V1's D recall under mild (0.27–0.36) and doc (0.46–0.56) do not overlap in any
match. V1 beats the fallback on D F1 and alive/dead F1 in all 12 match × preset cells.

## All seven matches: frozen benchmark, per match by league (2026-11-05)

J03WOH, J03WOY, J03WQQ and J03WR9 (metadata and positions only) passed G1–G6 with the same
scripts, presets, seeds 0, 1 and 2, V1 config hash and scoring. Every kloppy cross-check is AGREE.
**These are DFL labels, not an estimate of competition performance. Nothing was tuned on them.**

Anomalies, preserved and not corrected:
- J03WN1: 11 v 10 from about minute 7. The second half has one stale first frame (dead, ball 29 m
  from the centre).
- J03WOH: the second half opens with 48 dead frames (about 1.9 s) in which the ball sits 51 m from
  the centre, then jumps to the centre spot in one frame. This looks like a tracking artifact
  during pre-kick-off. The labels are consistent (dead).
- J03WQQ: 11 v 10 from minute 13.4 of the second half.
- J03WMX and J03WN1 carry referee FrameSets (skipped as designed); the Düsseldorf files do not.

**Frozen V1 D recall / D F1 / stoppage-start F1 ±1 s (mean over seeds)**

| league | match | clean | mild | doc | severe |
|---|---|---|---|---|---|
| Bundesliga | J03WMX | 0.69 / 0.81 / 0.24 | 0.34 / 0.51 / 0.26 | 0.54 / 0.69 / 0.22 | 0.42 / 0.57 / 0.14 |
| Bundesliga | J03WN1 | 0.72 / 0.83 / 0.28 | 0.30 / 0.46 / 0.32 | 0.52 / 0.67 / 0.22 | 0.37 / 0.52 / 0.12 |
| 2. Bundesliga | J03WPY | 0.69 / 0.82 / 0.20 | 0.30 / 0.45 / 0.30 | 0.49 / 0.65 / 0.17 | 0.35 / 0.51 / 0.12 |
| 2. Bundesliga | J03WOH | 0.66 / 0.79 / 0.16 | 0.27 / 0.43 / 0.25 | 0.44 / 0.60 / 0.15 | 0.34 / 0.50 / 0.10 |
| 2. Bundesliga | J03WOY | 0.66 / 0.79 / 0.26 | 0.22 / 0.36 / 0.27 | 0.44 / 0.60 / 0.21 | 0.35 / 0.50 / 0.14 |
| 2. Bundesliga | J03WQQ | 0.61 / 0.76 / 0.14 | 0.13 / 0.22 / 0.12 | 0.29 / 0.44 / 0.12 | 0.26 / 0.40 / 0.07 |
| 2. Bundesliga | J03WR9 | 0.63 / 0.77 / 0.22 | 0.34 / 0.51 / 0.25 | 0.49 / 0.65 / 0.22 | 0.38 / 0.54 / 0.13 |

V1's D precision is 0.88–1.00 in every cell. Its stoppage-end F1 on clean is 0.81–0.89. It beats the
fallback on D F1 and alive/dead F1 in all 28 match × preset cells (fallback D F1 under doc is
0.08–0.10).

**Confirmed observations (7 of 7 matches, both leagues):** low D recall with high precision; weak
stoppage starts (start F1 well below end F1); mild below doc on D recall, with disjoint seed ranges in
every match; V1 ahead of the fallback.
**Match-specific:** J03WQQ is the weakest match for V1 (mild D recall 0.13, doc 0.29).
**Unverified:** why mild underperforms doc, and whether noise suppresses the static-ball cue.

