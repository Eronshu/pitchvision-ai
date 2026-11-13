# PitchVision V2 — P0–P2: external benchmark design

Date: 2026-11-04 · Branch: `pv/v2` (from frozen V1 `a8ee2b7`) · Scope: P0–P2 only. No learned
classifier and no HSMM yet.

## Goal

A reproducible benchmark that scores V1 and the official fallback, **under identical conditions**,
against an independently licensed reference for alive/dead and possession. That reference is the
IDSSE / DFL dataset, degraded to look like automatic tracking. It is the evaluation V2 will later
be judged on, built and validated before any V2 model exists.

## Constraints

- V1 (`a8ee2b7`, Draft PR #1) is frozen. `pitchvision/` V1 modules are imported, never edited.
  Official files are untouched.
- Inputs: public documentation, synthetic fixtures, and IDSSE files only. **No BAR-ATH data, and no
  data derived from it, is used for fitting or calibration**, including aggregate statistics. Degradation
  parameters come from public documentation (`DATA.md`) and from severity sweeps.
- `BallStatus` and `BallPossession` are **targets only**. They are never part of the tracking that
  any labeller reads.
- IDSSE files stay outside the repository and are never committed. Downloads are pinned by URL and
  MD5.
- The public 5-minute GT plays no part. Benchmark numbers are not predictions of BAR-ATH scores.

## External data (verified from public sources, 2026-11-04)

- Figshare 10.6084/m9.figshare.28196177 v1, licence CC BY 4.0. The paper states it is "released
  under a CC-BY 4.0 license with the authorization from the DFL". 21 files: matchinformation
  (~12 KB), events (~0.6–0.8 MB) and positions (349–419 MB) for each match. The Hugging Face mirror
  `pysport/idsse-data` has byte-identical sizes.
- Match IDs, from the actual file names and kloppy: **J03WMX, J03WN1** (DFL-COM-000001) and
  **J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9** (DFL-COM-000002). The Hugging Face README's
  J03WPF and J03WQF do not exist.
- Teams (paper, Table 1): J03WMX Köln–Bayern; J03WN1 Bochum–Leverkusen. **Fortuna Düsseldorf is
  the home team in all five 2. Bundesliga matches** (opponents: Nürnberg, Regensburg, St. Pauli,
  Rostock, Kaiserslautern).
- Field semantics (paper): the ball's `BallPossession` is 1 = home, 2 = away, "defined by the player
  in control of the ball". `BallStatus` is 0 = inactive, 1 = active, "inactive when the match is
  interrupted by the referee". Positions are x/y in metres, one FrameSet per person or ball per game
  section, Frame attribute `N` is the frame number. The paper does not state the origin, axes or
  frame rate, so P0 measures them. Known issues: occlusion gaps in dense situations, identity swaps
  at close range, and post-processing smoothing.
- kloppy quirk: a missing `BallPossession` maps to away and a missing `BallStatus` maps to dead.
  Our parser treats missing as **unknown** instead.

## Components

```
pvext/                         V2 external-data package (no V1 module edited)
  idsse.py        manifest of the 7 matches (ids, competition, urls, md5, teams)
  dfl_parse.py    streaming DFL XML -> Tracking (no labels) + Labels (targets), separately
  convert.py      DFL arrays -> V1 contracts: pitch transform, blocks, ATD-format writer
  degrade.py      seeded degradation: ball dropout, tracklets/slots, noise, false detections
  benchmark.py    per-match run of V1 and fallback on identical ATD files; official metrics
scripts/fetch_idsse.py         download + md5 verify into an external directory
scripts/run_benchmark.py       benchmark CLI -> per-match JSON/CSV report
docs/EXTERNAL_DATA.md          licence, attribution, changes made, storage rules
tests/ext/                     synthetic DFL XML fixtures + unit/integration tests
```

### Data contracts

- `dfl_parse.parse(meta_xml, positions_xml) -> (Tracking-like DFLTrack, Labels, Meta)`.
  - `DFLTrack`: per game section, a frame index `N` (sorted, contiguous range checked), player
    positions `(T, P, 2)` in metres from the centre spot keyed by PersonId with team, officials
    excluded, player speed in **km/h** (`players_speed_kmh`; converted to m/s only through
    `kmh_to_mps()`), and ball `(T, 2)` plus `z` (z is used only by the degrader and never by a labeller).
  - `Labels`: `(T,)` arrays `ball_status ∈ {0, 1, -1 unknown}` and `possession ∈ {1, 2, -1}`.
  - `Meta`: team ids/names (home/away), pitch length/width, game sections, frame-rate estimate.
- Target labels for the metrics, built from `Labels` only: `D` if status = 0; `H` / `A` if status = 1
  and possession = 1 / 2; `X` if anything is unknown (never graded).
- Conversion to V1: coordinates are transformed to the ATD convention (origin at a corner, x along
  the length, normalised 0–1 when written to ATD files) after P0 measures DFL's native origin and
  axes. Each game section becomes one graded block, laid out on a single frame axis with a gap
  between sections.
- The benchmark feeds **the same ATD txt + meta XML pair** to V1 (`pitchvision.atd.load_arrays` →
  `pipeline.run`) and to the fallback (`starter.baseline.nearest_player_labels` → `smooth(…, 50)`),
  with the DFL pitch passed explicitly. Identical input, identical parse path, identical blocks.

### Degradation (P1)

Seeded and deterministic. The parameters come from presets, not from BAR-ATH:

| knob | effect |
|---|---|
| `ball_present` | target share of frames with a ball observation, removed in bursts (two-state Markov chain with a mean gap length); dropout more likely when ≥ k players are within r of the ball or when z > 1.5 m |
| `tracklet_mean_s` | each player track is cut into tracklets, and each tracklet is assigned to a free slot from a pool (default 46) |
| `swap_rate` | identity swaps between same-team players closer than 1 m |
| `player_dropout` | short occlusion gaps per player |
| `noise_player_m`, `noise_ball_m` | Gaussian position noise |
| `false_ball_rate` | spurious ball detections displaced from the true ball |

Presets: `clean` (format conversion only), `mild`, `doc` (matches only the **public** `DATA.md`
figures: ball present on about 59 % of in-play frames and about 20.5 tracked objects; nothing else),
and `severe`. The benchmark reports every preset. Conclusions are drawn only from what holds across
`mild`–`severe`.

### Benchmark (P2)

- Each match is scored separately with `possession_metrics.evaluate` (spellF1, macroF1, per-class
  F1 including D, alive/dead F1, possession counts), for V1 and the fallback, at every preset.
- Reported **per match**, with match-level bootstrap intervals (spellF1 via `bootstrap_spell_f1`).
  Aggregates are also given in two strata: the two Bundesliga matches, and the five Düsseldorf
  home matches. Five of seven matches share a home team and a stadium, so they are not independent
  samples.
- An "early stop/go" later (after P3) needs a gain larger than the intervals, consistent across
  matches and strata. Seven matches is a small sample, and the report states that.
- Definition gap: DFL's possession ("player in control") and BallStatus ("interrupted by the
  referee") need not match the competition's labels. P0 reports the reference's own possession
  counts and durations, so the gap is visible, and the benchmark compares methods only *relative
  to each other*.

## Leakage guards

- `dfl_parse` returns targets in a separate object. The writer `convert.write_atd` takes only
  `DFLTrack`, so the ATD files cannot contain labels.
- Test: permuting or zeroing every label leaves both labellers' outputs byte-identical.
- Test: the ATD file and meta written for a fixture contain no BallStatus or BallPossession text.

## T4–T6 as implemented (2026-11-04)

- **Conversion (`pvext/convert.py`).** `layout()` returns the tracking (`FramesTrack`, metres from a
  corner, speed in m/s through `kmh_to_mps`) and the scoring target separately. Sections are laid
  end to end with a 250-frame gap between them, and each section is one graded block.
  `write_atd()` has no label parameter, writes the home team first, and round-trips exactly
  through the frozen `pitchvision.atd.load_arrays`.
- **Degradation (`pvext/degrade.py`).** The presets are `clean`, `mild`, `doc` and `severe`. Only
  `doc`'s ball presence (59.3 %), visible-object count (about 20.5 of 22) and 46 slots come from
  `DATA.md`; the other values are a stated severity sweep. Deviations from the original spec:
  **identity swaps are not modelled separately**, because tracklet fragmentation already breaks
  identities and V1 uses no identity across frames; and noise is **AR(1)** (rho 0.99) rather than
  i.i.d. Gaussian, because per-frame jitter would invent speeds of several m/s.
- **Benchmark (`pvext/benchmark.py`).** For each preset and seed, one ATD file pair is written and
  read by both V1 (frozen `configs/v1.json`) and the fallback (`smooth(…, 50)`). Primary metrics:
  alive/dead F1, D precision/recall/F1, and dead-ball boundary F1 at ±0.5, ±1 and ±2 s with the
  median offset. Secondary: possession spellF1. Tests prove identical inputs (the same path and
  sha256 for both loaders) and that predictions do not depend on the targets.

## Known issues (non-blocking, from the T4–T6 review of `a8596f3`)

To be fixed, each with a regression test, the next time the module is touched:

1. **`degrade._crowd()` ignores the preset's `crowd_radius_m`.** It reads the hard-coded
   `_BASE["crowd_radius_m"]` (`pvext/degrade.py`). It has **no effect so far**, because every
   degraded preset sets 3.0, equal to `_BASE`, so the fix must leave all current presets'
   outputs byte-identical. The regression test will be a preset whose radius differs from `_BASE`
   and must change which frames count as crowded, plus byte-identity for the four frozen presets.
2. **`convert.write_atd()` writes a constant 0 in the ball-speed field** (the third value of the
   ball group). Neither V1 nor the fallback reads it, so the benchmark is unaffected. **It must
   not be used as a feature until corrected.** The fix will write a real, position-derived ball
   speed in m/s (or NaN where unknown), verified against the clean track, with a test that the
   predictions of the frozen V1 and the fallback are unchanged. Until then, any V2 feature code
   must compute ball speed from positions and never read this field.

## Evaluation protocol for learned models (P3+, fixed before any model is trained)

The seven IDSSE matches are the only labelled external data, so leakage between training and
evaluation is the main threat to any conclusion.

- **Match-held-out only.** Every split keeps whole matches together. There are never random
  frame-level or segment-level splits: neighbouring frames of one match are near-duplicates.
- **Primary scheme: leave-one-match-out (7 folds).** Report per-match results and the spread, not
  just a mean.
- **Repeated team and venue.** Fortuna Düsseldorf is at home, in the same stadium with the same
  tracking installation, in 5 of 7 matches. Two extra views are therefore reported, and a
  conclusion must hold in both:
  1. **Leave-competition-out**: train on the five 2. Bundesliga matches and test on the two
     Bundesliga matches (J03WMX, J03WN1), which share no team or venue with them, and vice versa.
  2. **Stratified reporting**: separate aggregates for the Bundesliga pair and the Düsseldorf five.
     Every fold whose test match shares a team with a training match is flagged.
- **Model selection inside the training folds only** (nested), never on the held-out match.
  Degradation seeds also differ between training and evaluation copies of the data.
- **Uncertainty**: match-level bootstrap intervals. With 7 matches, a gain counts only if it holds
  across matches and in both views above. The early stop/go decision after the first learned
  classifier uses this rule.
- **No competition data** (BAR-ATH or anything derived from it) is used for fitting, selection or
  calibration, and the public 5-minute GT is never used to compare models.

## Out of scope (P3+)

The learned alive/dead classifier, scikit-learn, the HSMM, restart-event features, ball-gap
interpolation, and any change to V1.
