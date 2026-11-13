# PitchVision V2 P0–P2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an external benchmark that scores frozen V1 and the official fallback, on identical inputs, against IDSSE/DFL alive/dead and possession targets, at several degradation levels.

**Architecture:** A new package `pvext/` parses DFL XML (streaming), separates targets from tracking, converts the tracking to the ATD file format V1 already reads, degrades it with seeded presets, and runs V1 and the fallback per match through the official metric code. No V1 file is edited.

**Tech Stack:** Python 3.13, numpy 2.5.4, pandas 3.0.6, stdlib `xml.etree.ElementTree.iterparse`, `hashlib`, `urllib`; pytest 9.2.0. No new runtime dependency. kloppy 3.18.0, already in `.venv` for the starter notebooks, is used only as an optional cross-check in one dev test.

**Spec:** `docs/superpowers/specs/2026-11-04-pitchvision-v2-external-benchmark-design.md`

## Global Constraints

- Branch `pv/v2` from `a8ee2b7`. Do not edit `pitchvision/*`, `scripts/run_v1.py`, `diagnose.py`, `verify_submission.py`, `configs/v1.json`, or any official file.
- No BAR-ATH data, and nothing derived from it, in fitting, calibration or tests. `Dataset/`, `data/` and `evaluation/gt/*.csv` are never opened.
- IDSSE files live in `IDSSE_DIR` (default `$IDSSE_DIR`, outside the repo). Each download needs explicit approval and is MD5-verified.
- `BallStatus` and `BallPossession` are targets only. No labeller input contains them.
- No public-GT use. Benchmark numbers are relative comparisons, not BAR-ATH predictions.
- Every commit needs approval. Never push without approval.
- Tests: `.venv/bin/python -m pytest -W error`. Use parametrize for multi-input assertions and `tmp_path` for files.

## Review Focus

- DFL frames where `BallStatus` / `BallPossession` are missing → target `X` (never default to away/dead) (Task 2).
- Game sections with non-contiguous or overlapping `N` ranges, or players whose FrameSets start late (substitutes) → aligned by `N` with NaN, gaps reported (Task 2).
- Officials' FrameSets in the positions file → excluded from players (Task 2).
- Coordinate convention wrong (origin/axis flip) → caught by a sanity test that ball and players fall inside `[0, L] × [0, W]` after conversion on the real match (Task 3/4).
- Label leakage through the ATD writer → text and permutation tests (Tasks 4, 6).

---

### Task 1: IDSSE manifest and verified fetch

**Files:** Create `pvext/__init__.py`, `pvext/idsse.py`, `scripts/fetch_idsse.py`, `tests/ext/__init__.py`, `tests/ext/test_idsse.py`

**Interfaces:**
- Produces: `idsse.MATCHES: dict[str, dict]` with keys `competition`, `home`, `away`, and `files` (`meta`, `events`, `positions` → `{name, url, md5, size}`). `idsse.verify(path, md5) -> bool`. `idsse.local_paths(match_id, root) -> dict`.
- `scripts/fetch_idsse.py --match J03WPY [--root DIR] [--parts meta,positions]` downloads to an exclusively created `mkstemp` temporary file inside the verified root (no pre-existing path or symlink is ever written through), verifies the MD5 and size, then moves it into place with `os.replace`; a symlinked target is refused. It skips files that already exist and verify, and exits non-zero on a mismatch.

- [ ] **Step 1: Write failing tests**
  - `test_manifest_ids`: the keys are exactly `{"J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"}`, and `"J03WPF"` and `"J03WQF"` are absent.
  - `test_manifest_competitions`: J03WMX and J03WN1 use `DFL-COM-000001`, the others `DFL-COM-000002`. File names follow `DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{id}.xml`.
  - `test_verify` (parametrize good/bad md5) on a `tmp_path` file.
  - `test_fetch_rejects_bad_md5`: point a fake manifest entry at a `file://` URL whose MD5 does not match. Expect exit 1, and no final file left behind.
  - `test_fetch_skips_verified`: the file already exists and verifies, so nothing is opened (the URL is unreachable and the call still succeeds).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/ext/test_idsse.py -v` → FAIL (no module).
- [ ] **Step 3: Implement.** Hard-code the 21 entries (names, figshare `ndownloader` URLs, MD5 values, sizes) from `https://api.figshare.com/v2/articles/28196177/files?page_size=50`. J03WPY's are already known: meta `51643487` / `65a4568e4f374af94d3c1015cbc7c4f5`, events `51643505` / `f68482441d38723d69f244f896482803`, positions `51643526` / `e56ff1336a334f8da51110fbe7cab764` (371,688,499 B). Teams come from the paper's Table 1.
- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Commit (after approval)** `feat(pvext): IDSSE manifest and verified fetch`.
- [ ] **Step 6: Feasibility download (needs explicit approval):** `fetch_idsse.py --match J03WPY` downloads 3 files, about 372 MB, from figshare.

### Task 2: Streaming DFL parser with separated targets

**Files:** Create `pvext/dfl_parse.py`, `tests/ext/dfl_fixture.py`, `tests/ext/test_dfl_parse.py`

**Interfaces:**
- Produces:
  - `DFLTrack` namedtuple: `sections: list[str]`; per section `n: (T,) int`; `players_xy: (T, P, 2) f32` (m, centre origin); `players_speed_kmh: (T, P) f32` (**km/h**, as DFL `S`); `person_ids: list[str]`; `team_of_person: (P,) int8` (0 home / 1 away); `ball_xy: (T, 2) f32`; `ball_z: (T,) f32`.
  - `Labels`: per section `ball_status: (T,) int8 ∈ {0, 1, -1}` and `possession: (T,) int8 ∈ {1, 2, -1}`.
  - `Meta`: `home_id`, `away_id`, `names`, `pitch: (L, W)`, `officials: set`, `fps_estimate`.
  - `parse(meta_xml, positions_xml) -> (DFLTrack, Labels, Meta)` streams the positions file with `iterparse` and clears elements as it goes.
  - `tests/ext/dfl_fixture.write(path, sections={...})` writes a tiny DFL-format positions file plus a match-information file.

- [ ] **Step 1: Write failing tests** (fixture: 2 sections, 2 home, 2 away, 1 official, ball):
  - `test_targets_separate`: `DFLTrack` has no status or possession field, and `Labels` holds them.
  - `test_missing_status_or_possession_is_unknown` (parametrize which attribute is dropped): the value is `-1`, never the away or dead default.
  - `test_officials_excluded`: the official's PersonId is not in `person_ids`.
  - `test_substitute_aligned_by_n`: a player starting at N + 50 has NaN before that point and the right position after it.
  - `test_n_gap_reported` (parametrize contiguous/gap/duplicate): a gap gives `ValueError` or a report, as the spec says. Duplicates raise.
  - `test_matches_kloppy_on_fixture` (dev-only cross-check; skip if kloppy is missing): for frames where both attributes are present, the ball x/y and status agree with `kloppy.sportec.load_tracking`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Use `iterparse` with `events=("start", "end")`. On FrameSet start, record `GameSection` and `PersonId`/`TeamId`. On Frame end, store `N`, `X`, `Y`, `S` (`Z`, `BallStatus` and `BallPossession` only for the ball). Clear each element after use. Build arrays per section over the union of `N`. Read pitch, teams and officials from the match-information XML.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (after approval)** `feat(pvext): streaming DFL parser, targets kept separate`.

### Task 3: P0 feasibility probe on J03WPY (public data, aggregate report)

**Files:** Create `scripts/inspect_idsse.py`, `tests/ext/test_inspect.py`

**Interfaces:**
- `inspect_idsse.py --match J03WPY --out JSON` writes only aggregates. For each section: frame count; N range and any gaps; `fps_estimate` (from N spacing, plus the `T` timestamps if present); x/y min/max/percentiles for the ball and for players (to infer the origin and axes); players present per frame; ball-present share; z range; value counts for `BallStatus` and `BallPossession`, including unknown; DEAD run count and duration percentiles; possession spell count and duration percentiles. It also gives the possession-switch rate per minute, to show the definition gap.

- [ ] **Step 1: Write failing test** on the synthetic fixture: the JSON has these keys, the counts are right, and there is no list longer than 3 (aggregates only).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Run on J03WPY**, after its download is approved, and record the findings in `docs/EXTERNAL_DATA.md`: origin and axes, frame rate, label semantics checks, unknown share, and the definition-gap figures. **This is the go/no-go point for using IDSSE at all.**
- [ ] **Step 6: Commit (after approval)** `feat(pvext): IDSSE feasibility probe`.

### Task 4: Conversion to V1 ATD files

**Files:** Create `pvext/convert.py`, `tests/ext/test_convert.py`

**Interfaces:**
- `convert.to_corner_metres(track, meta, origin) -> DFLTrack`. The `origin` rule ("centre" or "corner", axis signs) is the one Task 3 measured: centre origin, x along the length, y bottom to top, so `x + L/2`, `y + W/2`. Speeds are converted from km/h with `dfl_parse.kmh_to_mps()` before they are written to the ATD speed field, which is in m/s; a test pins `3.6 km/h -> 1.0` in the written file.
- `convert.layout(track, labels, gap_frames=250) -> (frames_track, target_labels '<U1', blocks)`. Sections are concatenated with a gap; targets come out as H/A/D/X.
- `convert.write_atd(frames_track, meta, dirpath, n_slots=None) -> (meta_path, raw_path)`. The output is in the exact raw ATD line format (`tests.synthetic` / `starter.baseline` conventions): normalised by the DFL pitch, one Player per slot with its teamId. It takes **only** tracking.

- [ ] **Step 1: Write failing tests**
  - `test_roundtrip_through_v1_loader`: `pitchvision.atd.load_arrays(write_atd(...), pitch=meta.pitch)` reproduces the positions in metres (to 1e-3) and `ball_present`.
  - `test_fallback_parser_agrees`: `baseline.nearest_player_labels` and `pitchvision.atd.nearest_team` agree on the written file.
  - `test_targets_mapping` (parametrize status/possession → `D`, `H`, `A`, `X`).
  - `test_no_label_text_in_atd`: the written files contain no "BallStatus" or "BallPossession", and the bytes are independent of the labels.
  - `test_blocks_cover_sections_with_gap`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (after approval)** `feat(pvext): DFL to ATD conversion`.

### Task 5: Seeded degradation presets

**Files:** Create `pvext/degrade.py`, `tests/ext/test_degrade.py`

**Interfaces:**
- `degrade.PRESETS: dict[str, dict]` with `clean`, `mild`, `doc`, `severe`. `doc` uses only `DATA.md`'s public figures (ball on about 59 % of in-play frames, about 20.5 tracked objects, 46 slots).
- `degrade.apply(frames_track, preset: dict, seed: int) -> frames_track`, which returns slot-based tracking (P → `n_slots`).

- [ ] **Step 1: Write failing tests**
  - `test_deterministic` (same seed gives byte-identical ATD after `write_atd`).
  - `test_ball_present_rate` (parametrize target 0.9/0.59): achieved within ±0.02 on a 20,000-frame fixture.
  - `test_dropout_bursty`: mean missing-run length ≥ the preset's `mean_gap_frames` × 0.8.
  - `test_slots_bounded_and_tracklets`: never more than `n_slots` slots, and no person keeps one slot for longer than the cap.
  - `test_clean_is_identity_up_to_slots`.
  - `test_degrader_never_sees_labels`: the signature takes no labels, and outputs do not change when the labels are permuted.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (after approval)** `feat(pvext): seeded degradation presets`.

### Task 6: External benchmark (V1 vs fallback, identical inputs)

**Files:** Create `pvext/benchmark.py`, `scripts/run_benchmark.py`, `tests/ext/test_benchmark.py`

**Interfaces:**
- `benchmark.run_match(match_id, root, presets, seeds, cfg) -> list[dict]`. For each (preset, seed) it calls `write_atd` once and then runs: V1 = `pitchvision.pipeline.run(pitchvision.atd.load_arrays(...), cfg, 25, meta.pitch, blocks)`, and fallback = `baseline.smooth(baseline.nearest_player_labels(raw, team_of_each_track(meta_xml), T, meta.pitch), 50)`. Both are scored with `possession_metrics.evaluate(target, pred)` and `bootstrap_spell_f1`.
- `run_benchmark.py --matches J03WPY [...] --presets clean,mild,doc,severe --seeds 0,1,2 --out DIR` writes `per_match.csv` (one row per match × preset × seed × method), `summary.json` (per preset and method: the mean over matches, a per-match table, and strata `bundesliga` / `duesseldorf_home`), and the run's config hash and git commit.

- [ ] **Step 1: Write failing tests** (2 synthetic DFL fixture matches):
  - `test_identical_inputs`: V1 and the fallback read the same `raw_path`, checked by sha256.
  - `test_label_permutation_invariance`: shuffling or zeroing the targets changes the scores but not either method's predictions.
  - `test_scores_match_official_metric`: recomputing with `pm.evaluate` gives identical numbers.
  - `test_per_match_and_strata_reported`.
  - `test_deterministic_report`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit (after approval)** `feat(pvext): external benchmark for V1 and fallback`.
- [ ] **Step 6:** Run on J03WPY. If P0 is a go and the remaining 6 downloads (about 2.25 GB) are approved, run on all 7. Record the per-match table in `docs/EXTERNAL_DATA.md` without interpreting it as BAR-ATH accuracy.

### Task 7: Licence, attribution and storage documentation

**Files:** Create `docs/EXTERNAL_DATA.md`

- [ ] **Step 1:** Write the CC BY 4.0 attribution. It covers: the data owner DFL Deutsche Fußball Liga, collected by Sportec Solutions (TRACAB); the dataset DOI 10.6084/m9.figshare.28196177; the paper (Bassek et al., *Scientific Data* 12:195, 2025, 10.1038/s41597-025-04505-y); the licence link; and **the changes made** (format conversion, synthetic degradation), which CC BY requires. It also records where the files are stored (outside the repo, never committed), the MD5 manifest, the verified match IDs and the README discrepancy, the P0 findings, the team-overlap caveat, and the definition-gap caveat.
- [ ] **Step 2: Run** the full suite → PASS.
- [ ] **Step 3: Commit (after approval)** `docs(pvext): external data licence and attribution`.

**Stop after Task 7** for the P0–P2 review. No classifier and no HSMM before approval.
