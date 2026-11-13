# PitchVision V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible rule-based H/A/D/X labeller for the tactical ATD. It writes a valid 199,319-row submission and keeps `baseline_smooth50` as a fallback.

**Architecture:** Raw ATD is parsed into arrays (`atd.py`) and turned into per-frame evidence (`features.py`). A weighted-evidence ALIVE/DEAD machine runs with persistence (`alive.py`), followed by an H/A/UNKNOWN hysteresis machine (`possession.py`). `pipeline.py` composes them per graded half. Smart Tagging is optional and added last.

**Tech Stack:** Python 3.13 (`.venv`), numpy 2.5.4, pandas 3.0.6 (scorer only), pytest 9.2.0 (dev only), stdlib `json`, `hashlib`, `xml.etree`.

**Spec:** `docs/superpowers/specs/2026-11-04-pitchvision-v1-design.md`

## Global Constraints

- Do not modify `games.py`, `starter/*`, `evaluation/*`, top-level `*.md`, or `.gitignore`. Import from them instead.
- Never open `Dataset/`, `data/`, or `evaluation/gt/*.csv` from Claude. Tests use synthetic fixtures only.
- Never run `scripts/run_v1.py` or `scripts/diagnose.py` against real data from Claude. The project lead runs them locally.
- No BAR-ATH frame numbers in `pitchvision/` code. Graded blocks come from `games.half_video_blocks(GAME)` only.
- No tuning on the public GT. Every default in `configs/v1.json` carries a one-line prior rationale in `configs/README.md`.
- Runtime dependencies: numpy and pandas only (`requirements-pv.txt`). Tests use pytest, pinned in `requirements-dev.txt`. Run with `.venv/bin/python -m pytest -v`. Use `@pytest.mark.parametrize` wherever one assertion is checked over several inputs (marked "parametrize" below), and `tmp_path` for fixture files.
- Video-frame index everywhere: `video_frame = frame_id - 1`.
- `git commit` steps need explicit approval from the project lead. Without it, leave the work uncommitted and continue. Never push.
- Outputs go to a user-supplied `--out` directory, never inside the repo.

## Review Focus

- Meta XML containing a third teamId (e.g. referees) or a slot count different from the raw line: this must raise `ValueError` naming the file, not silently map to AWAY (Task 2).
- A frame where the ball is present but every player slot is NaN: no evidence and no crash; the possession state holds (Task 3).
- Ball coordinates a few cm outside [0, 1] from tracking noise near a line: this must not create D unless it exceeds `out_margin_m` and persists `min_dead_s` (Task 4).
- An ALIVE stretch whose UNKNOWN state never resolves before the next DEAD: nearest-team majority, else `X`, no exception (Task 5).
- State leaking across halves: the label at the first frame of half 2 must not inherit half 1's last state, and frames outside graded blocks are `X` (Task 6).

---

### Task 1: Package scaffold and config

**Files:**
- Create: `pitchvision/__init__.py`, `pitchvision/config.py`, `configs/v1.json`, `configs/README.md`, `requirements-pv.txt`, `requirements-dev.txt`, `pytest.ini`, `tests/__init__.py`, `tests/test_config.py`

`pytest.ini`: `testpaths = tests`, `pythonpath = .` (so `pitchvision`, `games`, `starter` and `tests.synthetic` import from the repo root), `addopts = -p no:cacheprovider` (no `.pytest_cache` in the repo). `requirements-dev.txt`: `-r requirements-pv.txt` and `pytest==9.2.0`.

**Interfaces:**
- Produces: `config.load(path: str) -> dict`, `config.from_dict(d: dict) -> dict` (merge over defaults + validate; `load` = `from_dict(json.load(...))`), `config.defaults() -> dict`, `config.sha256(cfg: dict) -> str` (hash of canonical `json.dumps(cfg, sort_keys=True)`).

`configs/v1.json` (exact defaults; `configs/README.md` gives one rationale line per key):

```json
{
  "features": {"ball_speed_smooth_frames": 5, "out_margin_m": 1.0},
  "alive": {"w_out": 1.0, "w_static": 0.35, "w_motion": 0.35, "w_gap": 0.3, "w_tag": 0.2,
            "static_speed_mps": 0.5, "static_min_s": 1.0,
            "motion_low_mps": 0.8, "motion_high_mps": 1.8, "gap_scale_s": 4.0,
            "smooth_s": 1.0, "dead_on": 0.6, "dead_off": 0.4,
            "min_dead_s": 2.0, "min_alive_s": 1.0,
            "dead_start_offset_frames": 0, "dead_end_offset_frames": 0},
  "possession": {"r_ctrl_m": 1.5, "r_decay_m": 1.5, "margin_m": 1.0,
                 "v_ref_mps": 15.0, "speed_floor": 0.3,
                 "win_s": 0.4, "evidence_on": 0.2, "switch_on": 0.5, "switch_margin": 0.2,
                 "init_on": 0.3, "boundary": "evidence_start"},
  "tagging": {"use_setpiece_dead": false, "use_setpiece_team": false}
}
```

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`:
  - `test_defaults_match_file`: `config.load("configs/v1.json") == config.defaults()`.
  - `test_invalid_config_rejected` (parametrize over `(override, message_fragment)`): extra key `alive.foo` → `"alive.foo"`; unknown section `bogus` → `"bogus"`; `possession.boundary="first_touch"` → `"boundary"`; `alive.dead_off=0.6, dead_on=0.6` → `"dead_off"`; negative `possession.r_ctrl_m` → `"r_ctrl_m"`. Each `config.from_dict(override)` raises `ValueError` whose message contains the fragment.
  - `test_valid_boundaries` (parametrize `"evidence_start"`, `"confirmed"`): `from_dict` accepts them.
  - `test_hash_stable`: `sha256(defaults()) == sha256(json.loads(json.dumps(defaults())))`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_config.py -v` and confirm it fails with `ModuleNotFoundError: pitchvision`.
- [ ] **Step 3: Implement** `config.py`. `load` merges the file over `defaults()` key by key, rejects unknown keys and validates the constraints above. `requirements-pv.txt` contains `numpy==2.5.4` and `pandas==3.0.6`.
- [ ] **Step 4: Run** the tests and confirm they pass.
- [ ] **Step 5: Commit (after approval)**: `git add pitchvision configs tests requirements-pv.txt requirements-dev.txt pytest.ini && git commit -m "feat(pv): package scaffold and validated config"`.

### Task 2: Synthetic fixtures and ATD arrays

**Files:**
- Create: `tests/synthetic.py`, `pitchvision/atd.py`, `tests/test_atd.py`

**Interfaces:**
- Consumes: `starter.baseline.team_of_each_track`, `team_names`, `nearest_player_labels`, `FIELD_SEP`, `GROUP_SEP` (import via `sys.path` on the repo root, as `starter/baseline.py` itself does).
- Produces:
  - `atd.Tracking` namedtuple `(players_xy (T,S,2) f32 m, players_speed (T,S) f32, team_of_slot (S,) i8, ball_xy (T,2) f32 m, ball_present (T,) bool)`.
  - `atd.load_arrays(meta_path, raw_path, n_frames: int, pitch: tuple) -> Tracking`.
  - `atd.nearest_team(tr: Tracking) -> np.ndarray[int8]` (0/1/-1).
  - `tests.synthetic.write_fixture(dirpath, frames, teams=("T1","T2"), slot_teams=...) -> (meta_path, raw_path)`, where `frames` is a list of `{"players": [(x,y,s) | None, ...], "ball": (x,y) | None}` in normalised coords, `frame_id` starts at 1, missing groups are written as `NaN,NaN,NaN`, and the ball is written as `x,y,0`.
  - `tests.synthetic.make_tracking(players_xy_m, ball_xy_m, team_of_slot, players_speed=None) -> Tracking` (in memory, metres).

The meta XML only needs what `team_of_each_track` reads: `<Team id=..><Name>..</Name></Team>` ×2 and one `<Player teamId=..>` per slot, in slot order.

- [ ] **Step 1: Write failing tests**:
  - `test_alignment`: a 3-frame fixture. `tr.ball_xy[0]` comes from `frame_id 1`, and `tr.ball_present` equals `[True, False, True]` for balls `[(.5,.5), None, (.1,.1)]`.
  - `test_metres`: ball `(0.5, 0.5)` with `pitch (105.3, 68.0)` gives `ball_xy[0] ≈ (52.65, 34.0)`.
  - `test_equivalence_with_baseline`: a 50-frame random fixture (seed 0, 6 slots, about 30% missing values). `atd.nearest_team(tr)` equals `baseline.nearest_player_labels(raw, team_of_each_track(meta), 50, pitch)`.
  - `test_slot_count_mismatch`: meta with 6 Players and raw lines with 5 groups raises `ValueError` containing the raw path.
  - `test_third_team_id`: a Player whose teamId is not one of the two Team ids raises `ValueError`.
  - `test_empty_group`: a raw line with `;;` raises `ValueError` with the line number.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_atd.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `atd.py`. Stream line by line as `baseline.nearest_player_labels` does: split on `FIELD_SEP`, then `GROUP_SEP`, then `,`. Fill preallocated NaN arrays at `int(frame_id) - 1`. Scale x by `pitch[0]` and y by `pitch[1]`. Validate the Player teamIds against the Team ids yourself before calling `team_of_each_track`. `nearest_team` takes the argmin of squared distance over non-NaN slots, with ties going to the lower slot index as in the baseline.
- [ ] **Step 4: Run** the tests and confirm they PASS.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): ATD array loader with baseline-equivalence test`.

### Task 3: Per-frame features

**Files:**
- Create: `pitchvision/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `Tracking`, `cfg["features"]`, `fps: int`.
- Produces: `features.compute(tr, cfg_features, fps, pitch) -> Features`, a namedtuple of `(T,)` float32 arrays: `dist_h, dist_a` (m; `inf` when there is no ball or no player of that team), `rel_speed_h, rel_speed_a` (m/s, |ball velocity − velocity of that team's nearest slot|, falling back to `ball_speed` when that slot was NaN at t−1), `ball_speed` (m/s, centred moving mean over `ball_speed_smooth_frames` of present-to-present finite differences; NaN when the ball is missing), `ball_out` (bool, outside `[−m, L+m] × [−m, W+m]`), `players_motion` (median non-NaN `players_speed`, NaN if none), `gap_len` (frames in the missing-ball run containing t, 0 if present).

- [ ] **Step 1: Write failing tests**:
  - `test_distances`: one home slot at (10,10), one away slot at (20,10), ball (11,10). `dist_h == 1`, `dist_a == 9`.
  - `test_no_players`: ball present, all slots NaN. `dist_h == dist_a == inf`, and there is no warning or exception.
  - `test_ball_speed`: the ball moves 0.4 m per frame at 25 fps, so `ball_speed ≈ 10.0` in the interior.
  - `test_rel_speed_dribble`: player and ball both move 0.2 m per frame. `rel_speed_h ≈ 0` while `ball_speed ≈ 5`.
  - `test_ball_out_margin` (parametrize over x in metres → expected, `out_margin_m=1.0`): `-0.5 → False`, `-1.5 → True`, `L+0.5 → False`, `L+1.5 → True`, and the same for y against `W`.
  - `test_gap_len`: presence `[T, F, F, F, T]` gives `gap_len [0, 3, 3, 3, 0]`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_features.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `compute`, vectorised with numpy. Use `np.nanmin` over per-team slot masks, and suppress all-NaN warnings with `np.errstate` plus explicit masks.
- [ ] **Step 4: Run** the tests and confirm they PASS.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): per-frame evidence features`.

### Task 4: ALIVE/DEAD machine

**Files:**
- Create: `pitchvision/alive.py`, `tests/test_alive.py`

**Interfaces:**
- Consumes: `Features`, `cfg["alive"]`, `fps`, optional `tag_dead: np.ndarray[float32] | None` (0..1, from Task 8).
- Produces:
  - `alive.dead_score(feat, cfg_alive, fps, tag_dead=None) -> np.ndarray[float32]`
  - `alive.dead_mask(feat, cfg_alive, fps, tag_dead=None) -> np.ndarray[bool]` (True = DEAD).

Score per frame, clipped to [0, 1]:
`w_out·ball_out + w_static·static + w_motion·low_motion + w_gap·min(1, gap_len/(gap_scale_s·fps)) + w_tag·tag_dead`.
Here `static` is 1 on frames inside a run of `ball_speed < static_speed_mps` lasting ≥ `static_min_s`, `low_motion = clip((motion_high − motion)/(motion_high − motion_low), 0, 1)`, and NaN motion counts as 0. Then apply a centred mean over `smooth_s`, the hysteresis `dead_on`/`dead_off`, dropping DEAD runs shorter than `min_dead_s` (first), absorbing ALIVE runs shorter than `min_alive_s` that sit between DEAD runs into DEAD, and finally shifting each DEAD run by `dead_start_offset_frames` / `dead_end_offset_frames`.

- [ ] **Step 1: Write failing tests** (scenarios built with `make_tracking`, 25 fps):
  - `test_missing_ball_active_play_not_dead`: 2 s of missing ball with player speeds of 3 m/s gives `dead_mask` all False.
  - `test_single_soft_cue_not_dead`: low motion alone for 10 s, with the ball present and moving, gives all False.
  - `test_static_and_low_motion_dead`: 5 s of static ball plus static players gives True in the interior.
  - `test_out_of_bounds_dead` (parametrize over `(metres_outside, expected)`): `(3.0, True)`, `(0.5, False)`, ball held outside for 4 s.
  - `test_short_dead_dropped`: an out-of-bounds blip of 1 s (`< min_dead_s`) gives all False.
  - `test_offsets`: `dead_start_offset_frames=5` moves the first True frame 5 later.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_alive.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** using run-length helpers (`_runs(mask) -> list[(start, stop)]`) kept private to `alive.py`.
- [ ] **Step 4: Run** the tests and confirm they PASS.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): weighted-evidence alive/dead machine`.

### Task 5: Possession machine

**Files:**
- Create: `pitchvision/possession.py`, `tests/test_possession.py`

**Interfaces:**
- Consumes: `Features`, `dead: bool (T,)`, `cfg["possession"]`, `fps`, `nearest: int8 (T,)` (from `atd.nearest_team`), optional `restart_team: int8 (T,) | None` (Task 8; −1 means none).
- Produces:
  - `possession.evidence(feat, cfg_pos) -> (e_h, e_a)`, float32 in [0, 1].
  - `possession.label_block(feat, dead, nearest, cfg_pos, fps, restart_team=None) -> np.ndarray[str '<U1']` with values in `{"H", "A", "D", "X"}` for one graded block.

Evidence per team `k`:
- `e_dist = 1` if `d_k ≤ r_ctrl`, decaying linearly to 0 at `r_ctrl + r_decay`.
- `e_margin = clip((d_other − d_k)/margin_m, 0, 1)`.
- `e_motion = speed_floor + (1 − speed_floor)/(1 + rel_speed_k/v_ref)`.
- `e_k = e_dist · e_margin · e_motion`, and 0 when the ball is missing.

State machine over each ALIVE run (a run starts UNKNOWN, or starts at `restart_team[start]` when that is ≥ 0), using `m_k` = causal mean of `e_k` over `round(win_s·fps)` frames:

```
UNKNOWN: if max(m_h, m_a) >= init_on and the larger one is strictly larger:
             state = that team; label [run_start, t] with it
state S: other team O: if m_O >= switch_on and m_O - m_S >= switch_margin:
             c = t if boundary == "confirmed" else
                 earliest k in (last_change, t] with e_O[k..t] all >= evidence_on
             relabel [c, t] as O; state = O; last_change = t
end of run still UNKNOWN: majority of nearest[run] over values >= 0
             (ties -> "H", matching the baseline); none -> "X"
DEAD frames -> "D"
```

- [ ] **Step 1: Write failing tests** (`make_tracking`, 25 fps, `dead` all False unless stated):
  - `test_pass_between_teammates_no_switch`: H holds for 2 s, ball flies past an away player at 3 m for 0.3 s, then H receives. Output is all `"H"` after init.
  - `test_fast_interception_switches` (parametrize `boundary` in `"evidence_start"`, `"confirmed"`): H holds, then an away player is within 0.5 m of a 20 m/s ball and keeps it for 1 s. The output switches to `"A"`. With `"evidence_start"` the first `"A"` is at the first frame where `e_a ≥ evidence_on`; with `"confirmed"` it is strictly later.
  - `test_deflection_then_regain_no_switch`: an away player is within 1 m for 3 frames, then H regains. All `"H"`.
  - `test_loose_ball_holds`: the ball is 8 m from everyone for 2 s. The label stays at the previous team.
  - `test_missing_ball_holds`: 2 s of missing ball inside an H spell gives `"H"`, never `"D"` or `"X"`.
  - `test_unknown_backfill`: the run starts with the ball 6 m from everyone, then A takes control at 1 s. The whole run is `"A"`.
  - `test_unknown_unresolved_uses_nearest_or_x`: evidence never reaches `init_on`. With some `nearest == 1` frames the result is all `"A"`; with `nearest` all −1 it is all `"X"`.
  - `test_dead_frames_are_d`: `dead[10:40] = True` gives `"D"` exactly there, and the next ALIVE run starts UNKNOWN.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_possession.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `evidence` vectorised. Implement `label_block` as one Python loop over frames, which is fast enough for 80k frames per block.
- [ ] **Step 4: Run** the tests and confirm they PASS.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): hysteresis possession machine with UNKNOWN state`.

### Task 6: Pipeline, CLI, manifest, fallback — first end-to-end milestone

**Files:**
- Create: `pitchvision/pipeline.py`, `scripts/run_v1.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–5. `games.GAME`, `games.GAMES`, `games.require`, `games.half_video_blocks`, `games.source_frames`, `baseline.write_submission`, `baseline.build`, `baseline.smooth`, and `possession_metrics.read_label_csv`, `validate_submission`, `label_codes`, `evaluate`.
- Produces:
  - `pipeline.run(tr, cfg, fps, pitch, blocks, events=None, team_names=None) -> np.ndarray['<U1']` of length `T`. `X` outside blocks; each block labelled independently.
  - `scripts/run_v1.py --config PATH --out DIR [--fallback-baseline]` writes `DIR/submission.csv` (or `DIR/baseline_smooth50.csv`) and `DIR/manifest.json` with the keys `config`, `config_sha256`, `git_commit`, `git_dirty`, `inputs` (basename → size in bytes), `output_sha256`, `n_rows`, `label_counts`. It exits non-zero if `pm.validate_submission` fails.

- [ ] **Step 1: Write failing tests** (a synthetic 600-frame fixture written with `write_fixture`, blocks `[(1, 50, 299), (2, 350, 549)]`):
  - `test_shape_and_x_outside`: length 600, and `labels[:50]`, `labels[300:350]`, `labels[550:]` are all `"X"`.
  - `test_no_leak_across_halves`: block 1 ends in an A spell and block 2 opens with clear H control from frame 350. `labels[350] == "H"`, because block 2 starts UNKNOWN and back-fills.
  - `test_valid_submission`: a CSV written via `baseline.write_submission` passes `pm.validate_submission(pm.read_label_csv(p), 600) is None`.
  - `test_deterministic`: two runs produce byte-identical CSVs (compare sha256).
  - `test_metric_sanity`: build a synthetic GT from the scripted scenario. `pm.evaluate(gt, pred)["spell_f1_mean"] >= 0.9` on a clean scenario with no missing ball.
  - `test_cli_manifest` (subprocess on the fixture via `--atd-meta/--atd-raw/--n-frames/--blocks` test-only overrides): `manifest.json` has all the keys listed above and `output_sha256` matches the file.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_pipeline.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement** `pipeline.run`: compute features and the nearest team once, then for each block slice the arrays and run `alive.dead_mask` and `possession.label_block`. In `run_v1.py` the defaults come from `games` (`require(["atd_meta","atd_raw"])`, `source_frames("tactical")`, `half_video_blocks(GAME)`, `GAMES[GAME]["pitch"]`, `["fps"]`). The test-only overrides are hidden from `--help`. `--fallback-baseline` calls `baseline.build()` and `baseline.smooth(codes, 50)`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -v` and confirm all tests PASS.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): end-to-end V1 pipeline, CLI and manifest`.
- [ ] **Step 6: Hand-off to the project lead (local terminal, real data, not run by Claude):**

```bash
MTAH_DATA_DIR=$MTAH_DATA_DIR .venv/bin/python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_v1
```

```bash
.venv/bin/python evaluation/validate_submission.py /tmp/pv_v1/submission.csv
```

Expected: `OK`, 199319 rows.

### Task 7: L1 diagnostics

**Files:**
- Create: `pitchvision/diagnostics.py`, `scripts/diagnose.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `pm.possession_spells`, `pm.segments`, `pm.spell_scores`, `pipeline.run`, `atd.Tracking`.
- Produces:
  - `diagnostics.summarize(labels, blocks, fps) -> dict` with keys `n_spells`, `duration_p10_p50_p90_s`, `share_spells_lt_1s`, `transitions_per_min`, `d_share_graded`, `d_run_p50_s`, `home_share`, each per block and as `total`.
  - `diagnostics.agreement(labels_a, labels_b, blocks) -> dict` with `frame_agreement` and `spell_f1_mean`, treating `labels_b` as the reference.
  - `diagnostics.perturb_ball(tr, frac, seed) -> Tracking`, which NaNs `frac` of the present ball frames.
  - `scripts/diagnose.py --config PATH --submission CSV --baseline CSV --out JSON` (real-data runs are done by the lead).

- [ ] **Step 1: Write failing tests**: `summarize` on a hand-built label array (`"H"*50 + "D"*25 + "A"*25` at 25 fps) gives `n_spells == 2` and `d_share_graded == 0.25`. `agreement(x, x)` gives `spell_f1_mean == 1.0`. `perturb_ball` is deterministic for the same seed and removes `round(frac * n_present)` frames. `summarize` ignores frames outside blocks.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_diagnostics.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement**, building blocks into the `runs` list form that `pm` expects: `[(start, stop + 1)]`.
- [ ] **Step 4: Run** the full suite and confirm it PASSES.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): L1 diagnostics`.

### Task 8: Optional Smart Tagging evidence and L2

**Files:**
- Create: `pitchvision/tagging.py`, `tests/test_tagging.py`
- Modify: `pitchvision/pipeline.py` (pass `tag_dead` / `restart_team` when enabled), `pitchvision/diagnostics.py` (add `setpiece_consistency`)

**Interfaces:**
- Consumes: `starter.load_smart_tagging.load` output shape (`{code, start_frame, end_frame, tags}`), and ATD `team_names(meta)`.
- Produces:
  - `tagging.team_map(tag_teams: set[str], atd_names: tuple) -> dict[str, int]`. Uses casefold plus whitespace-collapsed exact matching. `N/A` maps to −1. Any other unmatched name raises `ValueError`.
  - `tagging.setpiece_dead(events, n_frames) -> float32 (T,)`, which is 1 inside `SET PIECES` intervals (code matched case-insensitively).
  - `tagging.restart_team(events, n_frames, tmap) -> int8 (T,)`, the team inside `SET PIECES` intervals, −1 elsewhere.
  - `diagnostics.setpiece_consistency(labels, events, cfg) -> dict` with `share_tags_with_restart` and `independent: bool`, which is `False` when `cfg["tagging"]["use_setpiece_dead"]` is on.

- [ ] **Step 1: Write failing tests**: `team_map({"Barcelona", "atletic  club", "N/A"}, ("Barcelona", "Atletic Club"))` returns `{…: 0, …: 1, "N/A": -1}`. An unknown name raises. `setpiece_dead` marks `[start_frame, end_frame)` exactly. `restart_team` ignores `N/A`. In the pipeline, with both flags off the output is byte-identical to Task 6. With `use_setpiece_team` on, a scripted restart resolves to the tagged team. `setpiece_consistency(...)["independent"]` is `False` when `use_setpiece_dead` is on.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_tagging.py -v` and confirm it FAILS.
- [ ] **Step 3: Implement**, wiring `scripts/run_v1.py` to load events through `require(["smart_tagging"])` only when a flag is on.
- [ ] **Step 4: Run** the full suite and confirm it PASSES.
- [ ] **Step 5: Commit (after approval)**: `feat(pv): optional Smart Tagging evidence and L2 consistency`.

### Task 9: Reproduction doc

**Files:**
- Create: `docs/PITCHVISION.md`

- [ ] **Step 1:** Document install (`pip install -r requirements-pv.txt`), the exact `run_v1.py` and `--fallback-baseline` commands, the config keys marked 🔶 (issues #17, #31), the validation layers, and the rule for choosing V1 over the fallback (L0 + L1/L2 evidence only; the public GT is a pipeline-correctness check and is never used to compare scores).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -v` and confirm all tests PASS.
- [ ] **Step 3: Commit (after approval)**: `docs(pv): reproduction guide`.
- [ ] **Step 4: Project lead, locally:** clean clone of the branch, run the same command, and confirm `manifest.output_sha256` matches.
