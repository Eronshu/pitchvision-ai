# PitchVision V1 — Design

Date: 2026-11-04 · Branch: `pv/v1` · Deadline: 2026-11-17

## Goal

A reproducible, rule-based possession labeller for the BAR-ATH tactical ATD that produces a valid
`frame,label` CSV, with every uncertain convention exposed in a config file. V1 is *not* assumed
to beat the official `baseline_smooth50`; that baseline stays an independent fallback.

## Constraints

- Official files (`games.py`, `starter/`, `evaluation/`, top-level docs) are not modified.
- No manual labels on BAR-ATH (organiser clarification, issues #8 and #10).
- No tuning or model selection on the public 5-minute GT; it is a pipeline sanity check only.
- No BAR-ATH frame-specific rules. All thresholds are physical/football priors in `configs/v1.json`.
- Claude-assisted work uses synthetic data only. Real-data runs happen in the project lead's local
  terminal; match-derived outputs are not shared with a cloud LLM unless the licence allows it.
- Dependencies: numpy and pandas only (stdlib `unittest` for tests, stdlib `json` for config).

## Validation layers

| layer | what | status |
|---|---|---|
| L0 | synthetic unit + scenario tests | mandatory gate for every change |
| L1 | unsupervised diagnostics (counts, durations, flicker, stability) | plausibility, not accuracy |
| L2 | Smart Tagging consistency | supporting evidence only; **not** independent when the same tags feed the model |
| — | public GT | pipeline correctness only — never a score comparison or a gate between versions |
| — | external licensed matches | deferred |

## Layout

```
pitchvision/
  config.py        load + validate configs/*.json, defaults, hash
  atd.py           raw ATD -> arrays; reuses starter/baseline constants + team_of_each_track
  features.py      per-frame evidence arrays
  alive.py         ALIVE/DEAD from weighted evidence + persistence
  possession.py    H/A/UNKNOWN state machine with hysteresis
  tagging.py       optional Smart Tagging evidence (wraps starter/load_smart_tagging.load)
  pipeline.py      compose -> labels; write via baseline.write_submission; validate via pm
  diagnostics.py   L1/L2 aggregate statistics (JSON)
configs/v1.json
scripts/run_v1.py      --config --out  -> submission.csv + manifest.json
scripts/diagnose.py    submission.csv -> diagnostics.json
tests/synthetic.py     writes tiny ATD txt + meta XML fixtures
tests/test_*.py
```

## Data contracts

`atd.load_arrays(meta_path, raw_path, n_frames, pitch)` returns a `Tracking` namedtuple, indexed by
**video frame** (`frame_id - 1`):

| field | shape / dtype | meaning |
|---|---|---|
| `players_xy` | `(T, S, 2)` float32, metres, NaN if missing | slot positions |
| `players_speed` | `(T, S)` float32 | slot speed from ATD (m/s) |
| `team_of_slot` | `(S,)` int8 | 0 = H, 1 = A (from `baseline.team_of_each_track`) |
| `ball_xy` | `(T, 2)` float32, metres, NaN if missing | ball position |
| `ball_present` | `(T,)` bool | ball observed |

Checks that raise `ValueError` with a clear message: slot count ≠ meta `Player` count, more than
two team ids, empty slot group, malformed line. Equivalence test: the nearest-team array derived
from `Tracking` equals `baseline.nearest_player_labels` on the same synthetic files.

Output of `pipeline.run(config)` is a `(199319,)` array of `"H" | "A" | "D" | "X"`.

## Features (`features.py`)

Per frame, from `Tracking`, all NaN-safe:

- `dist_h`, `dist_a`: nearest home / away player to the ball (m).
- `ball_speed`: from finite differences of `ball_xy` over present frames (m/s), lightly smoothed.
- `ball_out`: ball outside the pitch by more than `out_margin_m`.
- `players_motion`: median tracked-player speed.
- `gap_len`: length of the missing-ball run containing the frame (0 if present).
- `rel_speed_h/a`: |ball velocity − velocity of each team's nearest player| (m/s); low when a
  player moves with the ball, so dribbling at speed still counts as control evidence.

## Alive / dead (`alive.py`) — refinement 1, revised after review of PR #1

Two states, entered on a trigger and left only on restart evidence:

- **ALIVE → DEAD triggers.** (a) The ball is observed out of bounds beyond `out_margin_m` for
  `out_confirm_frames` frames in a row, or for fewer when a real exit explains it: an in-play
  sighting within `exit_lookback_s` before, at most `exit_near_m` from a line, with an implied speed
  ≤ `max_ball_speed_mps`. Tracking often loses the ball right after it leaves the pitch, so the
  state, not the visible frames, carries the stoppage. A lone detection jumping out of play from
  mid-pitch is noise. (b) Soft evidence `s_dead` (centred mean over `smooth_s`)
  stays ≥ `dead_on` for `min_dead_s`. `s_dead` = `w_static`·(ball at rest) + `w_motion`·low field motion +
  `w_gap`·missing-ball gap + `w_tag`·tag cue (optional). No single cue reaches `dead_on`. A still
  ball within `r_hold_m` of a player counts as at rest only if the still spell outlasts
  `hold_max_s` (default 6 s). That bound is a **provisional heuristic**, loosely inspired by the
  goalkeeper six-second rule and not a reliable ground-truth boundary: a goalkeeper holding the ball
  is live play, while a taker standing over a set piece is not. A slow field alone never means a
  stoppage.
- **DEAD → ALIVE** only when the ball is observed in bounds and moving ≥ `restart_speed_mps` for
  `restart_min_s`, or after `max_dead_s` as a safety valve. A kicker standing beside the ball
  before kick-off does not end the stoppage. A trigger re-arms only after it has been off: after a
  restart the soft trigger, after a timeout both. The ALIVE stretch a timeout opens is never
  absorbed back into DEAD, so the timeout is a deterministic recovery, not a pause.
- Post-processing: drop DEAD runs shorter than `min_dead_s` (a trigger followed at once by a
  restart is a tracking blip), absorb ALIVE runs shorter than `min_alive_s` between two DEAD runs,
  then apply the 🔶 `dead_start` / `dead_end` offsets (issue #31, default 0).

## Possession (`possession.py`) — refinements 2 and 3

States: `H`, `A`, `UNKNOWN`. Within DEAD the output is `D`, whatever the state is.

**Per-frame control evidence** `e_h, e_a ∈ [0, 1]`: a soft score from (a) the team's nearest
distance against `r_ctrl` (decaying, not a hard cut), (b) the margin over the other team, and
(c) relative motion: the ball decelerating toward that player or staying near them. Ball speed
lowers the evidence but **never vetoes it**, so fast receptions, one-touch play and interceptions
still count. Missing ball → no evidence.

**Hysteresis**: from state X, switch to the other team Y when Y's evidence, accumulated over a
sliding window of `win_s`, exceeds `switch_on` and beats X's evidence by `switch_margin`. Below
that the state holds. Loose ball, passes in flight and missing ball therefore keep the last
controlled possession.

🔶 `boundary` (issue #17): `"evidence_start"` back-dates a confirmed switch to the first frame of
the evidence run that produced it; `"confirmed"` places it at the confirmation frame. Raw ATD
cannot detect touches, hence the name.

**UNKNOWN**: the state at the start of each graded half and after every DEAD→ALIVE restart. It
resolves to the first team whose accumulated evidence passes `init_on`, which is a lower bar than
`switch_on`. That label is back-filled to the restart frame. Optionally, a Smart Tagging set-piece
Team tag breaks a tie. If UNKNOWN never resolves before the next DEAD, the frames are labelled
with the nearest-team majority over the stretch. If even that is undefined, they get `X`, which
the scorer forward-fills.

**Half starts**: graded blocks come from `games.half_video_blocks`. State resets at each block.
Frames outside the blocks are `X`. The pre-kickoff stretch inside a graded block, including the
second-half one, needs no special case. Players are static and the ball sits at the centre spot,
so `alive.py` labels it DEAD through the generic static-ball and low-motion evidence. There is no
frame-specific rule.

## Smart Tagging (`tagging.py`) — refinement 4

Off by default in the first end-to-end run. Two independently switchable uses:
`use_setpiece_dead` (a soft dead-ball cue, `w_tag`) and `use_setpiece_team` (seeds the possession
state at a restart inside a SET PIECES interval; the switch rule can overturn it). Team names
are mapped to H/A by normalised exact match against the ATD meta team names. A mismatch raises an
error. `N/A` is ignored. L2 figures are marked independent only when **no** tagging input is
enabled in that run: either cue can move either figure (the dead cue creates restarts the team
figure scores; the team cue resolves restart runs the timing figure counts). Agreement with an
enabled input is reported, labelled "not independent". Tags count when their interval overlaps a
graded block; restarts outside graded blocks never match.

## Baseline fallback — refinement 5

`baseline_smooth50` is kept verbatim. `scripts/run_v1.py --fallback-baseline` writes it through the
official code path. The decision to submit V1 instead of the fallback is made explicitly at the end,
from the L0 tests and the L1/L2 evidence, and is never assumed. The public 5-minute GT plays no part
in it: it is used only to check that the pipeline is correct (rows aligned, file well-formed,
labels meaning what we think), never to compare the scores of V1 and the fallback or of any two
versions.

## Reproducibility

`run_v1.py` writes `manifest.json`: config and config sha256, git commit, dirty flag, input file
sizes, and output sha256. Two runs with the same config must be byte-identical (tested on
synthetic data). No randomness except seeded perturbations in diagnostics.

## Diagnostics (`diagnostics.py`)

Aggregates only, per graded half and in total: number of spells, duration percentiles, share of
spells < 1 s, transitions/min, D share and D-run durations, home share, agreement with
`baseline_smooth50`, and stability under seeded removal of 10 % of ball observations (spellF1
between the two outputs via `pm.spell_scores`). L2: share of set-piece tags containing a predicted
DEAD→ALIVE transition, and team agreement within tagged phases.

## Testing (synthetic, `unittest`)

Fixture generator writes ATD txt + meta XML for scripted scenarios. Cases: frame alignment;
equivalence with the baseline parser; malformed input errors; pass between teammates (no switch);
interception at speed (switch; boundary per config); deflection then regain (no switch); loose
ball (hold); 2 s missing ball while play continues (hold, no D); ball out → static → restart
(D, then the restarting team); pre-kickoff static period (D, no special case); UNKNOWN resolution
and back-fill; determinism; output passes `pm.validate_submission`; `pm.evaluate` on a synthetic
GT gives the expected spellF1.

## Delivery order

1. Minimal end-to-end: `config` → `atd` → `features` → `alive` → `possession` → `pipeline` →
   `run_v1.py` → valid CSV. Smart Tagging off.
2. `diagnose.py` (L1).
3. `tagging.py` + L2 + ablations.
4. Robustness, technical document, clean-clone reproduction.

## Known model limitations

- **Hold threshold.** `alive.hold_max_s` = 6 s is a provisional heuristic, loosely inspired by the
  goalkeeper six-second rule. It is not a ground-truth boundary and should be revisited with L1/L2
  diagnostics.
- **Slot-based velocities.** `rel_speed_*` uses finite differences of tracking-slot positions.
  Slots are tracklets without stable identities, so identity switches and changes of nearest
  slot produce velocity spikes. Smoothing (5 frames) and `speed_floor` limit their effect on
  control evidence; they do not remove it.
- **Set-piece Team tag semantics.** `use_setpiece_team` assumes the tag names the restarting side.
  The tagging documentation does not say so; L2 `restart_team.agreement` with the flag off is the
  check.
- **Brief exits without a tracked approach** (seen out for 1–2 frames, not seen in play just
  before) are not triggers.

## Out of scope for V1

HMM/HSMM, learned models, video, broadcast feed, homography, use of `X` inside possessions.
