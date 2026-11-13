# PitchVision V1 — approach, limitations and reproduction

> **Historical document.** It describes V1, now the last-resort fallback. The submitted method is
> described in [`docs/submission/PitchVision AI.md`](submission/PitchVision%20AI.md).

PitchVision V1 is a rule-based possession labeller for the More than a Hack 2027 Phase 1 match
(FC Barcelona vs Athletic Club). It reads the tactical Automatic Tracking Data and, optionally, the
Smart Tagging, and it writes one `H` / `A` / `D` / `X` label per tactical video frame. It does not
train on anything. Every threshold is a stated prior in [`configs/v1.json`](../configs/v1.json), and
each one is justified in [`configs/README.md`](../configs/README.md).

The design is [`docs/superpowers/specs/2026-11-04-pitchvision-v1-design.md`](superpowers/specs/2026-11-04-pitchvision-v1-design.md).
The official repository's files (`games.py`, `starter/`, `evaluation/`, the top-level docs) are
used unmodified.

## Approach

```
raw ATD ─► atd.py ─► features.py ─► alive.py ──┐
                                               ├─► possession.py ─► labels ─► CSV + manifest
Smart Tagging (optional) ─► tagging.py ────────┘
```

| stage | what it does |
|---|---|
| `atd.py` | Parses the raw ATD into arrays (metres, video-frame index `frame_id - 1`). It uses the official slot-to-team rule and fails loudly on a third team id, a slot-count mismatch, a malformed line, or a duplicate, missing or truncated frame sequence. |
| `features.py` | Computes, per frame: each team's nearest-player distance, ball–player relative speed, ball speed, out-of-bounds, a plausibility check for brief exits, median field speed, and missing-ball gap length. |
| `alive.py` | ALIVE/DEAD. A stoppage is **triggered**, either by the ball out of play (sustained, or a brief sighting a real exit explains) or by soft evidence (a ball at rest, a slow field, a missing ball) that no single cue can supply alone. It then **persists** until a **restart**: the ball moving in play, or a timeout after which triggers must re-arm. |
| `possession.py` | Runs an H/A/UNKNOWN state machine with hysteresis. Control evidence is distance × margin over the other team × relative motion, and ball speed lowers it but never vetoes it. A loose, contested, in-flight or missing ball keeps the last possession. Each live stretch starts UNKNOWN and is back-filled once a team takes control. |
| `tagging.py` | Optional and off by default. SET PIECES intervals act as a soft dead-ball cue (`use_setpiece_dead`) and/or seed the restarting team (`use_setpiece_team`). |
| `pipeline.py` | Labels each graded half independently. Frames outside the graded halves are `X`. |

### Possession rules as implemented

- **Change of possession:** the other team shows sustained, clear control evidence: a 0.4 s mean
  of at least 0.5, beating the holder by 0.2. The change is placed at the start of that evidence
  (🔶 `boundary`, issue #17).
- **No change:** a deflection, a touch in passing, a ball flying past an opponent, a contested
  ball (both teams within a metre of each other), a loose ball, or a missing ball.
- **Restarts:** a dead ball has no owner. The live stretch after it starts UNKNOWN and resolves to
  the first team with clear control, back-filled to the restart.
- **Dead ball:** out of play, or a stoppage shown by several soft cues together. It starts and ends
  at the observed boundary (🔶 offsets, issue #31).

### Assets

| asset | used? |
|---|---|
| tactical ATD | yes, the only required input |
| Smart Tagging | optional (`configs/*.json` → `tagging`), off by default |
| homography, tactical video, broadcast video and ATD | not used in V1 |
| public 5-minute GT | used only to check pipeline correctness (see below), never to choose or tune anything |

## Validation, and how V1 is chosen over the fallback

| layer | what it is | what it is not |
|---|---|---|
| L0 synthetic tests (`pytest`) | proves the code does what the design says on scripted scenarios | **not** evidence of accuracy on the match |
| L1 diagnostics (`scripts/diagnose.py`) | plausibility (possession counts, durations, flicker, dead-ball share) and stability under ball dropout, for V1 and the fallback | not accuracy |
| L2 Smart Tagging consistency | restart timing and team against SET PIECES tags | not ground truth; not independent when any tagging input is on |

`baseline_smooth50`, the official published floor, is kept as an independent fallback
(`--fallback-baseline`, byte-identical to the official file). Submitting V1 rather than the
fallback is an explicit decision taken from L0 together with the L1/L2 evidence. It is never
assumed. The public 5-minute GT plays no part in that decision. It is used only to check that the
pipeline is correct (rows aligned, file well-formed, labels meaning what we think) and never to
compare the scores of two versions.

## Known limitations

- `alive.hold_max_s` = 6 s is a provisional heuristic loosely inspired by the goalkeeper
  six-second rule. It is not a ground-truth boundary. Keeper holds longer than this can be read as
  stoppages, and set pieces taken sooner with the taker beside the ball can be missed.
- Player velocities are finite differences of **tracking-slot** positions. Slots are tracklets, not
  people, so identity switches create velocity spikes. Smoothing and `speed_floor` limit the
  effect but do not remove it.
- A brief exit (seen out of play for 1–2 frames) without an in-play sighting just before it is
  not a trigger.
- Restart detection uses ball speed alone. A ball kicked away during a stoppage can end it early.
- A stoppage longer than `max_dead_s` (120 s) without an observed restart ends at the timeout.
- The SET PIECES Team tag is *assumed* to name the restarting side. The tagging documentation
  does not say so.
- The thresholds are priors, not fitted. Passing synthetic tests shows that the logic is
  implemented as designed, not that it is accurate on this match.

## Results

The numbers come from local runs on the competition data, made outside any cloud LLM session.
They are recorded here by the project lead: the possession count and duration, the L1 stability
figures, the L2 figures with tagging off, and the decision taken between V1 and the fallback.

## Reproduction

### Environment

Python 3.13.2 was used (numpy 2.5.4, pandas 3.0.6, pytest 9.2.0).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

### Tests (synthetic, no data needed)

```bash
.venv/bin/python -m pytest -W error
```

### Data

Unpack the competition bundle as described in `DATA.md`, then point `MTAH_DATA_DIR` at it. The
data is never committed, and `data/` is gitignored.

```bash
export MTAH_DATA_DIR=/path/to/bundle
```

### Generate the submission and the fallback

```bash
.venv/bin/python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_v1
```

```bash
.venv/bin/python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_fb --fallback-baseline
```

Each run writes the CSV and a `manifest.json`. The manifest records the config and its hash, the
git commit, whether the tree was clean, the input sizes, and the output's sha256.

### Diagnostics (aggregates only; check the data licence before sharing them)

```bash
.venv/bin/python scripts/diagnose.py --config configs/v1.json --submission /tmp/pv_v1/submission.csv --baseline /tmp/pv_fb/baseline_smooth50.csv --out /tmp/pv_v1/diagnostics.json
```

### Final checks before uploading

```bash
.venv/bin/python evaluation/validate_submission.py /tmp/pv_v1/submission.csv
```

```bash
.venv/bin/python scripts/verify_submission.py /tmp/pv_v1/manifest.json --strict
```

`--strict` fails unless the run was made from a clean tree at the commit checked out now. That
commit is the one the public repository must hold, and it must regenerate the identical file.

### Reproduce from a clean clone

```bash
git clone --branch pv/v1 <repository-url> pv_clone
```

```bash
cd pv_clone && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-dev.txt && .venv/bin/python -m pytest -W error
```

```bash
.venv/bin/python scripts/run_v1.py --config configs/v1.json --out /tmp/pv_clone && cmp /tmp/pv_clone/submission.csv /tmp/pv_v1/submission.csv && echo identical
```
