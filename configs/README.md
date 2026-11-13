# V1 configuration

`v1.json` holds every threshold. None of them was fitted to match data. Each one is a physical or
football prior, written down here so that any change to it is a deliberate edit with a stated
reason. 🔶 marks a convention that is waiting on an organiser answer.

## features
| key | default | prior |
|---|---|---|
| `ball_speed_smooth_frames` | 5 | 0.2 s at 25 fps: removes per-frame detection jitter, keeps a pass's acceleration |
| `out_margin_m` | 1.0 | tracking noise near the lines is well under a metre; beyond that the ball is out |
| `exit_near_m` | 3.0 | a brief out-of-play sighting is a real exit only if the ball was last seen in play this close to a line |
| `exit_lookback_s` | 0.4 | ...and that in-play sighting is at most this recent |
| `max_ball_speed_mps` | 40.0 | above the hardest shots (~35 m/s): a faster implied move between the two sightings is a detection jump |

## alive
| key | default | prior |
|---|---|---|
| `w_static` | 0.35 | a static ball **nobody holds** suggests a set-piece setup |
| `w_motion` | 0.35 | a slow field suggests a stoppage, but so does a slow build-up |
| `w_gap` | 0.3 | a missing ball is weak evidence: 41 % of graded frames have no ball, most of them in play |
| `w_tag` | 0.2 | Smart Tagging set-piece window: automatic and noisy, so a nudge only |
| `static_speed_mps` | 0.5 | below a walking roll; a dribbled ball moves faster |
| `static_min_s` | 1.0 | a ball at rest for a second is no longer a touch in passing |
| `motion_low_mps` | 0.8 | median player speed during a stoppage (walking) |
| `motion_high_mps` | 1.8 | median player speed in open play (jogging and up) |
| `gap_scale_s` | 4.0 | a gap reaches full weight only after several seconds, longer than an aerial ball |
| `smooth_s` | 1.0 | evidence is averaged over a second before thresholding |
| `dead_on` | 0.6 | above any single soft cue (max 0.35), so a soft stoppage needs two of them; an out-of-bounds ball triggers on its own |
| `r_hold_m` | 1.0 | a still ball this close to a player may be held (goalkeeper with the ball in hand) |
| `hold_max_s` | 6.0 | **provisional heuristic**, not a ground-truth boundary: loosely inspired by the six-second goalkeeper rule (which referees rarely enforce to the second), a held still spell longer than this is treated as a set-piece wait or a pre-kick-off rather than a keeper's hold. Revisit with L1 diagnostics |
| `out_confirm_frames` | 3 | out-of-play sightings this many frames in a row need no further confirmation |
| `min_dead_s` | 2.0 | soft evidence must last this long to trigger, and any stoppage shorter than this is a blip: the fastest restarts (quick throw-ins) still take a couple of seconds |
| `min_alive_s` | 1.0 | live play shorter than a second between two stoppages is noise |
| `max_dead_s` | 120.0 | safety valve when no restart is ever observed; long enough for injuries and VAR checks. After it fires, a trigger must switch off before it can start DEAD again |
| `restart_speed_mps` | 3.0 | a restart puts the ball in motion faster than it is rolled or carried into place |
| `restart_min_s` | 0.2 | the ball must keep moving in play for 5 frames, so one jittery detection is not a restart |
| 🔶 `dead_start_offset_frames` | 0 | issue #31: whether D starts at the ball crossing the line or at the whistle |
| 🔶 `dead_end_offset_frames` | 0 | issue #31: whether D ends at the restart touch |

## possession
| key | default | prior |
|---|---|---|
| `r_ctrl_m` | 1.5 | a player can play a ball within about a stride |
| `r_decay_m` | 1.5 | evidence fades to zero at 3 m: out of playing distance |
| `margin_m` | 1.0 | when both teams are within a metre of each other, the ball is contested, not controlled |
| `v_ref_mps` | 15.0 | a firm pass; relative ball speed at this level halves the motion factor |
| `speed_floor` | 0.3 | speed never vetoes control: fast receptions and interceptions keep 30 % of their evidence |
| `win_s` | 0.4 | evidence is accumulated over 0.4 s, longer than a deflection, shorter than a reception |
| `evidence_on` | 0.2 | frames above this belong to an evidence run (used to back-date a switch) |
| `switch_on` | 0.5 | a switch needs sustained, clear evidence |
| `switch_margin` | 0.2 | and it must beat the current holder's evidence clearly |
| `init_on` | 0.3 | after a restart there is no holder to defend, so a lower bar resolves UNKNOWN |
| 🔶 `boundary` | `evidence_start` | issue #17: back-date a switch to the start of its evidence, or place it at confirmation (`confirmed`) |

## tagging
| key | default | prior |
|---|---|---|
| `use_setpiece_dead` | false | SET PIECES intervals add `alive.w_tag` (0.2) to the soft dead-ball evidence: a nudge that only helps when another cue is present, never a stoppage on its own |
| `use_setpiece_team` | false | the tagged team seeds possession at a restart inside a SET PIECES interval (N/A, e.g. every throw-in, seeds nothing); the normal switch rule overturns it on clear control. Assumes the Team tag names the restarting side, which is not documented |

Both are off by default and independent of each other. With both off the tagging is never read.
When either is on, L2 diagnostics mark **both** consistency figures `independent: false`, because either cue can move either figure.

## Known model limitations
- `alive.hold_max_s` (6 s) is a provisional heuristic, not a ground-truth boundary. Keeper holds
  longer than it look like set-piece waits; set pieces taken sooner with the taker beside the ball
  look like keeper holds.
- Player velocities (`features.rel_speed_*`) are finite differences of a **tracking slot's**
  position. Slots are tracklets, not people: when a slot switches identity or the nearest slot
  changes between frames, the estimated relative speed jumps. It is smoothed over 5 frames and
  can only lower control evidence (never below `speed_floor`), but it is not a per-player
  velocity.
