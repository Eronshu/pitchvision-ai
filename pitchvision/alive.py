"""ALIVE / DEAD: triggered stoppages that persist until there is restart evidence.

A spurious D splits a possession in two under spellF1, and a missed one merges two, so both
directions matter. The decision is a two-state machine:

  ALIVE -> DEAD  on a trigger:
      * the ball observed out of play (beyond `out_margin_m`) for `out_confirm_frames` frames in a
        row, or for fewer when a real exit explains it (features.exit_plausible: seen in play
        near a line just before, at a possible ball speed). A lone detection that jumps out of
        play from mid-pitch is noise, however long the ball stays lost afterwards; or
      * soft stoppage evidence sustained for `min_dead_s`: a weighted sum of a ball at rest, a
        slow field and a missing ball, where no single cue reaches `dead_on`. A still ball within
        `r_hold_m` of a player counts as at rest only if the spell outlasts `hold_max_s` (a
        provisional heuristic: a goalkeeper's hold is live play; a set-piece wait is not), and a slow field alone never
        means a stoppage.
  DEAD -> ALIVE  only on restart evidence: the ball observed in play and moving faster than
      `restart_speed_mps` for `restart_min_s`; or after `max_dead_s` as a safety valve. A trigger
      only re-arms after it has been off, so neither exit re-enters on evidence that was already
      there, and the live stretch a timeout opens is never absorbed back into DEAD.

The tracking often loses the ball right after it leaves the pitch, so the stoppage is carried by
the state, not by the frames that happen to show it. Afterwards: DEAD runs shorter than
`min_dead_s` (a trigger followed at once by a restart: a tracking blip) are dropped, live blips
shorter than `min_alive_s` between two stoppages are absorbed, and the 🔶 boundary offsets apply.
"""

import numpy as np

from pitchvision.features import _centred_nanmean


def dead_score(feat, cfg_alive, fps, tag_dead=None):
    """Per-frame soft stoppage evidence in [0, 1], before smoothing. Out-of-bounds is not in it:
    that is a trigger of its own."""
    c = cfg_alive
    static = _long_runs(feat.ball_speed < c["static_speed_mps"],
                        int(round(c["static_min_s"] * fps)))
    unheld_static = _resting_ball(static, np.minimum(feat.dist_h, feat.dist_a) <= c["r_hold_m"],
                                  int(round(c["hold_max_s"] * fps)))
    motion = np.where(np.isnan(feat.players_motion), np.inf, feat.players_motion)
    low_motion = np.clip((c["motion_high_mps"] - motion)
                         / (c["motion_high_mps"] - c["motion_low_mps"]), 0.0, 1.0)
    gap = np.minimum(1.0, feat.gap_len / (c["gap_scale_s"] * fps))

    score = c["w_static"] * unheld_static + c["w_motion"] * low_motion + c["w_gap"] * gap
    if tag_dead is not None:
        score = score + c["w_tag"] * np.asarray(tag_dead, dtype=np.float64)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def dead_mask(feat, cfg_alive, fps, tag_dead=None):
    """True where the ball is dead."""
    c = cfg_alive
    n = len(feat.ball_out)
    min_dead = int(round(c["min_dead_s"] * fps))

    smoothed = _centred_nanmean(dead_score(feat, c, fps, tag_dead).astype(np.float64),
                                int(round(c["smooth_s"] * fps)))
    soft = _long_runs(smoothed >= c["dead_on"], min_dead)
    out = np.asarray(feat.ball_out, dtype=bool)
    out = _long_runs(out, int(c["out_confirm_frames"])) | (out & feat.exit_plausible)
    restart = _restart_starts(feat, c, fps)
    max_dead = int(round(c["max_dead_s"] * fps))

    # A trigger re-arms only after it has been off: an exit never re-enters on the evidence that
    # was already there. After a restart this applies to the soft trigger; after a timeout to
    # both, so a timeout always ends DEAD and the live stretch it opens is kept.
    dead = np.zeros(n, dtype=bool)
    state, since = False, 0
    soft_armed = out_armed = True
    timed_out = set()
    for t in range(n):
        soft_armed = soft_armed or not soft[t]
        out_armed = out_armed or not out[t]
        if state:
            if restart[t]:
                state, soft_armed = False, False
            elif t - since >= max_dead:
                state, soft_armed, out_armed = False, False, False
                timed_out.add(t)
        elif (soft[t] and soft_armed) or (out[t] and out_armed):
            state, since = True, t
        dead[t] = state

    for start, stop in _runs(dead):
        if stop - start < min_dead:
            dead[start:stop] = False

    min_alive = int(round(c["min_alive_s"] * fps))
    for start, stop in _runs(~dead):
        if 0 < start and stop < n and stop - start < min_alive and start not in timed_out:
            dead[start:stop] = True

    return _shift_runs(dead, int(c["dead_start_offset_frames"]), int(c["dead_end_offset_frames"]))


def _restart_starts(feat, c, fps):
    """True at t when the ball is in play and moving from t on for `restart_min_s`."""
    with np.errstate(invalid="ignore"):
        moving = ~np.asarray(feat.ball_out, dtype=bool) & (feat.ball_speed >= c["restart_speed_mps"])
    k = max(1, int(round(c["restart_min_s"] * fps)))
    counts = np.concatenate(([0], np.cumsum(moving)))
    index = np.arange(len(moving))
    high = np.minimum(len(moving), index + k)
    return moving & (counts[high] - counts[index] == high - index) & (high - index == k)


def _resting_ball(static, held, hold_max):
    """Static frames that count as a ball at rest for a restart.

    A static ball nobody is near counts. A static ball a player is near counts only when the
    whole still spell outlasts `hold_max`, a provisional heuristic loosely inspired by the
    goalkeeper six-second rule, not a reliable boundary: a goalkeeper usually holds the ball
    briefly in live play, while a taker standing over a set piece, or a kicker on the spot
    before kick-off, keeps the ball still for longer. Such a spell counts from its first frame.
    """
    out = static & ~held
    for start, stop in _runs(static):
        if stop - start >= hold_max:
            out[start:stop] = True
    return out


def _runs(mask):
    """[(start, stop)] half-open spans where mask is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _long_runs(mask, min_len):
    """mask, keeping only True runs of at least min_len frames."""
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in _runs(mask):
        if stop - start >= min_len:
            out[start:stop] = True
    return out


def _shift_runs(dead, start_offset, end_offset):
    """Move every DEAD run's start by start_offset frames and its end by end_offset frames."""
    if not start_offset and not end_offset:
        return dead
    out = np.zeros(len(dead), dtype=bool)
    for start, stop in _runs(dead):
        new_start = max(0, start + start_offset)
        new_stop = min(len(dead), stop + end_offset)
        if new_stop > new_start:
            out[new_start:new_stop] = True
    return out
