"""Per-frame evidence computed from a Tracking. Pure numpy, NaN-safe, no warnings.

Everything downstream reads these arrays and nothing else from the tracking, so this is the one
place that decides what "no ball" or "no player" turns into: an infinite distance, a NaN speed,
and never an exception.
"""

import collections

import numpy as np

HOME, AWAY = 0, 1

Features = collections.namedtuple("Features", [
    "dist_h", "dist_a", "rel_speed_h", "rel_speed_a", "ball_speed",
    "ball_out", "exit_plausible", "players_motion", "gap_len"])


def compute(tr, cfg_features, fps, pitch):
    """Features for every frame of `tr`."""
    window = int(cfg_features["ball_speed_smooth_frames"])
    present = tr.ball_present

    ball_velocity = _velocity(tr.ball_xy, fps)
    ball_speed = _centred_nanmean(np.linalg.norm(ball_velocity, axis=1), window)
    ball_speed[~present] = np.nan

    dist, rel = {}, {}
    for team in (HOME, AWAY):
        dist[team], rel[team] = _nearest_of_team(tr, team, ball_velocity, ball_speed, window, fps)

    ball_out = _out_of_bounds(tr.ball_xy, present, pitch, cfg_features["out_margin_m"])
    return Features(
        dist_h=dist[HOME], dist_a=dist[AWAY],
        rel_speed_h=rel[HOME], rel_speed_a=rel[AWAY],
        ball_speed=ball_speed.astype(np.float32),
        ball_out=ball_out,
        exit_plausible=_exit_plausible(tr.ball_xy, present, ball_out, pitch, cfg_features, fps),
        players_motion=_nanmedian_rows(tr.players_speed),
        gap_len=_gap_len(present),
    )


def _velocity(xy, fps):
    """(T, 2) finite-difference velocity in m/s; NaN where this or the previous frame is missing."""
    out = np.full(xy.shape, np.nan, dtype=np.float64)
    out[1:] = (xy[1:].astype(np.float64) - xy[:-1]) * fps
    return out


def _nearest_of_team(tr, team, ball_velocity, ball_speed, window, fps):
    """(distance, relative speed) between the ball and `team`'s nearest slot, per frame.

    Relative speed is |ball velocity - that slot's velocity|. It is low when a player runs with
    the ball, which is what lets a fast dribble count as control. Where either velocity is
    unknown, it falls back to the ball's own speed.
    """
    n = len(tr.ball_xy)
    d2 = ((tr.players_xy - tr.ball_xy[:, None, :]) ** 2).sum(axis=2)
    d2 = np.where(np.isnan(d2) | (tr.team_of_slot != team)[None, :], np.inf, d2)
    slot = np.argmin(d2, axis=1)
    rows = np.arange(n)
    dist = np.sqrt(d2[rows, slot]).astype(np.float32)

    player_velocity = np.full((n, 2), np.nan)
    player_velocity[1:] = (tr.players_xy[rows[1:], slot[1:]].astype(np.float64)
                           - tr.players_xy[rows[1:] - 1, slot[1:]]) * fps
    rel = np.linalg.norm(ball_velocity - player_velocity, axis=1)
    rel[np.isinf(dist)] = np.nan
    rel = _centred_nanmean(rel, window)
    rel = np.where(np.isnan(rel), ball_speed, rel)
    rel[~tr.ball_present] = np.nan
    return dist, rel.astype(np.float32)


def _centred_nanmean(x, window):
    """Mean of the non-NaN values in a centred window of `window` frames; NaN if there are none."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    known = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(known, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(known)))
    index = np.arange(len(x))
    low = np.maximum(0, index - window // 2)
    high = np.minimum(len(x), index + window - window // 2)
    count = counts[high] - counts[low]
    out = np.full(len(x), np.nan)
    np.divide(sums[high] - sums[low], count, out=out, where=count > 0)
    return out


def _nanmedian_rows(values):
    """Row-wise median of the non-NaN entries; NaN for an all-NaN row, without a warning."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    rows = ~np.isnan(values).all(axis=1)
    if rows.any():
        out[rows] = np.nanmedian(values[rows], axis=1)
    return out


def _out_of_bounds(ball_xy, present, pitch, margin):
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
    return out & present


def _exit_plausible(ball_xy, present, ball_out, pitch, cfg_features, fps):
    """True on an out-of-bounds frame that a real exit could explain.

    That needs an in-play sighting within `exit_lookback_s` before it, no more than
    `exit_near_m` from a line, and a ball speed between the two below `max_ball_speed_mps`.
    A detection that jumps out of play from mid-pitch, or appears out of play with no sighting
    just before it, fails the test. Sustained out-of-play sightings are judged elsewhere.
    """
    length, width = pitch
    n = len(ball_xy)
    in_play = present & ~ball_out
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(in_play, index, -1))
    previous = np.concatenate(([-1], last[:-1]))   # strictly before t
    lookback = int(round(cfg_features["exit_lookback_s"] * fps))
    ok = ball_out & (previous >= 0) & (index - previous <= lookback)

    out = np.zeros(n, dtype=bool)
    t = np.flatnonzero(ok)
    if len(t):
        p = previous[t]
        x, y = ball_xy[p, 0], ball_xy[p, 1]
        near = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y)) \
            <= cfg_features["exit_near_m"]
        speed = np.linalg.norm(ball_xy[t] - ball_xy[p], axis=1) * fps / (t - p)
        out[t] = near & (speed <= cfg_features["max_ball_speed_mps"])
    return out


def _gap_len(present):
    """Length of the missing-ball run each frame sits in; 0 where the ball is present."""
    out = np.zeros(len(present), dtype=np.int32)
    edges = np.diff(np.concatenate(([0], (~present).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out
