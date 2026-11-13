"""ATD-only, slot-invariant features for the V2 Alive/Dead model.

The only input is a pitchvision.atd.Tracking, i.e. exactly what V1's frozen loader reads from a
competition-format ATD file: player slot positions, slot speeds (m/s) and slot teams, and the ball
position or its absence. Nothing else is available on competition data, so nothing else is used: no
ball height, no ATD ball "speed" field (atd.load_arrays never keeps it), no DFL attribute and no label.

Slots are tracklets, not people, so every feature aggregates over slots and cannot depend on slot
order or identity. Ball speed comes from positions. Frozen V1 contributes two signals through its
own code: the soft dead-ball score and its dead mask, computed per graded block exactly as the
pipeline does. Frames outside the graded blocks are all zero.
"""

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pitchvision import alive, config
from pitchvision import features as v1_features
from pitchvision.features import _centred_nanmean

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1_CONFIG = os.path.join(ROOT, "configs", "v1.json")

# Params
MAX_PLAYER_SPEED = 12.0      # m/s; faster position steps are tracklet jumps, not running
MOVING_MPS = 2.0
NEAR_BALL_M = 10.0
NEAR_LINE_M = 3.0
WINDOWS_S = (1, 3, 10)
GOAL_AREA = (5.5, 9.16)      # depth, half-width (m)
PENALTY_AREA = (16.5, 20.16)


def compute(tr, pitch, blocks, fps=25):
    """(names, X) with X float32 of shape (T, F); rows outside `blocks` are 0."""
    cfg = config.load(V1_CONFIG)
    length, width = pitch
    T = len(tr.ball_xy)
    cols = {}

    # -- ball -------------------------------------------------------------------------------------
    v1 = v1_features.compute(tr, cfg["features"], fps, pitch)
    present = tr.ball_present
    bx, by = tr.ball_xy[:, 0].astype(np.float64), tr.ball_xy[:, 1].astype(np.float64)
    speed = np.where(present, v1.ball_speed, np.nan)
    cols["ball_present"] = present.astype(float)
    cols["ball_gap_s"] = np.minimum(v1.gap_len / fps, 30.0)
    cols["ball_speed_pos"] = speed
    acc = np.full(T, np.nan)
    acc[1:] = np.abs(np.diff(speed)) * fps
    cols["ball_accel_pos"] = _centred_nanmean(acc, 5)
    cols["ball_static"] = (speed < cfg["alive"]["static_speed_mps"]).astype(float)
    cols["ball_out"] = v1.ball_out.astype(float)
    cols["ball_exit_plausible"] = v1.exit_plausible.astype(float)
    to_goal_line = np.minimum(bx, length - bx)
    to_touchline = np.minimum(by, width - by)
    cols["ball_to_touchline"] = to_touchline
    cols["ball_to_goal_line"] = to_goal_line
    cols["ball_to_centre"] = np.hypot(bx - length / 2, by - width / 2)
    cols["ball_in_goal_area"] = ((to_goal_line < GOAL_AREA[0])
                                 & (np.abs(by - width / 2) < GOAL_AREA[1])).astype(float)
    cols["ball_near_corner"] = ((to_goal_line < NEAR_LINE_M)
                                & (to_touchline < NEAR_LINE_M)).astype(float)

    # -- field motion -----------------------------------------------------------------------------
    xy = tr.players_xy.astype(np.float64)
    seen = ~np.isnan(xy[..., 0])
    step = np.full(seen.shape, np.nan)
    step[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=2) * fps
    step[step > MAX_PLAYER_SPEED] = np.nan
    visible = seen.sum(axis=1)
    cols["players_visible"] = visible.astype(float)
    pos_median = _row_quantile(step, 0.5)
    cols["players_speed_pos_p50"] = pos_median
    cols["players_speed_pos_p90"] = _row_quantile(step, 0.9)
    moving = np.nansum(np.where(np.isnan(step), 0.0, step > MOVING_MPS), axis=1)
    counted = (~np.isnan(step)).sum(axis=1)
    cols["players_share_moving"] = _safe_div(moving, counted)
    cols["players_speed_slot_p50"] = v1.players_motion.astype(np.float64)
    cols["players_speed_slot_p90"] = _row_quantile(tr.players_speed.astype(np.float64), 0.9)
    filled = _fill_forward(pos_median)
    for seconds in (1, 3):
        k = seconds * fps
        cols["decel_{}s".format(seconds)] = _past_mean(filled, k) - _future_mean(filled, k)

    # -- shape ------------------------------------------------------------------------------------
    team = tr.team_of_slot
    for side, name in ((0, "home"), (1, "away")):
        on = seen & (team == side)[None, :]
        n = on.sum(axis=1)
        x = np.where(on, xy[..., 0], 0.0)
        cx = _safe_div(x.sum(axis=1), n)
        cols["{}_centroid_x_rel".format(name)] = cx / length
        sq = np.where(on, (xy[..., 0] - cx[:, None]) ** 2, 0.0)
        cols["{}_spread_x".format(name)] = np.sqrt(_safe_div(sq.sum(axis=1), n))
    cols["centroid_gap_x"] = np.abs(cols["home_centroid_x_rel"] - cols["away_centroid_x_rel"]) * length
    left = seen & (xy[..., 0] < length / 2)
    right = seen & (xy[..., 0] >= length / 2)
    home, away = (team == 0)[None, :], (team == 1)[None, :]
    split_a = (left & home).sum(axis=1) + (right & away).sum(axis=1)
    split_b = (right & home).sum(axis=1) + (left & away).sum(axis=1)
    cols["own_half_split"] = _safe_div(np.maximum(split_a, split_b), visible)
    in_box_y = np.abs(xy[..., 1] - width / 2) < PENALTY_AREA[1]
    box_left = (seen & in_box_y & (xy[..., 0] < PENALTY_AREA[0])).sum(axis=1)
    box_right = (seen & in_box_y & (xy[..., 0] > length - PENALTY_AREA[0])).sum(axis=1)
    cols["players_in_box_max"] = np.maximum(box_left, box_right).astype(float)
    cols["players_in_boxes"] = (box_left + box_right).astype(float)
    to_ball = np.linalg.norm(xy - tr.ball_xy[:, None, :].astype(np.float64), axis=2)
    with np.errstate(invalid="ignore"):
        cols["players_near_ball"] = (to_ball < NEAR_BALL_M).sum(axis=1).astype(float)
    nearest = np.minimum(v1.dist_h, v1.dist_a).astype(np.float64)
    cols["nearest_player_to_ball"] = np.where(np.isinf(nearest), np.nan, nearest)
    to_line = np.minimum(xy[..., 1], width - xy[..., 1])
    with np.errstate(invalid="ignore"):
        cols["players_near_touchline"] = (seen & (to_line < NEAR_LINE_M)).sum(axis=1).astype(float)

    # -- frozen V1 signals, per block exactly as the pipeline computes them ------------------------
    score = np.zeros(T)
    mask = np.zeros(T)
    for _, first, last in blocks:
        span = slice(first, last + 1)
        block = v1_features.Features(*(array[span] for array in v1))
        score[span] = alive.dead_score(block, cfg["alive"], fps)
        mask[span] = alive.dead_mask(block, cfg["alive"], fps)
    cols["v1_dead_score"] = score
    cols["v1_dead_mask"] = mask

    # -- temporal context -------------------------------------------------------------------------
    for base in ("ball_present", "ball_speed_pos", "ball_static", "players_speed_pos_p50",
                 "players_share_moving", "players_near_ball", "v1_dead_score", "decel_1s"):
        for seconds in WINDOWS_S:
            cols["{}_mean_{}s".format(base, seconds)] = _centred_nanmean(
                np.asarray(cols[base], dtype=np.float64), 2 * seconds * fps + 1)
    for base in ("v1_dead_score", "ball_out"):
        cols["{}_max_3s".format(base)] = _centred_max(np.asarray(cols[base]), 3 * fps)

    names = list(cols)
    X = np.column_stack([np.asarray(cols[n], dtype=np.float64) for n in names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    graded = np.zeros(T, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    X[~graded] = 0.0
    return names, X.astype(np.float32)


def _row_quantile(values, q):
    """Per-row quantile of the non-NaN entries (lower interpolation); NaN for empty rows."""
    ordered = np.sort(values, axis=1)                  # NaN sorts last
    n = (~np.isnan(values)).sum(axis=1)
    index = np.clip(np.floor(q * (n - 1)).astype(int), 0, values.shape[1] - 1)
    out = ordered[np.arange(len(values)), index]
    out[n == 0] = np.nan
    return out


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _fill_forward(values):
    out = np.asarray(values, dtype=np.float64).copy()
    known = ~np.isnan(out)
    if not known.any():
        return np.zeros_like(out)
    index = np.maximum.accumulate(np.where(known, np.arange(len(out)), -1))
    first = np.flatnonzero(known)[0]
    index[index < 0] = first
    return out[index]


def _past_mean(values, k):
    """Mean of values[t-k:t] (strictly before t), shrinking at the start."""
    sums = np.concatenate(([0.0], np.cumsum(values)))
    t = np.arange(len(values))
    low = np.maximum(0, t - k)
    count = t - low
    out = np.zeros(len(values))
    np.divide(sums[t] - sums[low], count, out=out, where=count > 0)
    return out


def _future_mean(values, k):
    """Mean of values[t:t+k] (from t on), shrinking at the end."""
    sums = np.concatenate(([0.0], np.cumsum(values)))
    t = np.arange(len(values))
    high = np.minimum(len(values), t + k)
    return (sums[high] - sums[t]) / (high - t)


def _centred_max(values, half):
    padded = np.pad(np.nan_to_num(values, nan=0.0), half, mode="edge")
    return sliding_window_view(padded, 2 * half + 1).max(axis=1)
