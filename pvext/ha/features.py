"""HA1 per-frame features for P(home in possession) (design: 2026-11-06 HA1 protocol).

ATD-only. Every feature is ANTISYMMETRIC under swapping the two teams (team_of_slot 0 <-> 1):
home-minus-away differences, or values signed by the home team's attacking direction (which also
flips when the teams swap). With a scale-only scaler and a logistic regression without intercept,
swapping the teams therefore maps P(home) to exactly 1 - P(home): no home bias, hence no team
identity, can be learned. NaN becomes 0 (which is its own negative), and rows outside the graded
blocks are 0.
"""

import numpy as np

from pitchvision import config
from pitchvision import features as v1_features
from pitchvision.features import _centred_nanmean

# Params
BALL_CARRY_S = 1.0           # carry the last seen ball position for at most this long
MAX_DIST_M = 60.0
RADII_M = (3.0, 5.0, 10.0)
CLOSE_M = 1.5                # "close to the ball" for the last-close-team memory
MEMORY_DECAY_S = 2.0
WINDOWS_S = (0.5, 1.0, 3.0)
SIDE_WINDOW_S = 1.0
HOME, AWAY = 0, 1
# HA3 ball-touch features (feature set "ha3" = the 22 HA1 features + these 5)
TOUCH_VEL_HALF = 2           # ball velocity: central difference over +-2 frames
TOUCH_DV_HALF = 3            # velocity change |v(t+3) - v(t-3)|
TOUCH_DV_MS = 3.0
TOUCH_M = 2.0
TOUCH_DECAY_S = 2.0
TOUCH_BALANCE_S = 3.0
FEATURE_SETS = ("ha1", "ha3")


def compute(tr, pitch, blocks, fps=25, cfg_features=None, feature_set="ha1"):
    """(names, X float32 (T, F)) of antisymmetric features; rows outside `blocks` are 0.
    feature_set "ha3" appends the ball-touch features to the HA1 ones."""
    if feature_set not in FEATURE_SETS:
        raise ValueError("unknown feature set {}".format(feature_set))
    T = len(tr.ball_xy)
    if cfg_features is None:
        cfg_features = config.defaults()["features"]
    length = float(pitch[0])
    cols = {}
    ball = _carry_ball(tr.ball_xy, blocks, int(round(BALL_CARRY_S * fps)))
    present = ~np.isnan(ball[:, 0])

    d = np.linalg.norm(tr.players_xy.astype(np.float64) - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    team = {}
    for side in (HOME, AWAY):
        ds = np.sort(np.where((tr.team_of_slot == side)[None, :], d, np.inf), axis=1)
        nearest = np.minimum(ds[:, 0], MAX_DIST_M) if ds.shape[1] else np.full(T, MAX_DIST_M)
        second = np.minimum(ds[:, 1], MAX_DIST_M) if ds.shape[1] > 1 else np.full(T, MAX_DIST_M)
        counts = [np.sum(ds < r, axis=1).astype(np.float64) for r in RADII_M]
        team[side] = (nearest, second, counts)
    no_ball = ~present
    d1 = np.where(no_ball, np.nan, team[AWAY][0] - team[HOME][0])     # > 0: home closer
    cols["nearest_dist_diff"] = d1
    cols["second_dist_diff"] = np.where(no_ball, np.nan, team[AWAY][1] - team[HOME][1])
    for r, ch, ca in zip(RADII_M, team[HOME][2], team[AWAY][2]):
        cols["within_{:g}m_diff".format(r)] = np.where(no_ball, np.nan, ch - ca)

    v1 = v1_features.compute(tr, cfg_features, fps, pitch)
    cols["v1_rel_speed_diff"] = (np.asarray(v1.rel_speed_a, np.float64)
                                 - np.asarray(v1.rel_speed_h, np.float64))

    sign = _attack_sign(tr, blocks)                    # +1 where home attacks towards +x
    vx = _ball_vx(ball, fps)
    cols["ball_vx_home_attack"] = vx * sign
    cols["ball_x_home_attack"] = np.where(no_ball, np.nan, (ball[:, 0] - length / 2) / (length / 2)) * sign

    close = np.where(no_ball, 0.0,
                     (team[HOME][0] < CLOSE_M).astype(float) - (team[AWAY][0] < CLOSE_M).astype(float))
    sticky, age = _sticky(close, blocks)
    cols["last_close_team"] = sticky
    cols["last_close_team_decayed"] = sticky * np.exp(-age / (MEMORY_DECAY_S * fps))

    for base in ("nearest_dist_diff", "within_3m_diff", "last_close_team"):
        for w in WINDOWS_S:
            cols["{}_mean_{:g}s".format(base, w)] = _centred_nanmean(cols[base], int(round(2 * w * fps)))
    k = int(round(SIDE_WINDOW_S * fps))
    cols["nearest_dist_diff_past_1s"] = _side_mean(cols["nearest_dist_diff"], k, past=True)
    cols["nearest_dist_diff_future_1s"] = _side_mean(cols["nearest_dist_diff"], k, past=False)
    if feature_set == "ha3":
        cols.update(_touch_features(tr, d, blocks, fps))

    names = list(cols)
    X = np.column_stack([np.asarray(cols[n], np.float64) for n in names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    graded = np.zeros(T, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    X[~graded] = 0.0
    return names, X.astype(np.float32)


def _touches(tr, d_carried, blocks, fps):
    """Per-frame touch team (+1 home, -1 away, 0 none) from the RAW ball: a velocity change of at
    least TOUCH_DV_MS with the nearest player within TOUCH_M. A missing ball never makes a touch."""
    T = len(tr.ball_xy)
    raw = tr.ball_xy.astype(np.float64)
    h, k = TOUCH_VEL_HALF, TOUCH_DV_HALF
    v = np.full((T, 2), np.nan)
    if T > 2 * h:
        v[h:T - h] = (raw[2 * h:] - raw[:-2 * h]) / (2 * h / fps)
    dv = np.full(T, np.nan)
    if T > 2 * k:
        dv[k:T - k] = np.linalg.norm(v[2 * k:] - v[:-2 * k], axis=1)
    d = np.where(np.isnan(raw[:, :1]), np.inf, d_carried)     # distances to the raw ball only
    home = np.min(np.where((tr.team_of_slot == HOME)[None, :], d, np.inf), axis=1, initial=np.inf)
    away = np.min(np.where((tr.team_of_slot == AWAY)[None, :], d, np.inf), axis=1, initial=np.inf)
    with np.errstate(invalid="ignore"):
        event = (dv >= TOUCH_DV_MS) & (np.minimum(home, away) <= TOUCH_M) & (home != away)
    out = np.where(event, np.where(home < away, 1.0, -1.0), 0.0)
    graded = np.zeros(T, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    return np.where(graded, out, 0.0)


def _touch_features(tr, d_carried, blocks, fps):
    touch = _touches(tr, d_carried, blocks, fps)
    last, age = _sticky(touch, blocks)
    nxt_r, ahead_r = _sticky(touch[::-1], [(p, len(touch) - 1 - b, len(touch) - 1 - a)
                                          for p, a, b in blocks])
    nxt, ahead = nxt_r[::-1], ahead_r[::-1]
    decay = TOUCH_DECAY_S * fps
    return {"last_touch_team": last, "last_touch_decayed": last * np.exp(-age / decay),
            "next_touch_team": nxt, "next_touch_decayed": nxt * np.exp(-ahead / decay),
            "touch_balance_3s": _centred_nanmean(touch, int(round(TOUCH_BALANCE_S * fps)))}


def _carry_ball(ball_xy, blocks, max_frames):
    out = np.full(ball_xy.shape, np.nan)
    for _, first, last in blocks:
        b = ball_xy[first:last + 1].astype(np.float64)
        known = ~np.isnan(b[:, 0])
        idx = np.where(known, np.arange(len(b)), -1)
        idx = np.maximum.accumulate(idx)
        ok = (idx >= 0) & (np.arange(len(b)) - idx <= max_frames)
        seg = np.full(b.shape, np.nan)
        seg[ok] = b[idx[ok]]
        out[first:last + 1] = seg
    return out


def _attack_sign(tr, blocks):
    """Per block: +1 if the home team's median centroid lies at lower x than the away team's (home
    attacks +x), -1 otherwise, 0 if unknown. Swapping the teams flips it."""
    T = len(tr.ball_xy)
    out = np.zeros(T)
    x = tr.players_xy[..., 0].astype(np.float64)
    for _, first, last in blocks:
        cx = {}
        for side in (HOME, AWAY):
            cols = x[first:last + 1][:, tr.team_of_slot == side]
            with np.errstate(all="ignore"):
                known = ~np.isnan(cols)
                means = np.where(known.any(axis=1), np.nansum(cols, axis=1) / np.maximum(known.sum(axis=1), 1), np.nan)
            cx[side] = np.nanmedian(means) if np.isfinite(means).any() else np.nan
        if np.isfinite(cx[HOME]) and np.isfinite(cx[AWAY]) and cx[HOME] != cx[AWAY]:
            out[first:last + 1] = 1.0 if cx[HOME] < cx[AWAY] else -1.0
    return out


def _ball_vx(ball, fps, half=2):
    T = len(ball)
    vx = np.full(T, np.nan)
    if T > 2 * half:
        vx[half:T - half] = (ball[2 * half:, 0] - ball[:-2 * half, 0]) / (2 * half / fps)
    return np.clip(vx, -40.0, 40.0)


def _sticky(close, blocks):
    """Last non-zero `close` value carried forward inside each block, and frames since it was set."""
    T = len(close)
    sticky, age = np.zeros(T), np.zeros(T)
    for _, first, last in blocks:
        c = close[first:last + 1]
        idx = np.where(c != 0, np.arange(len(c)), -1)
        idx = np.maximum.accumulate(idx)
        have = idx >= 0
        sticky[first:last + 1] = np.where(have, c[np.maximum(idx, 0)], 0.0)
        age[first:last + 1] = np.where(have, np.arange(len(c)) - idx, 0.0)
    return sticky, age


def _side_mean(values, k, past):
    """Mean of the non-NaN values over the k frames strictly before (past) or after (future)."""
    v = np.asarray(values, np.float64)
    known = ~np.isnan(v)
    sums = np.concatenate(([0.0], np.cumsum(np.where(known, v, 0.0))))
    counts = np.concatenate(([0], np.cumsum(known)))
    i = np.arange(len(v))
    lo, hi = (np.maximum(0, i - k), i) if past else (i + 1, np.minimum(len(v), i + 1 + k))
    lo = np.minimum(lo, len(v))
    n = counts[hi] - counts[lo]
    out = np.full(len(v), np.nan)
    np.divide(sums[hi] - sums[lo], n, out=out, where=n > 0)
    return out
