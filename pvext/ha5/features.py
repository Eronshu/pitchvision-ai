"""HA5a features (pre-registration: docs/superpowers/specs/2026-11-13-pitchvision-ha5a-design.md).

`ha5a` = HA3's 26 antisymmetric features (frozen pvext.ha.features, unchanged) + 12 new ones:
six team-invariant observation-reliability features (S1-S6) and six antisymmetric, fresh-only,
short-horizon, slot-free control features (A1-A6). `anti_mask()` says which columns flip sign under
a team swap; the model mirrors only those. Rows outside the graded blocks are 0.
"""

import numpy as np

from pvext.ha import features as ha_features

# Params (fixed by the pre-registration)
AGE_CAP_S = 5.0
PAST_MISSING_FRAMES = 25
NEAREST_CAP_M = 30.0
CLOSE_M = 1.5
SOFT_SCALE_M = 1.5
SIDE_FRAMES = 10
CLOSE_DECAY_S = 0.5
REAPPEAR_MIN_GAP = 10
REAPPEAR_M = 3.0
REAPPEAR_DECAY_S = 1.0
VX_HALF = 2
VX_CLIP = 40.0
HOME, AWAY = 0, 1

EXTRA = (("ball_fresh", False), ("ball_obs_age_s", False), ("ball_missing_share_past_1s", False),
         ("any_nearest_m", False), ("close_event_age_s", False), ("touch_event_age_s", False),
         ("fresh_soft_control", True), ("fresh_soft_control_past_0.4s", True),
         ("fresh_soft_control_future_0.4s", True), ("fresh_close_team_decay_0.5s", True),
         ("reappear_team_decay_1s", True), ("fresh_ball_vx_home_attack", True))


def _graded(T, blocks):
    g = np.zeros(T, dtype=bool)
    for _, a, b in blocks:
        g[a:b + 1] = True
    return g


def _age_since(event, blocks, fps, before_first):
    """Seconds since the last True of `event` inside each block (0 at the event), capped."""
    T = len(event)
    out = np.full(T, before_first, dtype=np.float64)
    for _, a, b in blocks:
        e = event[a:b + 1]
        idx = np.maximum.accumulate(np.where(e, np.arange(len(e)), -1))
        out[a:b + 1] = np.where(idx >= 0, (np.arange(len(e)) - idx) / fps, before_first)
    return np.minimum(out, AGE_CAP_S)


def _team_min(d, team_of_slot, side):
    cols = team_of_slot == side
    if not cols.any():
        return np.full(len(d), np.inf)
    return np.min(np.where(np.isnan(d[:, cols]), np.inf, d[:, cols]), axis=1)


def extra(tr, pitch, blocks, fps=25):
    """(names, X float64 (T, 12)) of the new HA5a features."""
    T = len(tr.ball_xy)
    raw_xy = tr.ball_xy.astype(np.float64)
    fresh = ~np.isnan(raw_xy[:, 0])
    carried = ha_features._carry_ball(tr.ball_xy, blocks, int(round(ha_features.BALL_CARRY_S * fps)))
    players = tr.players_xy.astype(np.float64)
    cols = {}

    cols["ball_fresh"] = fresh.astype(np.float64)
    cols["ball_obs_age_s"] = _age_since(fresh, blocks, fps, AGE_CAP_S)
    missing = (~fresh).astype(np.float64)
    past = ha_features._side_mean(np.where(_graded(T, blocks), missing, np.nan), PAST_MISSING_FRAMES, past=True)
    cols["ball_missing_share_past_1s"] = np.where(np.isnan(past), 1.0, past)

    d_car = np.linalg.norm(players - carried[:, None, :], axis=2)
    any_near = np.min(np.where(np.isnan(d_car), np.inf, d_car), axis=1)
    cols["any_nearest_m"] = np.minimum(np.where(np.isfinite(any_near), any_near, NEAREST_CAP_M), NEAREST_CAP_M)

    d_raw = np.linalg.norm(players - raw_xy[:, None, :], axis=2)          # NaN without a fresh ball
    h_min = _team_min(d_raw, tr.team_of_slot, HOME)
    a_min = _team_min(d_raw, tr.team_of_slot, AWAY)
    h_close, a_close = h_min <= CLOSE_M, a_min <= CLOSE_M
    close = np.where(fresh & (h_close ^ a_close), np.where(h_close, 1.0, -1.0), 0.0)
    cols["close_event_age_s"] = _age_since(close != 0, blocks, fps, 0.0)

    touch = ha_features._touches(tr, d_car, blocks, fps)
    cols["touch_event_age_s"] = _age_since(touch != 0, blocks, fps, 0.0)

    w = np.where(np.isnan(d_raw), 0.0, np.exp(-np.nan_to_num(d_raw, nan=0.0) / SOFT_SCALE_M))
    home = tr.team_of_slot == HOME
    away = tr.team_of_slot == AWAY
    soft = np.where(fresh, w[:, home].sum(axis=1) - w[:, away].sum(axis=1), 0.0)
    cols["fresh_soft_control"] = soft
    soft_nan = np.where(fresh, soft, np.nan)
    for name, is_past in (("fresh_soft_control_past_0.4s", True), ("fresh_soft_control_future_0.4s", False)):
        v = ha_features._side_mean(soft_nan, SIDE_FRAMES, past=is_past)
        cols[name] = np.where(np.isnan(v), 0.0, v)

    sticky, age = ha_features._sticky(close, blocks)
    cols["fresh_close_team_decay_0.5s"] = sticky * np.exp(-age / (CLOSE_DECAY_S * fps))

    reappear = np.zeros(T)
    for _, a, b in blocks:
        f = fresh[a:b + 1]
        miss_run = np.zeros(len(f), dtype=np.int64)
        run = 0
        for i in range(len(f)):
            miss_run[i] = run
            run = 0 if f[i] else run + 1
        idx = np.flatnonzero(f & (miss_run >= REAPPEAR_MIN_GAP))
        for i in idx:
            t = a + i
            hm, am = h_min[t], a_min[t]
            if min(hm, am) <= REAPPEAR_M and hm != am:
                reappear[t] = 1.0 if hm < am else -1.0
    sticky_r, age_r = ha_features._sticky(reappear, blocks)
    cols["reappear_team_decay_1s"] = sticky_r * np.exp(-age_r / (REAPPEAR_DECAY_S * fps))

    sign = ha_features._attack_sign(tr, blocks)
    vx = np.zeros(T)
    h = VX_HALF
    if T > 2 * h:
        ok = fresh[2 * h:] & fresh[:-2 * h]
        vx[h:T - h] = np.where(ok, (raw_xy[2 * h:, 0] - raw_xy[:-2 * h, 0]) / (2 * h / fps), 0.0)
    cols["fresh_ball_vx_home_attack"] = np.clip(np.nan_to_num(vx, nan=0.0), -VX_CLIP, VX_CLIP) * sign

    names = [n for n, _ in EXTRA]
    X = np.column_stack([np.asarray(cols[n], np.float64) for n in names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[~_graded(T, blocks)] = 0.0
    return names, X


def compute(tr, pitch, blocks, fps=25):
    """(names, X float32 (T, 38), anti_mask) for feature set ha5a."""
    names_h, Xh = ha_features.compute(tr, pitch, blocks, fps, feature_set="ha3")
    names_e, Xe = extra(tr, pitch, blocks, fps)
    return names_h + names_e, np.column_stack([Xh, Xe.astype(np.float32)]), anti_mask(len(names_h))


def anti_mask(n_ha3=26):
    return np.array([True] * n_ha3 + [anti for _, anti in EXTRA], dtype=bool)
