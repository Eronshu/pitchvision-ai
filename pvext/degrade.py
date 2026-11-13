"""Seeded degradation: make clean DFL tracking look like the competition's automatic ATD.

IDSSE positions are near perfect (all 22 players every frame, the ball always located), the ATD is
not (DATA.md: the ball is located on 59.3 % of graded frames, about 20.5 tracked objects, 46 slots
that are tracklets rather than people). apply() closes that gap deterministically from a seed:

  1. player occlusion gaps     two-state Markov per person (share `player_missing`, mean length)
  2. smooth position noise     AR(1) per coordinate (rho `noise_rho`), not per-frame jitter, so
                               it does not invent speed
  3. tracklets -> slots        each visible stretch is cut into tracklets of geometric length
                               (mean `tracklet_mean_s`), each put in a free slot of its team out
                               of `n_slots`; identities do not persist
  4. bursty ball dropout       two-state Markov with share `ball_present` and mean gap
                               `ball_gap_s`, `hard_weight` times likelier when the ball is high
                               (z > 1.5 m) or crowded (`crowd_k` players within `crowd_radius_m`)
  5. false ball detections     single frames displaced 5-30 m, at `false_ball_rate`

The ball height z is tracking, not a label, and it is used only here. apply() has no access to
BallStatus or BallPossession, and no preset is tuned on them or on any competition data. Values
with no public basis are a severity sweep, and each preset states its source.
"""

import numpy as np

from pvext.convert import FramesTrack

# Params
FPS = 25.0

_BASE = {"ball_gap_s": 1.0, "hard_weight": 3.0, "crowd_radius_m": 3.0, "crowd_k": 4,
         "player_gap_s": 1.0, "noise_rho": 0.99, "n_slots": 46}

PRESETS = {
    "clean": {"source": "format conversion only, no degradation", "identity": True},
    "mild": dict(_BASE, source="severity sweep (assumption): lighter than the documented ATD",
                 identity=False, ball_present=0.85, player_missing=0.02, noise_player_m=0.1,
                 noise_ball_m=0.2, false_ball_rate=0.001, tracklet_mean_s=120.0),
    "doc": dict(_BASE, source="DATA.md: ball located on 59.3 % of graded frames, ~20.5 tracked "
                              "objects of 22, 46 slots; gap lengths, noise, false detections and "
                              "tracklet length are assumptions",
                identity=False, ball_present=0.593, ball_gap_s=2.0,
                player_missing=round(1 - 20.5 / 22, 3), player_gap_s=2.0, noise_player_m=0.2,
                noise_ball_m=0.4, false_ball_rate=0.003, tracklet_mean_s=30.0),
    "severe": dict(_BASE, source="severity sweep (assumption): harsher than the documented ATD",
                   identity=False, ball_present=0.45, ball_gap_s=3.0, player_missing=0.15,
                   player_gap_s=3.0, noise_player_m=0.4, noise_ball_m=0.8, false_ball_rate=0.01,
                   tracklet_mean_s=10.0),
}


def apply(frames, preset, seed):
    """A degraded copy of `frames` (persons become slots). The input is never modified."""
    if preset.get("identity"):
        return frames._replace(players_xy=frames.players_xy.copy(),
                               players_speed_mps=frames.players_speed_mps.copy(),
                               ball_xy=frames.ball_xy.copy(), ball_z=frames.ball_z.copy())
    rng = np.random.default_rng(seed)
    p = preset
    T, P = frames.players_xy.shape[:2]
    playable = ~np.isnan(frames.ball_xy[:, 0])

    present = ~np.isnan(frames.players_xy[..., 0])
    visible = present & ~_markov_missing(rng, (T, P), p["player_missing"],
                                         p["player_gap_s"] * FPS)
    players = frames.players_xy.astype(np.float64) + _ar1(rng, (T, P, 2), p["noise_player_m"],
                                                          p["noise_rho"])
    players[~visible] = np.nan
    players_xy, speed, team_of_slot = _to_slots(rng, players, frames.players_speed_mps, visible,
                                                frames.team_of_slot, p)

    ball = frames.ball_xy.astype(np.float64) + _ar1(rng, (T, 2), p["noise_ball_m"],
                                                    p["noise_rho"])
    hard = playable & ((np.nan_to_num(frames.ball_z) > 1.5)
                       | (_crowd(frames, p["crowd_radius_m"]) >= p["crowd_k"]))
    weights = np.where(hard, p["hard_weight"], 1.0)
    dropped = _ball_dropout(rng, weights, playable, p["ball_present"], p["ball_gap_s"] * FPS)
    ball[dropped | ~playable] = np.nan

    keep = ~np.isnan(ball[:, 0])
    false = keep & (rng.random(T) < p["false_ball_rate"])
    angle = rng.uniform(0, 2 * np.pi, T)
    radius = rng.uniform(5.0, 30.0, T)
    shift = np.stack([np.cos(angle), np.sin(angle)], axis=1) * radius[:, None]
    length, width = frames.pitch
    ball[false] = np.clip(ball[false] + shift[false], [-5.0, -5.0], [length + 5.0, width + 5.0])

    return FramesTrack(players_xy=players_xy.astype(np.float32),
                       players_speed_mps=speed.astype(np.float32), team_of_slot=team_of_slot,
                       ball_xy=ball.astype(np.float32), ball_z=frames.ball_z.copy(),
                       pitch=frames.pitch, speed_unit=frames.speed_unit)


def _markov_missing(rng, shape, share, mean_gap):
    """Boolean missing mask from a two-state chain per column: stationary missing `share`,
    mean missing-run length `mean_gap` frames."""
    missing = np.zeros(shape, dtype=bool)
    if share <= 0:
        return missing
    recover = 1.0 / max(mean_gap, 1.0)
    lose = recover * share / (1.0 - share)
    u = rng.random(shape)
    state = rng.random(shape[1:]) < share
    for t in range(shape[0]):
        state = np.where(state, u[t] >= recover, u[t] < lose)
        missing[t] = state
    return missing


def _ball_dropout(rng, weights, playable, present_share, mean_gap):
    """Missing mask for the ball: hazard base * weight per frame, base solved by bisection so the
    stationary missing share over playable frames is 1 - present_share."""
    T = len(weights)
    missing = np.zeros(T, dtype=bool)
    target = 1.0 - present_share
    if target <= 0:
        return missing
    recover = 1.0 / max(mean_gap, 1.0)
    w = weights[playable]
    low, high = 0.0, 1.0
    for _ in range(60):
        base = (low + high) / 2
        q = np.minimum(base * w, 1.0)
        if np.mean(q / (q + recover)) < target:
            low = base
        else:
            high = base
    hazard = np.minimum(base * weights, 1.0)
    u = rng.random(T)
    state = False
    for t in range(T):
        if not playable[t]:
            state = False
            continue
        state = (u[t] >= recover) if state else (u[t] < hazard[t])
        missing[t] = state
    return missing


def _ar1(rng, shape, sigma, rho):
    """Stationary AR(1) noise with standard deviation sigma, along axis 0."""
    noise = np.zeros(shape)
    if sigma <= 0:
        return noise
    eps = rng.normal(0.0, sigma * np.sqrt(1 - rho * rho), size=shape)
    noise[0] = rng.normal(0.0, sigma, size=shape[1:])
    for t in range(1, shape[0]):
        noise[t] = rho * noise[t - 1] + eps[t]
    return noise


def _crowd(frames, radius_m):
    """Players within radius_m of the ball, per frame (clean positions; NaN-safe)."""
    d = np.linalg.norm(frames.players_xy - frames.ball_xy[:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        return np.sum(d < radius_m, axis=1)


def _to_slots(rng, players, speed, visible, team_of_person, p):
    """Cut every person's visible stretches into tracklets and place each in a free slot of its
    team. A tracklet that finds no free slot is dropped (the tracker lost that player)."""
    T, P = visible.shape
    S = int(p["n_slots"])
    half = S // 2
    team_of_slot = np.array([0] * half + [1] * (S - half), dtype=np.int8)
    out_xy = np.full((T, S, 2), np.nan)
    out_speed = np.full((T, S), np.nan)
    busy = np.zeros((S, T), dtype=bool)
    mean_len = max(p["tracklet_mean_s"] * FPS, 1.0)
    for person in range(P):
        team = int(team_of_person[person])
        slots = np.arange(half) if team == 0 else np.arange(half, S)
        edges = np.diff(np.concatenate(([0], visible[:, person].astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            a = start
            while a < stop:
                b = min(stop, a + int(rng.geometric(1.0 / mean_len)))
                order = np.roll(slots, -int(rng.integers(len(slots))))
                for slot in order:
                    if not busy[slot, a:b].any():
                        busy[slot, a:b] = True
                        out_xy[a:b, slot] = players[a:b, person]
                        out_speed[a:b, slot] = speed[a:b, person]
                        break
                a = b
    return out_xy, out_speed, team_of_slot
