"""H / A / UNKNOWN possession state machine with hysteresis.

Per frame each team gets a control evidence score in [0, 1] from three factors: how close its
nearest player is, how clearly it beats the other team, and how slowly the ball moves relative to
that player. Relative speed lowers the score but never zeroes it (`speed_floor`), so a fast
reception or an interception still counts. A missing ball gives no evidence at all.

The state only changes on sustained, clear evidence; everything else -- a loose ball, a pass in
flight, a deflection, a contested ball, a missing ball -- keeps the last controlled possession,
which is the official rule. A run of live play starts UNKNOWN (no holder to defend) and resolves on
a lower bar, back-filled to the restart.
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1
LETTERS = np.array(["H", "A"])


def evidence(feat, cfg_pos):
    """(e_h, e_a): per-frame control evidence for each team, float32 in [0, 1]."""
    c = cfg_pos

    def one(dist, other, rel):
        with np.errstate(invalid="ignore"):
            e_dist = np.clip(1.0 - (dist - c["r_ctrl_m"]) / c["r_decay_m"], 0.0, 1.0)
            e_margin = np.clip((other - dist) / c["margin_m"], 0.0, 1.0)
        rel = np.where(np.isnan(rel), 0.0, rel)
        e_motion = c["speed_floor"] + (1.0 - c["speed_floor"]) / (1.0 + rel / c["v_ref_mps"])
        e = e_dist * np.nan_to_num(e_margin, nan=1.0) * e_motion
        e[np.isinf(dist)] = 0.0
        return e.astype(np.float32)

    return (one(feat.dist_h, feat.dist_a, feat.rel_speed_h),
            one(feat.dist_a, feat.dist_h, feat.rel_speed_a))


def label_block(feat, dead, nearest, cfg_pos, fps, restart_team=None):
    """Labels for one graded block: 'H', 'A', 'D' for dead frames, 'X' where nobody is known."""
    n = len(dead)
    out = np.full(n, "D", dtype="<U1")
    e = np.stack(evidence(feat, cfg_pos))
    win = max(1, int(round(cfg_pos["win_s"] * fps)))
    for start, stop in _runs(~np.asarray(dead, dtype=bool)):
        seed = NOBODY if restart_team is None else int(restart_team[start])
        out[start:stop] = _label_run(e[:, start:stop], nearest[start:stop], cfg_pos, win, seed)
    return out


def _label_run(e, nearest, c, win, seed):
    """Labels for one ALIVE run, given its (2, n) evidence slice."""
    n = e.shape[1]
    sums = np.concatenate((np.zeros((2, 1)), np.cumsum(e, axis=1)), axis=1)
    state = seed
    team = np.full(n, state, dtype=np.int8)
    last_change = -1

    for t in range(n):
        low = max(0, t - win + 1)
        m = (sums[:, t + 1] - sums[:, low]) / (t + 1 - low)
        if state == NOBODY:
            best = int(np.argmax(m))
            if m[best] >= c["init_on"] and m[best] > m[1 - best]:
                state = best
                team[:t] = state  # back-fill to the restart
                last_change = t
        else:
            other = 1 - state
            if m[other] >= c["switch_on"] and m[other] - m[state] >= c["switch_margin"]:
                change = t
                if c["boundary"] == "evidence_start":
                    while change - 1 > last_change and e[other, change - 1] >= c["evidence_on"]:
                        change -= 1
                team[change:t] = other
                state = other
                last_change = t
        team[t] = state

    labels = np.full(n, "X", dtype="<U1")
    known = team != NOBODY
    labels[known] = LETTERS[team[known]]
    if not known.any():
        votes = nearest[nearest != NOBODY]
        if len(votes):
            labels[:] = "H" if np.sum(votes == HOME) >= np.sum(votes == AWAY) else "A"
    return labels


def _runs(mask):
    """[(start, stop)] half-open spans where mask is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
