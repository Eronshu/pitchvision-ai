"""HA1 decoding: a two-state (H/A) Viterbi with a switch penalty, per alive segment.

Each frame contributes dt * log p to H and dt * log(1 - p) to A, with dt = 1 / fps seconds, so the
evidence is per second and the switch penalty `lam` (nats) means the same at any frame rate. Each
alive segment (a maximal run of non-dead frames inside a graded block) starts from a uniform prior
and is decoded independently. Dead frames are D, frames outside the blocks X.
"""

import numpy as np

# Params
P_CLIP = 1e-6


def segments(dead, blocks):
    """[(start, stop)] half-open alive runs inside the blocks."""
    out = []
    for _, first, last in blocks:
        alive = ~np.asarray(dead[first:last + 1], dtype=bool)
        edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out.append((first + int(a), first + int(b)))
    return out


def viterbi(p, lam, fps):
    """Most probable H(True)/A(False) path for one segment of home probabilities `p`."""
    p = np.clip(np.asarray(p, np.float64), P_CLIP, 1 - P_CLIP)
    dt = 1.0 / fps
    eh, ea = dt * np.log(p), dt * np.log1p(-p)
    n = len(p)
    from_h = np.zeros(n, dtype=bool)     # best predecessor of state H at t is H?
    from_a = np.zeros(n, dtype=bool)     # best predecessor of state A at t is A?
    vh, va = eh[0], ea[0]
    for t in range(1, n):
        stay_h, switch_to_h = vh, va - lam
        stay_a, switch_to_a = va, vh - lam
        from_h[t] = stay_h >= switch_to_h
        from_a[t] = stay_a >= switch_to_a
        vh = eh[t] + (stay_h if from_h[t] else switch_to_h)
        va = ea[t] + (stay_a if from_a[t] else switch_to_a)
    path = np.zeros(n, dtype=bool)
    state = vh >= va                      # ties -> H, and symmetric handling below
    if vh == va:
        state = p.mean() >= 0.5
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t:
            state = state if (from_h[t] if state else from_a[t]) else (not state)
    return path


def decode(p_home, dead, blocks, lam, fps):
    """H/A/D/X labels: D where `dead`, X outside the blocks, Viterbi H/A in each alive segment."""
    n = len(p_home)
    out = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        out[first:last + 1] = np.where(np.asarray(dead[first:last + 1], bool), "D", "X")
    for a, b in segments(dead, blocks):
        out[a:b] = np.where(viterbi(p_home[a:b], lam, fps), "H", "A")
    return out
