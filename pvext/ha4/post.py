"""HA4 post-processing (design: docs/superpowers/specs/2026-11-13-pitchvision-ha4-design.md).

dead_mask()   M1 persistence with a configurable minimum dead length, then A1 restart snapping.
decode()      HA3's two-state Viterbi per alive segment, with an optional A2 start prior for the
              team a set-piece tag names. With beta = 0 it is HA3's frozen decoder exactly.
The frozen modules (pvext/v2/m1.py, pvext/ha/decode.py) are imported, never edited.
"""

import math

import numpy as np

from pvext.ha import decode as ha_decode
from pvext.v2 import m1

# Params
FPS = 25
SMOOTH_S = 1.0          # M1 / V1 persistence, unchanged
MIN_ALIVE_S = 1.0
ASSIGN_MIN_S = 3.0
NO_TEAM = -1


def persist(p_dead, blocks, threshold, min_dead_s, fps=FPS):
    cfg = {"smooth_s": SMOOTH_S, "min_dead_s": float(min_dead_s), "min_alive_s": MIN_ALIVE_S}
    return m1.persist(p_dead, blocks, cfg, fps, threshold)


def snap(dead, blocks, tag_frames, snap_s, insert_s, fps=FPS):
    """A1: move the end of the nearest dead run (that starts before r) to each tag frame r, if it
    ends within snap_s of r; with no dead frame within +-snap_s, insert [r - insert_s, r)."""
    dead = np.array(dead, dtype=bool, copy=True)
    if snap_s <= 0 or not len(tag_frames):
        return dead
    w = int(round(snap_s * fps))
    ins = int(round(insert_s * fps))
    for r in np.asarray(tag_frames, dtype=np.int64):
        block = next(((a, b) for _, a, b in blocks if a <= r <= b), None)
        if block is None:
            continue
        first, last = block
        lo, hi = max(first, r - w), min(last, r + w)
        window = dead[lo:hi + 1]
        if window.any():
            # dead runs touching the window, each as [a, b) on the block
            seg = dead[first:last + 1]
            best = None
            for a, b in m1._runs(seg):
                a, b = a + first, b + first
                if a < r and abs(b - r) <= w and (best is None or abs(b - r) < abs(best[1] - r)):
                    best = (a, b)
            if best is not None:
                a, b = best
                if b > r:
                    dead[r:b] = False
                else:
                    dead[b:r] = True
        elif ins > 0:
            dead[max(first, r - ins):r] = True
    return dead


def dead_mask(p_dead, blocks, threshold, min_dead_s, tag_frames=(), snap_s=0.0, insert_s=0.0,
              fps=FPS):
    return snap(persist(p_dead, blocks, threshold, min_dead_s, fps), blocks, tag_frames, snap_s,
                insert_s, fps)


def viterbi(p, lam, fps=FPS, prior_home=0.0):
    """HA3's viterbi() with an additive start prior (nats) for H (> 0) or A (< 0). The same
    IEEE-double operations in the same order on Python floats (faster than numpy scalars), so with
    prior 0 it is bit-identical to pvext.ha.decode.viterbi (tested)."""
    p = np.clip(np.asarray(p, np.float64), ha_decode.P_CLIP, 1 - ha_decode.P_CLIP)
    dt = 1.0 / fps
    eh = (dt * np.log(p)).tolist()
    ea = (dt * np.log1p(-p)).tolist()
    n = len(eh)
    from_h = [False] * n
    from_a = [False] * n
    vh = eh[0] + max(prior_home, 0.0)
    va = ea[0] + max(-prior_home, 0.0)
    for t in range(1, n):
        sh, wh = vh, va - lam
        sa, wa = va, vh - lam
        fh = sh >= wh
        fa = sa >= wa
        from_h[t], from_a[t] = fh, fa
        vh = eh[t] + (sh if fh else wh)
        va = ea[t] + (sa if fa else wa)
    path = np.zeros(n, dtype=bool)
    state = vh >= va
    if vh == va:
        state = float(np.mean(p)) >= 0.5
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t:
            state = state if (from_h[t] if state else from_a[t]) else (not state)
    return path


def segment_priors(segments, tag_frames, tag_teams, beta, assign_s, fps=FPS):
    """{segment start: prior_home} from the nearest tag with a team within assign_s of it."""
    out = {}
    if beta == 0 or not len(tag_frames):
        return out
    frames = np.asarray(tag_frames, dtype=np.int64)
    teams = np.asarray(tag_teams)
    known = teams != NO_TEAM
    frames, teams = frames[known], teams[known]
    if not len(frames):
        return out
    w = int(round(assign_s * fps))
    for a, _ in segments:
        i = int(np.argmin(np.abs(frames - a)))
        if abs(int(frames[i]) - a) <= w:
            out[a] = beta if int(teams[i]) == 0 else -beta
    return out


def decode(p_home, dead, blocks, lam, tag_frames=(), tag_teams=(), beta=0.0, assign_s=0.0,
           fps=FPS):
    """H/A/D/X labels as pvext.ha.decode.decode, with the A2 start priors."""
    n = len(p_home)
    out = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        out[first:last + 1] = np.where(np.asarray(dead[first:last + 1], bool), "D", "X")
    segs = ha_decode.segments(dead, blocks)
    priors = segment_priors(segs, tag_frames, tag_teams, beta, max(assign_s, ASSIGN_MIN_S), fps)
    for a, b in segs:
        out[a:b] = np.where(viterbi(p_home[a:b], lam, fps, priors.get(a, 0.0)), "H", "A")
    return out


def assign_seconds(snap_s):
    return max(float(snap_s), ASSIGN_MIN_S)


def finite(x):
    return x is not None and math.isfinite(x)
