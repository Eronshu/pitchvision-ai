"""S2: joint three-state (H/A/D) Viterbi on frozen probabilities.

  pi(D) = d,  pi(H) = (1 - d) h,  pi(A) = (1 - d)(1 - h)       (d, h clipped to [1e-6, 1 - 1e-6])
  S(y)  = sum_t dt log pi_t(y_t) - sum_{t>1} C(y_{t-1}, y_t)
C: H<->A = lam_ha, H/A -> D = lam_on, D -> H/A = lam_off; all six transitions allowed; uniform start.
Each graded block is decoded independently; frames outside the blocks are X.
Ties (deterministic, never team information): a stay beats an equal-score switch; equal-score
switches into a state take the lower state index in the order D, H, A of the predecessor; an equal
final score takes D, then H, then A.
"""

import numpy as np

P_CLIP = 1e-6
H, A, D = 0, 1, 2
CODES = np.array(["H", "A", "D"])
PRED_ORDER = (D, H, A)


def emissions(p_dead, p_home, dt):
    d = np.clip(np.asarray(p_dead, np.float64), P_CLIP, 1 - P_CLIP)
    h = np.clip(np.asarray(p_home, np.float64), P_CLIP, 1 - P_CLIP)
    e = np.empty((len(d), 3))
    e[:, H] = np.log1p(-d) + np.log(h)
    e[:, A] = np.log1p(-d) + np.log1p(-h)
    e[:, D] = np.log(d)
    return dt * e


def cost_matrix(lam_ha, lam_on, lam_off):
    if lam_on + lam_off < lam_ha - 1e-12:
        raise ValueError("lam_on + lam_off must be >= lam_ha (no cheaper H->D->A detour)")
    C = np.zeros((3, 3))
    C[H, A] = C[A, H] = lam_ha
    C[H, D] = C[A, D] = lam_on
    C[D, H] = C[D, A] = lam_off
    return C


def viterbi3(e, C):
    """Best state path (int8 array) for one block of dt-scaled log emissions e (T, 3)."""
    T = len(e)
    if T == 0:
        return np.zeros(0, np.int8)
    rows = e.tolist()
    cost = C.tolist()
    back = np.zeros((T, 3), np.int8)
    v = list(rows[0])
    for t in range(1, T):
        nv = [0.0, 0.0, 0.0]
        bt = back[t]
        r = rows[t]
        for j in (0, 1, 2):
            best, arg = v[j], j                       # stay wins ties
            for i in PRED_ORDER:
                if i != j:
                    s = v[i] - cost[i][j]
                    if s > best:
                        best, arg = s, i
            nv[j] = best + r[j]
            bt[j] = arg
        v = nv
    state = max(PRED_ORDER, key=lambda k: (v[k], -PRED_ORDER.index(k)))
    path = np.empty(T, np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = state
        state = back[t][state]
    return path


def decode(p_dead, p_home, blocks, lam_ha, lam_on, lam_off, fps):
    """H/A/D/X labels for the whole series."""
    e = emissions(p_dead, p_home, 1.0 / fps)
    C = cost_matrix(lam_ha, lam_on, lam_off)
    out = np.full(len(e), "X", dtype="<U1")
    for _, first, last in blocks:
        out[first:last + 1] = CODES[viterbi3(e[first:last + 1], C)]
    return out


def path_score(e, C, path):
    s = float(sum(e[t, path[t]] for t in range(len(path))))
    return s - float(sum(C[path[t - 1], path[t]] for t in range(1, len(path))))
