"""HA5a model: HA3's symmetric gradient boosting, generalised to mixed columns.

Mirroring negates only the antisymmetric columns (team-invariant ones are kept) and flips y; the
prediction p(A, S) = (g(A, S) + 1 - g(-A, S)) / 2 is exactly team-swap symmetric. With every column
antisymmetric this is pvext.ha.experiment.SymmetricGB / fit_gb_sym, bit for bit.
"""

import numpy as np

# Params (HA3's, fixed)
GB_ROWS = 300000
GB_PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 0}


def mirror(X, anti):
    out = np.array(X, copy=True)
    out[:, anti] = -out[:, anti]
    return out


class MixedSymmetricGB:
    def __init__(self, model, anti):
        self.model = model
        self.anti = np.asarray(anti, dtype=bool)

    def predict_proba(self, X):
        p = 0.5 * (self.model.predict_proba(X)[:, 1] + 1.0
                   - self.model.predict_proba(mirror(X, self.anti))[:, 1])
        return np.column_stack([1.0 - p, p])


def fit(X, y, anti):
    from sklearn.ensemble import HistGradientBoostingClassifier
    anti = np.asarray(anti, dtype=bool)
    if X.shape[1] != len(anti):
        raise ValueError("anti mask width differs from X")
    idx = np.random.default_rng(0).choice(len(X), min(len(X), GB_ROWS), replace=False)
    Xa = np.concatenate([X[idx], mirror(X[idx], anti)])
    ya = np.concatenate([y[idx], ~y[idx]])
    return MixedSymmetricGB(HistGradientBoostingClassifier(**GB_PARAMS).fit(Xa, ya), anti)
