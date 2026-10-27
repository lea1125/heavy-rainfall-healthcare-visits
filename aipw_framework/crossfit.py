"""Date-grouped cross-fitting of the nuisance functions e(x), mu1(x) and mu0(x).

All rows from the same calendar date are held out together, so the
predictions for a facility-day never use other facilities observed on the
same date (which share the same weather shock).
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def crossfit_nuisance(X, T, Y, dates, propensity_model, outcome_model,
                      n_splits=5, verbose=True):
    """Cross-fit the propensity score and the outcome regressions.

    Parameters
    ----------
    X : (n, p) covariates (effect modifiers and confounders together).
    T : (n,) binary treatment indicator (1 = heavy-rainfall exposure).
    Y : (n,) outcome.
    dates : (n,) calendar date of each row; used as the cross-fitting group.
    propensity_model : classifier whose ``fit`` accepts ``groups`` (e.g.
        GroupIsotonicCalibratedClassifier); it receives the training-fold
        dates so that its inner calibration folds are also grouped by date.
    outcome_model : regressor fitted on [X, T]; mu1 / mu0 are its
        predictions with T set to 1 / 0.

    Returns
    -------
    dict with out-of-fold arrays ``e``, ``mu1`` and ``mu0``.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int)
    Y = np.asarray(Y, dtype=float)
    dates = np.asarray(dates)
    n = len(T)
    e, mu1, mu0 = np.zeros(n), np.zeros(n), np.zeros(n)
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits).split(X, T, groups=dates), 1):
        c = clone(propensity_model)
        c.fit(X[tr], T[tr], groups=dates[tr])
        e[va] = c.predict_proba(X[va])[:, 1]

        r = clone(outcome_model)
        r.fit(np.hstack([X[tr], T[tr].reshape(-1, 1)]), Y[tr])
        mu1[va] = r.predict(np.hstack([X[va], np.ones((len(va), 1))]))
        mu0[va] = r.predict(np.hstack([X[va], np.zeros((len(va), 1))]))
        if verbose:
            print(f'  fold {k}/{n_splits} done ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return {'e': e, 'mu1': mu1, 'mu0': mu0}
