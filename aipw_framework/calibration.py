"""Out-of-fold isotonic recalibration of propensity scores, grouped by date.

The base classifier is first cross-fitted *within the training data* using
GroupKFold over dates; an isotonic regression is then learned on those
out-of-fold probabilities and applied to the predictions of a base classifier
refitted on all training rows.  Grouping the inner folds by date matters:
with ordinary KFold the calibration map is learned on rows that share a date
(and hence a weather shock) with the rows being calibrated, and largely fails
to correct overconfident scores.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


class GroupIsotonicCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier wrapper: base model + date-grouped OOF isotonic calibration.

    Parameters
    ----------
    base_estimator : sklearn classifier with ``predict_proba``.
    n_splits : number of inner GroupKFold folds used to build the OOF scores.
    """

    def __init__(self, base_estimator=None, n_splits=5):
        self.base_estimator = base_estimator
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(self.n_splits).split(X, y, groups=groups):
            m = clone(self.base_estimator)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        self.iso_.fit(oof, y)
        self.base_ = clone(self.base_estimator)
        self.base_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.base_.predict_proba(np.asarray(X, dtype=float))[:, 1]
        pc = np.clip(self.iso_.predict(p), 0.0, 1.0)
        return np.column_stack([1.0 - pc, pc])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
