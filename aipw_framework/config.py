"""Nuisance-model specifications used in the paper.

Hyperparameters were tuned with date-grouped cross-validation on the study
data; they are reproduced here unchanged.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .calibration import GroupIsotonicCalibratedClassifier

CLIP = 0.02          # propensity-score clipping bound
N_SPLITS = 5         # outer (cross-fitting) and inner (calibration) folds
B_BOOTSTRAP = 1000   # date-clustered bootstrap draws
SEED = 42


def paper_propensity_model(n_splits=N_SPLITS):
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.0010102163357543319, solver='lbfgs',
                                   max_iter=1000, random_state=42)),
    ])
    return GroupIsotonicCalibratedClassifier(base_estimator=base, n_splits=n_splits)


def paper_outcome_model(n_jobs=2):
    return XGBRegressor(n_estimators=1522, max_depth=12,
                        learning_rate=0.010639487030139548, min_child_weight=20,
                        subsample=0.7459971983795849, colsample_bytree=0.8018588067632909,
                        reg_alpha=0.07806218291775723, reg_lambda=44.844647134760066,
                        n_jobs=n_jobs, random_state=42)
