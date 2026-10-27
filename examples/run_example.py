"""Run the full estimation workflow on the synthetic panel.

    python examples/make_synthetic_data.py
    python examples/run_example.py            # paper nuisance specifications
    python examples/run_example.py --fast     # lighter outcome model for a quick check

Steps: date-grouped cross-fitting of e(x), mu1(x), mu0(x) with date-grouped
out-of-fold isotonic calibration of e(x) -> AIPW scores -> group-level
ratio-of-means relative effects with date-clustered bootstrap 95% CIs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aipw_framework import aipw_scores, crossfit_nuisance, group_effects  # noqa: E402
from aipw_framework.config import (B_BOOTSTRAP, CLIP, N_SPLITS, SEED,  # noqa: E402
                                   paper_outcome_model, paper_propensity_model)

COVARIATES = ['lon', 'lat', 'is_offday', 'month_8', 'month_9', 'temperature', 'humidity',
              'type_clinic', 'type_pharmacy']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='use a smaller XGBoost outcome model')
    args = ap.parse_args()

    df = pd.read_csv(HERE / 'data' / 'synthetic_panel.csv', parse_dates=['date'])
    truth = json.loads((HERE / 'data' / 'synthetic_truth.json').read_text())['relative_effect_pct']
    X = df[COVARIATES].to_numpy(dtype=float)
    T = df['T'].to_numpy(dtype=int)
    Y = df['Y'].to_numpy(dtype=float)
    dates = df['date'].to_numpy()
    print(f'{len(df):,} facility-days, {df.date.nunique()} dates, exposure rate {T.mean():.3f}')

    outcome = (XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                            min_child_weight=20, n_jobs=2, random_state=SEED)
               if args.fast else paper_outcome_model(n_jobs=2))
    nuis = crossfit_nuisance(X, T, Y, dates, paper_propensity_model(), outcome,
                             n_splits=N_SPLITS)
    psi = aipw_scores(Y, T, nuis['e'], nuis['mu1'], nuis['mu0'], clip=CLIP)

    groups = {'Overall': np.ones(len(df), bool),
              'Workday': (df.is_offday == 0).to_numpy(),
              'Rest day': (df.is_offday == 1).to_numpy(),
              'August': (df.month_8 == 1).to_numpy()}
    for t in ('hospital', 'clinic', 'pharmacy'):
        groups[t] = (df.facility_type == t).to_numpy()
    res = group_effects(psi, nuis['mu0'], dates, groups, B=B_BOOTSTRAP, seed=SEED)
    res['true_relative_effect_pct'] = res.group.map(truth)

    out = HERE / 'output'
    out.mkdir(exist_ok=True)
    res.to_csv(out / 'example_group_effects.csv', index=False)
    show = res[['group', 'n_rows', 'n_dates', 'relative_effect_pct', 'relative_ci_low_pct',
                'relative_ci_high_pct', 'true_relative_effect_pct']]
    print(show.to_string(index=False, float_format=lambda v: f'{v:8.2f}'))
    print(f'saved {out / "example_group_effects.csv"}')


if __name__ == '__main__':
    main()
