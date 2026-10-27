"""Generate a small synthetic facility-day panel with a known treatment effect.

The panel mimics the structure of the study data (facilities observed daily
over 80 summer days; heavy-rainfall exposure that is spatially and temporally
clustered and confounded with weather and location), but every value is
simulated.  No mobile-phone signalling data are used or reproduced.

Outputs (next to this script):
  data/synthetic_panel.csv   facility-day rows used by run_example.py
  data/synthetic_truth.json  true average effects implied by the simulation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2018
N_FACILITIES = 200
FACILITY_TYPES = ['hospital', 'clinic', 'pharmacy']
OUT = Path(__file__).resolve().parent / 'data'


def simulate(seed=SEED, n_fac=N_FACILITIES):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2018-07-06', periods=80, freq='D')

    # facilities: location, type and baseline activity level
    fac = pd.DataFrame({
        'facility_id': [f'F{i:04d}' for i in range(1, n_fac + 1)],
        'lon': rng.uniform(116.4, 121.9, n_fac),
        'lat': rng.uniform(30.8, 35.1, n_fac),
        'facility_type': rng.choice(FACILITY_TYPES, n_fac, p=[0.2, 0.4, 0.4]),
    })
    type_level = fac.facility_type.map({'hospital': 3000.0, 'clinic': 600.0, 'pharmacy': 400.0})
    fac['base_level'] = type_level * rng.lognormal(0.0, 0.4, n_fac)

    # dates: day type, month, province-wide weather shock
    day = pd.DataFrame({'date': dates})
    day['is_offday'] = (day.date.dt.dayofweek >= 5).astype(int)
    day['month'] = day.date.dt.month
    day['rain_shock'] = rng.normal(0.0, 1.0, len(day))
    day['temp_day'] = 30.0 + 2.0 * np.sin(np.arange(len(day)) / 12.0) + rng.normal(0, 1.0, len(day))

    df = day.merge(fac, how='cross')
    n = len(df)

    # weather covariates: cooler and more humid when it rains; south is wetter
    south = (35.1 - df.lat) / 4.3
    df['temperature'] = df.temp_day - 1.5 * df.rain_shock + rng.normal(0, 0.8, n)
    df['humidity'] = 70 + 8 * df.rain_shock + 5 * south + rng.normal(0, 4, n)

    # exposure: date shock + local field + wetter south -> clustered by date and place
    local = rng.normal(0.0, 0.8, n)
    latent = -1.6 + 1.3 * df.rain_shock + 0.8 * south + local
    df['T'] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int)

    # potential outcomes: rest-day and August effects are stronger, as an example
    seasonal = 1.0 + 0.10 * df.is_offday - 0.01 * (df.temperature - 30) - 0.05 * south
    y0 = df.base_level * seasonal * rng.lognormal(0.0, 0.15, n)
    tau = (-0.03 - 0.10 * df.is_offday - 0.05 * (df.month == 8)
           - 0.04 * (df.facility_type == 'clinic'))
    y1 = y0 * (1.0 + tau)
    df['Y'] = np.where(df['T'] == 1, y1, y0)

    for m in (8, 9):
        df[f'month_{m}'] = (df.month == m).astype(int)
    for t in FACILITY_TYPES[1:]:
        df[f'type_{t}'] = (df.facility_type == t).astype(int)

    groups = {'Overall': np.ones(n, bool),
              'Workday': (df.is_offday == 0).values,
              'Rest day': (df.is_offday == 1).values,
              'August': (df.month == 8).values}
    for t in FACILITY_TYPES:
        groups[t] = (df.facility_type == t).values
    truth = {g: 100.0 * float((y1 - y0)[m].mean() / y0[m].mean()) for g, m in groups.items()}

    cols = ['facility_id', 'date', 'facility_type', 'T', 'Y', 'lon', 'lat', 'is_offday',
            'month_8', 'month_9', 'temperature', 'humidity', 'type_clinic', 'type_pharmacy']
    return df[cols], truth


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    panel, truth = simulate()
    panel.to_csv(OUT / 'synthetic_panel.csv', index=False, float_format='%.4f')
    (OUT / 'synthetic_truth.json').write_text(json.dumps(
        {'description': 'True relative effects (%) = 100 * mean(Y1 - Y0) / mean(Y0) per group',
         'relative_effect_pct': truth}, indent=2))
    print(f'wrote {len(panel):,} rows; exposure rate {panel["T"].mean():.3f}')
    for g, v in truth.items():
        print(f'  true relative effect {g:10s} {v:7.2f}%')
