# Heavy rainfall and healthcare-facility visit intensity: core estimation code

This repository contains the core code for the causal estimation framework used in
*Uneven disruption and rapid recovery of healthcare visits under heavy rainfall*
(submitted to Nature Sustainability). It implements:

1. **Date-grouped cross-fitting** of the nuisance functions — propensity score
   e(x) and outcome regressions mu1(x), mu0(x) — with all facility-days of the same
   calendar date held out together (`aipw_framework/crossfit.py`).
2. **Out-of-fold isotonic propensity-score calibration**, with the inner calibration
   folds also grouped by date (`aipw_framework/calibration.py`).
3. **Doubly robust AIPW scores** and their **group-level aggregation**
   (`aipw_framework/aipw.py`, `aipw_framework/inference.py`).
4. **Group-level ratio-of-means transformation**: relative effect =
   100 × mean(psi) / mean(mu0) within a group (`aipw_framework/inference.py`).
5. **Date-clustered bootstrap inference**: dates are resampled with replacement
   (B = 1,000) and the ratio is recomputed in every draw (`aipw_framework/inference.py`).

The nuisance-model specifications used in the paper (L2-regularized logistic
regression for the propensity score; XGBoost for the outcome model, with T as an input
feature) and all tuning constants (propensity clipping at 0.02, 5 folds, B = 1,000,
seed 42) are in `aipw_framework/config.py`.

## Data

The original mobile-phone signalling data were provided by a telecommunications
operator under a confidentiality agreement and cannot be shared. To demonstrate the
workflow, `examples/make_synthetic_data.py` generates a small synthetic facility-day
panel (200 facilities × 80 days) with the same structure — date- and space-clustered
heavy-rainfall exposure, confounding by weather and location, and effects that differ
by day type, month and facility type. Because the data are simulated, the true effects
are known and are written to `examples/data/synthetic_truth.json`.

## Installation

Python 3.12 was used. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Running the example

```bash
python examples/make_synthetic_data.py    # writes examples/data/
python examples/run_example.py            # paper specifications, about 30 s
python examples/run_example.py --fast     # smaller outcome model, about 5 s
```

The script prints the relative effect and its date-clustered 95% CI for each group,
next to the true value from the simulation, and saves them to
`examples/output/example_group_effects.csv`.

## Using the code with your own data

```python
from aipw_framework import crossfit_nuisance, aipw_scores, group_effects
from aipw_framework.config import paper_propensity_model, paper_outcome_model

nuis = crossfit_nuisance(X, T, Y, dates, paper_propensity_model(), paper_outcome_model())
psi = aipw_scores(Y, T, nuis['e'], nuis['mu1'], nuis['mu0'], clip=0.02)
table = group_effects(psi, nuis['mu0'], dates, {'Overall': all_rows, 'Rest day': rest_mask})
```

`X` holds all covariates (effect modifiers and confounders), `T` is the binary
exposure, `Y` the outcome and `dates` the calendar date of each row, which defines
both the cross-fitting groups and the bootstrap clusters.

## Reproducibility note

This code was checked against the study data with the pinned package versions.
The cross-fitted outcome predictions mu1 and mu0 were identical to those used in the
paper. The propensity scores agreed to within 5 × 10⁻⁶; the small differences are
floating-point effects of the numerical solver (for example, a different number of
CPU threads). All point estimates and 95% CIs reported for the overall sample, day
types, months, facility types and cities (26 groups) agreed to within 10⁻⁷ percentage
points. Given the same nuisance predictions, the AIPW, aggregation and bootstrap
steps reproduce the reported values exactly.

## Citation and licence

If you use this code, please cite the paper above (full citation to be added on
publication). The code is released under the MIT Licence (see `LICENSE`).
