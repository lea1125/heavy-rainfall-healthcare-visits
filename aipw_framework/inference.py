"""Group-level aggregation, ratio-of-means transformation and date-clustered bootstrap.

For a group G of rows the absolute effect is mean_G(psi) and the relative
effect is

    100 * mean_G(psi) / mean_G(mu0)        (ratio of means, in %),

i.e. the average change expressed as a percentage of the average
counterfactual (no-heavy-rainfall) outcome.  Confidence intervals resample
whole dates with replacement; the ratio is recomputed inside every draw so
that the covariance between numerator and denominator is retained.  The same
B x D matrix of resampled dates is shared by all groups.
"""
import numpy as np
import pandas as pd


def date_bootstrap_draws(dates, B=1000, seed=42):
    """Index each row by its date and draw B resamples of the D unique dates."""
    dates = np.asarray(dates)
    uniq = np.sort(pd.unique(dates))
    date_index = pd.Series(dates).map({d: i for i, d in enumerate(uniq)}).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return date_index, draws


def group_effects(psi, mu0, dates, groups, B=1000, seed=42):
    """Absolute and relative effects with date-clustered 95% CIs for each group.

    Parameters
    ----------
    psi : (n,) AIPW scores.
    mu0 : (n,) cross-fitted counterfactual outcome without treatment.
    dates : (n,) cluster (date) of each row.
    groups : dict mapping group name -> boolean mask of length n.

    Returns
    -------
    DataFrame with one row per group.
    """
    psi = np.asarray(psi, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    date_index, draws = date_bootstrap_draws(dates, B=B, seed=seed)
    D = draws.shape[1]
    rows = []
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        psi_d = np.bincount(date_index[mask], weights=psi[mask], minlength=D)
        mu0_d = np.bincount(date_index[mask], weights=mu0[mask], minlength=D)
        cnt_d = np.bincount(date_index[mask], minlength=D)

        b_psi, b_mu0, b_cnt = psi_d[draws].sum(1), mu0_d[draws].sum(1), cnt_d[draws].sum(1)
        ok = (b_cnt > 0) & (b_mu0 > 0)
        abs_lo, abs_hi = np.percentile(b_psi[ok] / b_cnt[ok], [2.5, 97.5])
        rel_lo, rel_hi = np.percentile(100.0 * b_psi[ok] / b_mu0[ok], [2.5, 97.5])

        absolute = psi[mask].mean()
        baseline = mu0[mask].mean()
        rows.append({'group': name,
                     'n_rows': int(mask.sum()),
                     'n_dates': int((cnt_d > 0).sum()),
                     'absolute_effect': absolute,
                     'absolute_ci_low': abs_lo, 'absolute_ci_high': abs_hi,
                     'counterfactual_mean': baseline,
                     'relative_effect_pct': 100.0 * absolute / baseline,
                     'relative_ci_low_pct': rel_lo, 'relative_ci_high_pct': rel_hi,
                     'valid_draws': int(ok.sum())})
    return pd.DataFrame(rows)
