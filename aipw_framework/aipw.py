"""Doubly robust (AIPW) scores."""
import numpy as np


def aipw_scores(Y, T, e, mu1, mu0, clip=0.02):
    """Row-level AIPW pseudo-outcomes psi_i.

    psi = mu1 - mu0 + T (Y - mu1) / e - (1 - T) (Y - mu0) / (1 - e),
    with the propensity score clipped to [clip, 1 - clip].  The mean of psi
    over any group of rows is the AIPW estimate of the average effect in that
    group.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    ec = np.clip(np.asarray(e, dtype=float), clip, 1 - clip)
    return mu1 - mu0 + T * (Y - mu1) / ec - (1 - T) * (Y - mu0) / (1 - ec)
