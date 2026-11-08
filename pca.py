"""
PCA -- same core math as your POD step (SVD of the (mean-centered)
snapshot matrix), with one added option: standardizing each of the 200
series to unit variance before decomposing. This matters when your series
have very different scales/units -- plain POD/PCA on raw units lets
large-magnitude series dominate the "energy" ranking regardless of how
informative they are; standardized PCA instead treats every series as
equally important a priori.

Input convention (matches your POD/SPOD/DMD pipeline):
    X : ndarray, shape (n_time, n_state)   e.g. (8000, 200)
"""

import numpy as np


def compute_pca(X, standardize=False):
    """
    Parameters
    ----------
    X : ndarray, shape (n_time, n_state)
    standardize : bool
        If True, divide each column by its std before decomposing (this is
        "PCA on the correlation matrix" instead of "PCA on the covariance
        matrix" -- i.e. classic standardized PCA). If False, this is
        numerically identical to your existing POD step.

    Returns
    -------
    components : ndarray, shape (n_state, n_modes)
        Principal directions (spatial modes), unit-norm columns, ordered
        by decreasing explained variance.
    scores : ndarray, shape (n_time, n_modes)
        Time coefficients a(t) for each component -- this is the direct
        analogue of your POD `modes_coeffs`, ready to feed into
        run_volterra_identification_sequential.
    explained_variance : ndarray, shape (n_modes,)
        Variance explained by each component (eigenvalue of the covariance
        matrix), descending.
    explained_variance_ratio : ndarray, shape (n_modes,)
        Same, normalized to sum to 1 -- convenient for picking r via a
        cumulative-variance threshold (e.g. "keep enough modes for 95%").
    mean_ : ndarray, shape (n_state,)
    scale_ : ndarray, shape (n_state,) or None
        The per-column std used if standardize=True, else None. Needed to
        invert back to physical units later if you reconstruct X.
    """
    mean_ = X.mean(axis=0)
    Xc = X - mean_

    scale_ = None
    if standardize:
        scale_ = Xc.std(axis=0)
        scale_[scale_ == 0] = 1.0
        Xc = Xc / scale_

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    n_time = X.shape[0]
    explained_variance = (S**2) / (n_time - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    components = Vt.T                 # (n_state, n_modes)
    scores = U * S                    # (n_time, n_modes) == Xc @ components

    return components, scores, explained_variance, explained_variance_ratio, mean_, scale_


def select_top_r(scores, components, explained_variance_ratio, r):
    """Convenience: just take the first r columns (already sorted by PCA)."""
    return components[:, :r], scores[:, :r], explained_variance_ratio[:r]

