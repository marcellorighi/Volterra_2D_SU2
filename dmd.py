"""
Dynamic Mode Decomposition (exact DMD, Tu et al. 2014).

Unlike POD/SPOD (which rank modes by energy/variance), DMD finds the
best-fit LINEAR operator A such that x(t+1) ~ A @ x(t), then extracts its
eigenvectors (DMD modes) and eigenvalues. Each mode therefore comes with
an intrinsic growth/decay rate and oscillation frequency, rather than
being ranked by how much variance it explains. This tends to suit
identification/dynamics problems (like your Volterra fitting) better than
POD, since the modes are already dynamically coherent by construction.

Input convention (to match your existing pipeline):
    X : ndarray, shape (n_time, n_state)   e.g. (8000, 200)
        same orientation as your POD/SPOD snapshot arrays.
"""

import numpy as np


def compute_dmd(X, dt, r=None):
    """
    Exact DMD on a time series.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_state)
        Time series (rows = time steps, columns = state/series index).
    dt : float
        Sampling interval, needed to convert discrete eigenvalues into
        physical growth rates / frequencies.
    r : int or None
        Truncation rank for the SVD step. If None, keeps all singular
        values above a small relative tolerance (1e-10 x largest). This
        is the main "how many degrees of freedom does the linear model
        get" knob -- too large risks fitting noise as if it were a mode,
        too small under-resolves genuine dynamics. Worth scanning a few
        values and checking the singular value spectrum (returned as `S`)
        directly.

    Returns
    -------
    eigvals : ndarray, shape (r,), complex
        Discrete-time DMD eigenvalues (lambda). |lambda| > 1 means growing,
        < 1 means decaying, = 1 means neutral/oscillatory. Complex
        conjugate pairs are typical for real, oscillatory data.
    omega : ndarray, shape (r,), complex
        Continuous-time eigenvalues, omega = log(lambda) / dt. Real part
        is growth/decay rate; imaginary part / (2*pi) is frequency in the
        same units as 1/dt.
    modes : ndarray, shape (n_state, r), complex
        DMD mode shapes (columns), analogous to POD/SPOD spatial modes,
        but each tied to a single eigval/omega rather than to an energy
        ranking.
    amplitudes : ndarray, shape (r,), complex
        Initial amplitude b_i of each mode, from fitting Phi @ b = x(0).
        |amplitudes[i]| * ||modes[:, i]|| is a reasonable proxy for how
        much a mode actually contributes to the observed data (unlike POD,
        DMD eigenvalues alone do NOT rank modes by importance).
    S : ndarray
        Singular values from the SVD truncation step, for diagnosing
        whether `r` was chosen well.
    """
    # DMD convention: state x snapshot, i.e. transpose of the (n_time, n_state) input
    Xt = X.T  # shape (n_state, n_time)
    X1 = Xt[:, :-1]
    X2 = Xt[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)

    if r is None:
        tol = S[0] * 1e-10
        r = int(np.sum(S > tol))
    r = min(r, U.shape[1])

    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh[:r, :].conj().T

    # Best-fit linear operator, projected onto the r-dim POD subspace
    A_tilde = U_r.conj().T @ X2 @ V_r @ np.diag(1.0 / S_r)

    eigvals, W = np.linalg.eig(A_tilde)

    # Exact DMD modes (Tu et al. 2014): reconstructs modes directly from
    # X2 rather than only from U_r, which is more accurate than the
    # "projected modes" (U_r @ W) variant when X1's column space doesn't
    # fully capture X2.
    modes = X2 @ V_r @ np.diag(1.0 / S_r) @ W

    omega = np.log(eigvals) / dt

    # Initial amplitudes: solve Phi @ b = x(0) in least-squares sense
    x0 = Xt[:, 0]
    amplitudes = np.linalg.lstsq(modes, x0, rcond=None)[0]

    return eigvals, omega, modes, amplitudes, S


def reconstruct_dynamics(modes, omega, amplitudes, t):
    """
    Analytic DMD reconstruction: x(t) = sum_i modes[:,i] * amplitudes[i] * exp(omega[i]*t).

    Parameters
    ----------
    t : ndarray, shape (n_time,)
        Time points (e.g. np.arange(n_time) * dt).

    Returns
    -------
    X_recon : ndarray, shape (n_time, n_state), real
        Reconstructed (real part) time series. Compare against your
        original X to check DMD fit quality / choice of r.
    """
    time_dynamics = np.exp(np.outer(omega, t))          # (r, n_time)
    X_recon = (modes * amplitudes[None, :]) @ time_dynamics  # (n_state, n_time)
    return np.real(X_recon).T


def dmd_time_coefficients(modes, X):
    """
    Project the FULL measured time series onto the DMD modes to get each
    mode's actual (measured) time coefficient, rather than only its
    idealized analytic exp(omega*t) trajectory.

    This is the DMD analogue of POD's a(t) / SPOD's projected coefficient:
    a(t) = pinv(Phi) @ x(t) for every t. Unlike the analytic reconstruction
    above, this coefficient tracks the REAL data (including whatever the
    linear DMD model does not capture), which is generally what you want
    to feed into a downstream identification step like your Volterra
    fitting -- same rationale as projecting for SPOD.

    Returns
    -------
    A : ndarray, shape (n_time, n_modes), complex
        Time coefficient for every mode at every time step. Take
        `np.real(A[:, i])` (or see select_top_dmd_modes for pairing up
        complex-conjugate modes into one real signal) before using as a
        Volterra target.
    """
    Phi_pinv = np.linalg.pinv(modes)      # (n_modes, n_state)
    A = (Phi_pinv @ X.T).T                # (n_time, n_modes)
    return A

