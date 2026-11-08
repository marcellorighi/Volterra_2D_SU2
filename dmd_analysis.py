"""
Practical DMD workflow: pick r modes, get real time coefficients, plot.

Builds on dmd.py (compute_dmd). Two things make DMD mode selection less
direct than POD's simple "top r by eigenvalue":

1. DMD eigenvalues do NOT rank modes by importance (unlike POD/SPOD
   eigenvalues) -- a mode can have an eigenvalue near the unit circle
   (neutral/persistent) but negligible amplitude, or vice versa. Selection
   should be based on |amplitude| * ||mode shape||, not on omega/eigval
   alone.

2. For real-valued input data, DMD eigenvalues/modes come in complex-
   conjugate pairs. Each pair together represents ONE real oscillatory
   structure -- so "r modes" should usually mean r conjugate PAIRS (or r
   real, non-oscillatory modes for eigenvalues that are purely real),
   not r raw complex eigenvalues.
"""

import numpy as np
import matplotlib.pyplot as plt

from dmd import compute_dmd, dmd_time_coefficients, reconstruct_dynamics


# ============================================================================
# 1) Select the r most important, non-redundant modes
# ============================================================================

def select_top_dmd_modes(eigvals, omega, modes, amplitudes, r, conj_tol=1e-8):
    """
    Rank modes by contribution (|amplitude| * ||mode shape||), collapsing
    complex-conjugate pairs into a single entry so you don't double-count
    the same physical oscillation.

    Returns
    -------
    selected : list of dict, length <= r, each with keys:
        'eigval', 'omega', 'shape' (n_state,) complex, 'amplitude' (complex),
        'importance' (float), 'is_conjugate_pair' (bool),
        'paired_index' (int or None) -- index into the ORIGINAL arrays of
        the conjugate partner, if any (useful if you need both halves for
        an exact analytic reconstruction).
    """
    n_modes = len(eigvals)
    importance = np.abs(amplitudes) * np.linalg.norm(modes, axis=0)

    used = np.zeros(n_modes, dtype=bool)
    candidates = []

    order = np.argsort(importance)[::-1]
    for i in order:
        if used[i]:
            continue
        # find this mode's conjugate partner, if any
        partner = None
        for j in order:
            if j != i and not used[j] and np.abs(eigvals[j] - np.conj(eigvals[i])) < conj_tol:
                partner = j
                break
        candidates.append({
            'eigval': eigvals[i],
            'omega': omega[i],
            'shape': modes[:, i],
            'amplitude': amplitudes[i],
            'importance': importance[i],
            'is_conjugate_pair': partner is not None,
            'paired_index': partner,
        })
        used[i] = True
        if partner is not None:
            used[partner] = True

    candidates.sort(key=lambda m: m['importance'], reverse=True)
    return candidates[:r]


# ============================================================================
# 2) Real-valued time coefficients for the selected modes
# ============================================================================

def get_dmd_time_coefficients(X, selected, modes_full):
    """
    Real-valued time coefficient a(t) per selected mode, suitable as a
    target column for run_volterra_identification_sequential.

    For a complex-conjugate pair, phi*b*exp(omega t) + conj(...) is already
    real, so the physically meaningful real signal is
    2 * Real( a_i(t) ) where a_i(t) is the projected coefficient of EITHER
    half of the pair. For a purely real eigenvalue (no partner), the
    coefficient is used as-is (already real up to numerical noise).
    """
    A_complex = dmd_time_coefficients(modes_full, X)  # (n_time, n_modes_full), complex

    # need each selected mode's column index within modes_full
    def _find_col(shape):
        # modes_full columns should match by identity/order; safest is to
        # match by (near-)equality since `selected` shapes are views into
        # the same array.
        diffs = np.linalg.norm(modes_full - shape[:, None], axis=0)
        return int(np.argmin(diffs))

    out = np.empty((X.shape[0], len(selected)))
    for i, mode in enumerate(selected):
        col = _find_col(mode['shape'])
        if mode['is_conjugate_pair']:
            out[:, i] = 2.0 * np.real(A_complex[:, col])
        else:
            out[:, i] = np.real(A_complex[:, col])
    return out


# ============================================================================
# 3) Plots
# ============================================================================

def plot_dmd_spectrum(eigvals, amplitudes, selected=None, ax=None):
    """
    DMD eigenvalues on the complex plane with the unit circle overlaid.
    Points ON the unit circle = neutral/persistent oscillations; INSIDE =
    decaying; OUTSIDE = growing (often a sign of an under-resolved / noisy
    fit if you see many points well outside the circle). Marker size
    reflects |amplitude| (more informative than position alone, since
    eigenvalue position does not indicate importance).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, label='unit circle')

    size = 200 * np.abs(amplitudes) / (np.max(np.abs(amplitudes)) + 1e-16)
    ax.scatter(eigvals.real, eigvals.imag, s=size + 5, alpha=0.6, label='all modes')

    if selected is not None:
        sel_eig = [m['eigval'] for m in selected]
        ax.scatter([e.real for e in sel_eig], [e.imag for e in sel_eig],
                   facecolors='none', edgecolors='k', s=250, linewidths=1.5,
                   label='selected (top r)')

    ax.set_xlabel('Re(lambda)')
    ax.set_ylabel('Im(lambda)')
    ax.set_title('DMD eigenvalue spectrum')
    ax.axis('equal')
    ax.legend()
    return ax


def plot_dmd_mode_shapes(selected, state_axis=None, ncols=3):
    """Magnitude + phase of each selected DMD mode shape vs state index."""
    r = len(selected)
    ncols = min(ncols, r)
    nrows = int(np.ceil(r / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)

    n_state = len(selected[0]['shape'])
    x = state_axis if state_axis is not None else np.arange(n_state)

    for i, mode in enumerate(selected):
        ax = axes[i // ncols, i % ncols]
        shape = mode['shape']
        ax.plot(x, np.abs(shape), label='|magnitude|')
        ax2 = ax.twinx()
        ax2.plot(x, np.angle(shape), color='tab:orange', alpha=0.6, label='phase')
        growth = mode['omega'].real
        freq = mode['omega'].imag / (2 * np.pi)
        ax.set_title(f"growth={growth:.3g}, f={freq:.3g}\n"
                     f"importance={mode['importance']:.3g}")
        ax.set_xlabel('state index')
        ax.set_ylabel('magnitude')
        ax2.set_ylabel('phase [rad]')

    for j in range(r, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    fig.tight_layout()
    return fig


def plot_dmd_time_coefficients(A, dt, selected):
    """Real time coefficient a(t) per selected mode."""
    r = A.shape[1]
    t = np.arange(A.shape[0]) * dt
    fig, axes = plt.subplots(r, 1, figsize=(8, 2 * r), sharex=True, squeeze=False)
    for i, mode in enumerate(selected):
        ax = axes[i, 0]
        ax.plot(t, A[:, i])
        freq = mode['omega'].imag / (2 * np.pi)
        ax.set_ylabel(f"f={freq:.3g}")
    axes[-1, 0].set_xlabel('time')
    fig.suptitle('DMD mode time coefficients')
    fig.tight_layout()
    return fig


# ============================================================================
# Example end-to-end usage
# ============================================================================
if __name__ == '__main__':
    # X : ndarray, shape (n_time, n_state)   e.g. (8000, 200)
    # dt = ...
    #
    # eigvals, omega, modes, amplitudes, S = compute_dmd(X, dt, r=20)
    #
    # # sanity check on rank choice:
    # plt.semilogy(S, 'o-'); plt.title('SVD spectrum (rank selection)'); plt.show()
    #
    # r_select = 6
    # selected = select_top_dmd_modes(eigvals, omega, modes, amplitudes, r_select)
    #
    # A = get_dmd_time_coefficients(X, selected, modes)   # shape (n_time, r_select)
    # # -> pass A in place of `q` to run_volterra_identification_sequential
    #
    # plot_dmd_spectrum(eigvals, amplitudes, selected=selected)
    # plot_dmd_mode_shapes(selected)
    # plot_dmd_time_coefficients(A, dt, selected)
    #
    # # optional: check overall reconstruction quality
    # t = np.arange(X.shape[0]) * dt
    # X_recon = reconstruct_dynamics(modes, omega, amplitudes, t)
    # print('reconstruction NMSE:', np.mean((X - X_recon)**2) / np.var(X))
    #
    # plt.show()
    pass
