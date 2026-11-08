"""
Down-to-earth Fourier modes on a wing-section surface.

Unlike POD/SPOD/DMD/ICA, these modes are NOT derived from your data at
all -- they're a fixed, analytic basis:

    y_i(s) = sin(2*pi*(i+1)*s/chord),   i = 0, ..., r-1

defined purely from the geometry (the surface coordinate s and chord
length). Because they're fixed, "extracting mode i" doesn't involve any
decomposition step -- you just evaluate the formula. Getting a TIME
coefficient still requires projecting your actual surface data onto each
mode shape, same final step as for SPOD/DMD.

Input convention:
    x : ndarray, shape (n_state,)
        Streamwise (or surface-ordered) coordinate of each of your 200
        points, in the SAME order as the columns of your data matrix X.
    X : ndarray, shape (n_time, n_state)
        Same orientation as your POD/SPOD/DMD pipeline.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def stretch_coordinate_cosine(x):
    """
    Smooth, parameter-free coordinate stretch that clusters resolution
    near BOTH ends of the range (x close to its min -> "leading edge", x
    close to its max -> "trailing edge"), leaving the mid-range coarser.

    Standard cosine-clustering transform, normalized to operate on x's
    own [min, max] range and mapped back to that same physical range
    (so units/scale of x are preserved -- only the internal spacing
    changes):

        x_norm = (x - x_min) / (x_max - x_min)          in [0, 1]
        stretched_norm = 0.5 * (1 - cos(pi * x_norm))    in [0, 1]
        x_stretched = x_min + stretched_norm * (x_max - x_min)

    Because stretched_norm's derivative w.r.t. x_norm is small near 0
    and 1 (and largest at the midpoint), evenly-spaced steps in x near
    the edges collapse into even smaller steps in x_stretched -- i.e.
    more "surface coordinate" resolution gets allocated to the edges.

    Parameters
    ----------
    x : ndarray, shape (n_state,)

    Returns
    -------
    x_stretched : ndarray, shape (n_state,), same range as x.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    stretched_norm = 0.5 * (1 - np.cos(np.pi * x_norm))
    return x_min + stretched_norm * (x_max - x_min)


def stretch_coordinate_tanh(x, beta=3.0):
    """
    Same purpose as stretch_coordinate_cosine (cluster resolution near
    both ends), but with an adjustable aggressiveness parameter `beta`:
    larger beta = more aggressive clustering right at the edges, smaller
    beta approaches a linear (no-op) mapping. This is the classic
    hyperbolic-tangent (Vinokur-style) stretching used for boundary-layer
    mesh clustering.

    Parameters
    ----------
    x : ndarray, shape (n_state,)
    beta : float > 0
        Clustering strength. Try values in roughly [1, 5]; beta -> 0
        approaches no stretching (identity map), larger beta pushes more
        of the range's resolution toward x_min and x_max.

    Returns
    -------
    x_stretched : ndarray, shape (n_state,), same range as x.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)

    tb = np.tanh(beta)
    stretched_norm = 0.5 * (1 + np.tanh(beta * (2 * x_norm - 1)) / tb)
    return x_min + stretched_norm * (x_max - x_min)


def compute_surface_coordinate(x, y=None):
    """
    Build a monotonic surface distance coordinate s from x=0.

    Two modes:
    - y is None: s(t) = cumulative integral of x itself along the point
      ordering (i.e. treating x as a "rate" and integrating over the
      index), matching s = int_0^s x dx as you wrote it:
          s_k = trapz(x[0:k+1], dx=1)   (cumulative, s_0 = 0)
    - y is provided: true arc length along the (x, y) surface curve,
          ds = sqrt(dx^2 + dy^2),  s_k = cumulative sum of ds
      This is the geometrically correct "distance from x=0 along the
      surface" if you have both surface coordinates -- worth using this
      version if you have y available, since it's the standard definition
      of arc length and doesn't depend on how densely points are spaced.

    Returns
    -------
    s : ndarray, shape (n_state,), with s[0] = 0, monotonically increasing
        (assuming x, or the (x,y) path, doesn't reverse direction).
    """
    x = np.asarray(x, dtype=float)

    if y is None:
        s = cumulative_trapezoid(x, initial=0.0)
    else:
        y = np.asarray(y, dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])

    return s


def build_fourier_modes(s, chord, r):
    """
    Build the r analytic Fourier sine modes.

    Parameters
    ----------
    s : ndarray, shape (n_state,)
        Surface coordinate from compute_surface_coordinate.
    chord : float
        Chord length used to normalize s. If you don't have a separate
        chord value, s[-1] (total surface distance covered) is a
        reasonable default -- see the example below.
    r : int
        Number of modes.

    Returns
    -------
    modes : ndarray, shape (n_state, r)
        modes[:, i] = sin(2*pi*(i+1)*s/chord), i = 0..r-1.
    """
    s = np.asarray(s, dtype=float)
    i = np.arange(r) + 1  # i+1, per your formula
    # outer product: (n_state, 1) * (1, r) -> (n_state, r)
    modes = np.sin(2 * np.pi * np.outer(s, i) / chord)
    return modes


def project_onto_modes(X, modes):
    """
    Least-squares projection of your actual data onto the fixed Fourier
    modes, to get a real time coefficient per mode -- this is the
    equivalent final step to SPOD/DMD's projection, needed because these
    modes are analytic, not data-derived, so there's no automatic
    "coefficient" the way POD's SVD directly hands you one.

    Least squares (via pinv) is used rather than a plain inner product
    because the sine modes are only exactly orthogonal for evenly-spaced s
    over a full set of harmonics on [0, chord] with matching boundary
    conditions -- real airfoil surface points are rarely evenly spaced in
    s, so treating this as a general linear least-squares fit is the
    robust choice regardless of point spacing.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_state)
    modes : ndarray, shape (n_state, r)

    Returns
    -------
    A : ndarray, shape (n_time, r)
        Time coefficient per mode -- drop this in place of `q` /
        `modes_coeffs` for run_volterra_identification_sequential.
    """
    modes_pinv = np.linalg.pinv(modes)      # (r, n_state)
    A = (modes_pinv @ X.T).T                # (n_time, r)
    return A


# ============================================================================
# Plotting
# ============================================================================

def plot_fourier_modes(s, modes, x=None):
    """
    Plot each mode shape vs the surface coordinate s (or vs x, if you
    prefer the physical chordwise coordinate on the axis -- pass x too).
    """
    import matplotlib.pyplot as plt

    r = modes.shape[1]
    fig, ax = plt.subplots(figsize=(7, 4))
    axis = x if x is not None else s
    label = 'x' if x is not None else 's'
    for i in range(r):
        ax.plot(axis, modes[:, i], label=f'mode {i+1}')
    ax.set_xlabel(label)
    ax.set_ylabel('mode amplitude')
    ax.set_title('Analytic Fourier surface modes')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mode_time_coefficients(A, dt):
    import matplotlib.pyplot as plt

    r = A.shape[1]
    t = np.arange(A.shape[0]) * dt
    fig, axes = plt.subplots(r, 1, figsize=(8, 2 * r), sharex=True, squeeze=False)
    for i in range(r):
        axes[i, 0].plot(t, A[:, i])
        axes[i, 0].set_ylabel(f'mode {i+1}')
    axes[-1, 0].set_xlabel('time')
    fig.suptitle('Fourier mode time coefficients')
    fig.tight_layout()
    return fig


# ============================================================================
# Example usage
# ============================================================================
if __name__ == '__main__':
    # x : ndarray, shape (n_state,)   surface x-coordinate of each of your 200 points
    # X : ndarray, shape (n_time, n_state)
    # dt = ...
    #
    # s = compute_surface_coordinate(x)          # or compute_surface_coordinate(x, y)
    # chord = s[-1]                              # or your known physical chord length
    #
    # r = 6
    # modes = build_fourier_modes(s, chord, r)   # (n_state, r)
    #
    # A = project_onto_modes(X, modes)           # (n_time, r)
    # # -> pass A in place of `q` to run_volterra_identification_sequential
    #
    # plot_fourier_modes(s, modes, x=x)
    # plot_mode_time_coefficients(A, dt)
    pass

