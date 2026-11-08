import numpy as np
from scipy.special import eval_laguerre
from glob import glob
import pyvista as pv
from tqdm import tqdm
import csv 
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

"""
Spectral POD (SPOD), following Towne, Schmidt & Colonius (2018).

Standard POD does ONE eigendecomposition of the time-domain covariance
matrix X @ X.T, giving spatial modes ranked by total energy, with each
mode's time behavior captured in a single time-coefficient vector a(t).

SPOD instead:
  1. splits the time series into overlapping (Welch-style) segments,
  2. FFTs each segment,
  3. does a SEPARATE eigendecomposition at every frequency, on the
     cross-spectral density estimated from all segments at that frequency.

The result is a set of modes PER FREQUENCY, each with its own eigenvalue
(= energy content of that mode at that frequency). Modes are complex and
frequency-coherent by construction, unlike POD modes which mix all
frequencies together into one spatial pattern.

Input convention here (to match your existing pipeline):
    X : ndarray, shape (n_time, n_state)   e.g. (8000, 200)
        i.e. same orientation as your `A_modes`/snapshot arrays: rows are
        time steps, columns are the 200 series.
"""


def _get_window(name, n):
    if name == 'hann':
        return np.hanning(n)
    elif name == 'hamming':
        return np.hamming(n)
    elif name == 'box' or name is None:
        return np.ones(n)
    else:
        raise ValueError(f"Unknown window '{name}'")


def compute_spod(X, dt, nfft=256, overlap=0.5, window='hann', mean_subtract=True):
    """
    Compute SPOD modes, eigenvalues, and frequencies from a time series.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_state)
        Time series data (rows = time steps, columns = state/series index).
    dt : float
        Sampling interval (needed to get physical frequencies out).
    nfft : int
        Segment length (block size) for the Welch-style FFT. Frequency
        resolution is 1/(nfft*dt); larger nfft = finer frequency
        resolution but fewer independent blocks to average over (i.e.
        noisier per-frequency covariance estimate) -- classic Welch
        trade-off.
    overlap : float in [0, 1)
        Fractional overlap between consecutive segments. 0.5 (50%) is
        the common default; higher overlap gives more (correlated)
        blocks, partially recovering statistical convergence lost to a
        larger nfft.
    window : {'hann', 'hamming', 'box'}
        Window applied to each segment before the FFT, to reduce spectral
        leakage. 'hann' is the standard SPOD choice.
    mean_subtract : bool
        Subtract the temporal mean of X before segmenting (standard: SPOD
        modes describe fluctuations about the mean, same convention as POD).

    Returns
    -------
    freqs : ndarray, shape (n_freq,)
        Physical frequencies (cycles per unit time, i.e. Hz if dt is in
        seconds), from 0 up to the Nyquist frequency.
    modes : ndarray, shape (n_freq, n_state, n_modes)
        modes[f, :, i] is the i-th SPOD mode shape at frequency freqs[f],
        complex-valued, ordered by decreasing eigenvalue (energy) at that
        frequency. n_modes = min(n_state, n_blocks).
    eigvals : ndarray, shape (n_freq, n_modes)
        SPOD eigenvalues (energy) at each frequency, decreasing along the
        mode axis. eigvals[f, 0] is the energy of the dominant coherent
        structure at freqs[f].
    """
    n_time, n_state = X.shape

    if mean_subtract:
        X = X - X.mean(axis=0, keepdims=True)

    step = int(nfft * (1 - overlap))
    if step < 1:
        raise ValueError("overlap too close to 1; step size collapsed to 0")

    starts = list(range(0, n_time - nfft + 1, step))
    n_blocks = len(starts)
    if n_blocks < 2:
        raise ValueError(
            f"Only {n_blocks} block(s) fit with nfft={nfft}, overlap={overlap}; "
            "need at least 2 for a meaningful cross-spectral estimate. "
            "Reduce nfft or increase overlap."
        )

    win = _get_window(window, nfft)
    # normalization so windowing doesn't bias the energy estimate
    win_norm = np.sqrt(np.mean(win**2))

    n_freq = nfft // 2 + 1
    freqs = np.fft.rfftfreq(nfft, d=dt)

    # Fourier coefficients for every block, every frequency, every state:
    # shape (n_blocks, n_freq, n_state)
    Q = np.empty((n_blocks, n_freq, n_state), dtype=complex)
    for b, s0 in enumerate(starts):
        seg = X[s0:s0 + nfft, :] * win[:, None] / win_norm
        Q[b] = np.fft.rfft(seg, axis=0) * dt  # scale for physical PSD units

    n_modes = min(n_state, n_blocks)
    modes = np.empty((n_freq, n_state, n_modes), dtype=complex)
    eigvals = np.empty((n_freq, n_modes))

    for f in range(n_freq):
        # Q_f: (n_state, n_blocks) matrix of Fourier coeffs across blocks
        Q_f = Q[:, f, :].T
        # SPOD modes/eigenvalues via SVD of Q_f (equivalent to eigendecomposition
        # of the cross-spectral density S_f = (1/n_blocks) * Q_f @ Q_f.conj().T,
        # but SVD is more numerically robust than forming S_f explicitly)
        U, S, _ = np.linalg.svd(Q_f, full_matrices=False)
        eigvals[f, :] = (S**2) / n_blocks
        modes[f, :, :] = U

    return freqs, modes, eigvals


def dominant_frequency_mode(freqs, modes, eigvals, mode_index=0):
    """
    Convenience: find the frequency at which the mode ranked `mode_index`
    (0 = most energetic at each frequency) reaches its peak eigenvalue
    across all frequencies, and return that frequency + mode shape.

    Useful for identifying "the" dominant coherent structure, analogous to
    picking POD mode i -- but note this is now tied to a specific
    frequency, not a broadband spatial pattern.
    """
    energy_vs_freq = eigvals[:, mode_index]
    f_peak_idx = np.argmax(energy_vs_freq)
    return freqs[f_peak_idx], modes[f_peak_idx, :, mode_index], energy_vs_freq


def spod_mode_time_coefficient(X, mode_shape, dt):
    """
    Project the full (real) time series onto a single complex SPOD mode
    shape to obtain a real-valued time-domain amplitude signal -- this is
    the SPOD analogue of POD's a(t) column, needed if you want to feed a
    mode's time history into your Volterra identification the same way
    you currently use POD's `modes_coeffs`.

    Because a SPOD mode is frequency-coherent by construction, the
    "correct" way to get its time coefficient is to project the analytic
    (Hilbert-transformed) signal onto the mode and take the real part,
    rather than naively taking a real dot-product at every time step
    (which would alias energy from other frequencies into the
    coefficient). This function does the naive real projection, which is
    a reasonable approximation when the mode's frequency is well
    separated from neighboring dynamics; for closely-spaced frequencies,
    band-pass filter X around the mode's frequency first.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_state)
        Original (mean-subtracted) time series.
    mode_shape : ndarray, shape (n_state,), complex
        A single SPOD mode shape (e.g. from dominant_frequency_mode).
    dt : float
        Sampling interval (unused here, kept for interface symmetry /
        future band-pass filtering extensions).

    Returns
    -------
    a : ndarray, shape (n_time,)
        Real-valued time coefficient for this mode.
    """
    phi = mode_shape / np.linalg.norm(mode_shape)
    a = np.real(X @ np.conj(phi))
    return a 
 
# ============================================================================
# 1) Select the r most energetic (frequency, mode-rank) pairs
# ============================================================================
 

def select_top_modes(freqs, modes, eigvals, r):
    """
    Flatten the (n_freq, n_modes) eigenvalue grid and take the r largest
    entries overall -- this is the direct SPOD analogue of "take POD modes
    1..r by descending eigenvalue".
 
    Returns
    -------
    selected : list of dict, length r, each with keys:
        'freq_idx', 'mode_idx', 'freq', 'eigval', 'shape' (n_state,) complex
        sorted by descending eigval.
    """
    n_freq, n_modes = eigvals.shape
    flat_idx = np.argsort(eigvals, axis=None)[::-1][:r]
    f_idx, m_idx = np.unravel_index(flat_idx, eigvals.shape)
 
    selected = []
    for f_i, m_i in zip(f_idx, m_idx):
        selected.append({
            'freq_idx': int(f_i),
            'mode_idx': int(m_i),
            'freq': freqs[f_i],
            'eigval': eigvals[f_i, m_i],
            'shape': modes[f_i, :, m_i],
        })
    return selected
 
 
# ============================================================================
# 2) Time coefficients for the selected modes
# ============================================================================
 
def _bandpass(X, dt, f0, bandwidth_frac=0.3, order=4, min_rel_bandwidth=0.02):
    """
    Zero-phase Butterworth band-pass around f0.
 
    Uses the second-order-sections (sos) filter representation rather than
    the raw (b, a) polynomial form. For narrow bands -- which is exactly
    what happens for a mode near DC (f0 ~ 0) or near the Nyquist edge --
    the (b, a) polynomial coefficients from `butter` become extremely
    ill-conditioned (poles cluster tightly), and filtfilt's forward/backward
    recursion then amplifies floating-point error explosively (values like
    1e50+ are a classic symptom, not a real signal). SOS filtering cascades
    low-order stable stages instead and avoids this blow-up.
 
    As a second safeguard, if the requested band is narrower than
    `min_rel_bandwidth` of the Nyquist frequency (i.e. genuinely too narrow
    to filter robustly, typically for f0 very close to 0), filtering is
    skipped entirely and the unfiltered signal is projected directly --
    the same fallback already used for the degenerate lo >= hi case.
    """
    fs = 1.0 / dt
    nyq = fs / 2
    bw = max(f0 * bandwidth_frac, 1e-6)
    lo = max(f0 - bw, 1e-6)
    hi = min(f0 + bw, nyq * 0.99)
 
    if lo >= hi or (hi - lo) / nyq < min_rel_bandwidth:
        return X  # band too narrow / degenerate (e.g. f0 ~ 0); skip filtering
 
    sos = butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, X, axis=0)
 
 
def get_mode_time_coefficients(X, dt, selected, bandpass=True, bandwidth_frac=0.3):
    """
    For each selected SPOD mode, return a real-valued time coefficient a(t),
    suitable as a target column for run_volterra_identification_sequential
    (same role as your current modes_coeffs[:, i]).
 
    bandpass=True filters X around each mode's own frequency before
    projecting, which avoids energy from other frequencies aliasing into
    the coefficient (see the caveat in spod_mode_time_coefficient).
    """
    X0 = X - X.mean(axis=0, keepdims=True)
    A = np.empty((X.shape[0], len(selected)))
    for i, mode in enumerate(selected):
        Xf = _bandpass(X0, dt, mode['freq'], bandwidth_frac) if bandpass else X0
        A[:, i] = spod_mode_time_coefficient(Xf, mode['shape'], dt)
    return A  # shape (n_time, r) -- pass this in place of `q` / modes_coeffs
 
 
# ============================================================================
# 3) Plots
# ============================================================================
 
def plot_spod_spectrum(freqs, eigvals, n_ranks=4, selected=None, ax=None):
    """
    Classic SPOD eigenvalue spectrum: energy vs frequency, one curve per
    mode rank (rank 0 = most energetic at each frequency). This is the
    standard diagnostic plot from Towne/Schmidt/Colonius papers -- a clear
    gap between rank-0 and rank-1 curves at a given frequency indicates a
    well-separated, low-rank coherent structure there.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for rank in range(min(n_ranks, eigvals.shape[1])):
        ax.semilogy(freqs, eigvals[:, rank], label=f'mode rank {rank}')
    if selected is not None:
        sel_f = [m['freq'] for m in selected]
        sel_e = [m['eigval'] for m in selected]
        ax.scatter(sel_f, sel_e, color='k', zorder=5, marker='x',
                   label='selected (top r)')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('SPOD eigenvalue (energy)')
    ax.set_title('SPOD eigenvalue spectrum')
    ax.legend()
    return ax
 
 
def plot_mode_shapes(selected, state_axis=None, ncols=3):
    """
    Plot |mode shape| (and phase) vs state index for each selected mode --
    the SPOD analogue of plotting POD spatial modes.
    """
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
        ax.set_title(f"f = {mode['freq']:.3g}, rank {mode['mode_idx']}\n"
                     f"eig = {mode['eigval']:.3g}")
        ax.set_xlabel('state index')
        ax.set_ylabel('magnitude')
        ax2.set_ylabel('phase [rad]')
 
    for j in range(r, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
 
    fig.tight_layout()
    return fig
 
 
def plot_mode_time_coefficients(A, dt, selected):
    """
    Plot the extracted real-valued time coefficient a(t) for each selected
    mode -- same role as plotting POD's temporal coefficients, and this is
    literally what you'd feed to the Volterra identification as one of
    the `modes_coeffs` columns.
    """
    r = A.shape[1]
    t = np.arange(A.shape[0]) * dt
    fig, axes = plt.subplots(r, 1, figsize=(8, 2 * r), sharex=True, squeeze=False)
    for i, mode in enumerate(selected):
        ax = axes[i, 0]
        ax.plot(t, A[:, i])
        ax.set_ylabel(f"f={mode['freq']:.3g}")
    axes[-1, 0].set_xlabel('time')
    fig.suptitle('SPOD mode time coefficients')
    fig.tight_layout()
    return fig
  

def generate_schroeder_multisine(freqs, dt, duration, amp=1.0):
    t = np.arange(0, duration, dt)
    N = len(freqs)
    # Schroeder Phase Formula: phi_k = -k*(k-1)*pi / N
    phases = [-k * (k - 1) * np.pi / N for k in range(  1, N + 1)]
    
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += amp * np.sin(2 * np.pi * f * t + phases[i])
    
    # Normalize to keep within original amplitude constraints
    # signal = signal / np.max(np.abs(signal)) * amp
    return t, signal

def build_regularization_matrices(n_coeffs, m, decay_rate=5.0):
    """
    Same D2 (smoothness) and L (decay) matrices as RegularizationComp,
    so results are directly comparable to the OpenMDAO pipeline.
    """
    n_orders = n_coeffs // m
 
    D2_single = np.diff(np.eye(m), n=2, axis=0)
    L_single = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
 
    D2 = block_diag(*[D2_single for _ in range(n_orders)])
    L = block_diag(*[L_single for _ in range(n_orders)])
 
    return D2, L
 
 
# ============================================================================
# Option 1: ridge-only regularization via SVD (minimal change from your code)
# ============================================================================
 
def solve_ridge_svd(A, ys, alpha=1e-3):
    """
    Tikhonov/ridge-regularized least squares via a single SVD, reused for
    every column of `ys`:
 
        h = argmin_h ||A h - y||^2 + alpha * ||h||^2
          = V @ diag(S / (S**2 + alpha)) @ U.T @ y
 
    This is the direct regularized analogue of your pseudoinverse formula
    (which is the alpha=0 case), and computes the SVD of A only ONCE
    regardless of how many y columns you solve for -- your original code
    recomputed U, S, Vt identically for mode_5 and mode_6, which is
    redundant since A_123 doesn't change between them.
 
    Parameters
    ----------
    A : ndarray, shape (n_samples, n_coeffs)
    ys : ndarray, shape (n_samples,) or (n_samples, n_targets)
        One or more target vectors (e.g. A_modes[:, [4, 5]]).
    alpha : float
        Ridge strength. alpha=0 recovers your original (unregularized)
        pseudoinverse solve.
 
    Returns
    -------
    h : ndarray, shape (n_coeffs,) or (n_coeffs, n_targets)
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    filt = S / (S**2 + alpha)          # damped singular-value filter
    if ys.ndim > 1:
        h = Vt.T @ (filt[:, None] * (U.T @ ys))
    else:
        h = Vt.T @ (filt * (U.T @ ys))
    return h
 
 
# ============================================================================
# Option 2: full ridge + smoothness + decay regularization, matching
# RegularizationComp exactly, solved once via Cholesky and reused for
# every mode.
# ============================================================================
 
def solve_full_regularized(A, ys, m,
                            lambda_ridge=1e-3,
                            lambda_smooth=1e-2,
                            lambda_decay=1e-1,
                            decay_rate=5.0):
    """
    Closed-form solve of:
 
        min_h ||A h - y||^2
              + lambda_ridge   * ||h||^2
              + lambda_smooth  * ||D2 h||^2
              + lambda_decay   * h.T @ L @ h
 
    i.e. the exact convex problem your VoltterraResidualComp +
    RegularizationComp encode, but solved directly instead of via SLSQP.
    The normal-form system matrix
 
        M = A.T @ A + lambda_ridge*I + lambda_smooth*D2.T@D2 + lambda_decay*L
 
    is shared by every kernel/mode (only the right-hand side A.T @ y
    changes), so we factor it ONCE (Cholesky) and reuse that factorization
    for every column of `ys` -- this is the generalization of "compute the
    SVD once" to the full regularization suite, and is both faster and
    exact (no optimizer convergence to worry about).
 
    Note: uses normal equations (A.T @ A), so if A is very ill-conditioned
    consider passing a pre-whitened/pre-conditioned A, or fall back to
    solve_ridge_svd for the ridge-only case, which avoids squaring the
    condition number.
 
    Parameters
    ----------
    A : ndarray, shape (n_samples, n_coeffs)
    ys : ndarray, shape (n_samples,) or (n_samples, n_targets)
    m : int
        Memory length per kernel order (needed to build D2, L consistently
        with RegularizationComp).
 
    Returns
    -------
    h : ndarray, shape (n_coeffs,) or (n_coeffs, n_targets)
    """
    n_samples, n_coeffs = A.shape
    D2, L = build_regularization_matrices(n_coeffs, m, decay_rate=decay_rate)
 
    M = (A.T @ A
         + lambda_ridge * np.eye(n_coeffs)
         + lambda_smooth * (D2.T @ D2)
         + lambda_decay * L)
 
    rhs = A.T @ ys  # shape (n_coeffs,) or (n_coeffs, n_targets)
 
    c, low = cho_factor(M)
    h = cho_solve((c, low), rhs)
    return h
    
AOA = 5
MACH = 80
dt = 2.5e-4

chord = 0.4064 #1. 
v_inf = 123. * MACH / 80 # * 277.24 # 123. 

home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_surfaceprint_test" % AOA

# ============================================================================
# cp values 
# ============================================================================

files = sorted(glob(home_pitch + "/surface_*.vtu"))

Cp_list = []
x_list = []
y_list = []

for f in tqdm(files, desc="Reading SU2 files"):

    mesh = pv.read(f)

    Cp_list.append(mesh['Pressure_Coefficient'])

    x_list.append(mesh.points[:,0])
    y_list.append(mesh.points[:,1])

Cp = np.array(Cp_list)
x  = np.array(x_list)
y  = np.array(y_list)

ntsteps = Cp.shape[0]

X = Cp.copy()

Cp_mean = Cp.mean(axis=0)
X = Cp - Cp_mean

freqs, modes, eigvals = compute_spod(X, dt, nfft=500, overlap=0.5)

r = 8
selected = select_top_modes(freqs, modes, eigvals, r)

A_modes = get_mode_time_coefficients(X, dt, selected)   # shape (n_time, r)
# -> pass A in place of `q` to run_volterra_identification_sequential

plot_spod_spectrum(freqs, eigvals, n_ranks=4, selected=selected)
plot_mode_shapes(selected,x[0,:])
plot_mode_time_coefficients(A_modes, dt, selected)
plt.show()

target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.95 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.065)

zdot_input = np.gradient(plunge_input,t)/v_inf 

m = 96 

A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

A = np.hstack( (A_pitch, A_zdot))
A_pitch2 = A_pitch*A_pitch
A_zdot2 = A_zdot*A_zdot
A_pitch_zdot = A_pitch*A_zdot

A2 = np.hstack( (A_pitch2,A_zdot2,A_pitch_zdot))

# third order
A_pitch3 = A_pitch*A_pitch*A_pitch
A_zdot3 = A_zdot*A_zdot*A_zdot
A_pitch_zdot2 = A_pitch*A_zdot2
A_zdot_pitch2 = A_zdot*A_pitch2

A3 = np.hstack( (A_pitch3,A_zdot3,A_pitch_zdot2, A_zdot_pitch2))

# fourth order
A_pitch4 = A_pitch*A_pitch*A_pitch*A_pitch
A_zdot4 = A_zdot*A_zdot*A_zdot*A_zdot
A_pitch2_zdot2 = A_pitch*A_pitch*A_zdot*A_zdot
A_pitch_zdot3 = A_pitch*A_zdot3
A_pitch3_zdot = A_zdot*A_pitch3

A4 = np.hstack( (A_pitch4,A_zdot4,A_pitch2_zdot2, A_pitch3_zdot,A_pitch_zdot3))

A_123 = np.hstack( (A, A2, A3) )
# A_123 = np.hstack( (A, A2, A3, A4) )


r = A_modes.shape[1]

h_modes = []
modes_check = []

for i in range(r):

    h_i = solve_full_regularized(
        A_123,
        A_modes[:, i],
        m,
        lambda_ridge=1e-3,
        lambda_smooth=1e-2,
        lambda_decay=1e-1,
        decay_rate=5.0
    )

    h_modes.append(h_i)

    modes_check.append(A_123 @ h_i)

h_modes = np.array(h_modes)
modes_check = np.array(modes_check)

import math

ncols = 2
nrows = math.ceil(r / ncols)

plt.figure(figsize=(12, 3*nrows))

for i in range(r):

    plt.subplot(nrows, ncols, i+1)

    plt.plot(modes_check[i], 'b-', linewidth=1,
             label='Volterra')

    plt.plot(A_modes[:, i], 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# DMD
from dmd import compute_dmd, reconstruct_dynamics
from dmd_analysis import (
    select_top_dmd_modes,
    get_dmd_time_coefficients,
    plot_dmd_spectrum,
    plot_dmd_mode_shapes,
    plot_dmd_time_coefficients,
)

eigvals, omega, modes, amplitudes, S = compute_dmd(X, dt, r=20)

# sanity check on rank choice:
plt.semilogy(S, 'o-'); plt.title('SVD spectrum (rank selection)'); plt.show()

r_select = r
selected = select_top_dmd_modes(eigvals, omega, modes, amplitudes, r_select)

A_modes = get_dmd_time_coefficients(X, selected, modes)   # shape (n_time, r_select)
# -> pass A in place of `q` to run_volterra_identification_sequential

plot_dmd_spectrum(eigvals, amplitudes, selected=selected)
plot_dmd_mode_shapes(selected,x[0,:])
plot_dmd_time_coefficients(A, dt, selected)

# optional: check overall reconstruction quality
t = np.arange(X.shape[0]) * dt
X_recon = reconstruct_dynamics(modes, omega, amplitudes, t)
print('reconstruction NMSE:', np.mean((X - X_recon)**2) / np.var(X))

plt.show()

r = A_modes.shape[1]

h_modes = []
modes_check = []

for i in range(r):

    h_i = solve_full_regularized(
        A_123,
        A_modes[:, i],
        m,
        lambda_ridge=1e-3,
        lambda_smooth=1e-2,
        lambda_decay=1e-1,
        decay_rate=5.0
    )

    h_modes.append(h_i)

    modes_check.append(A_123 @ h_i)

h_modes = np.array(h_modes)
modes_check = np.array(modes_check)

import math

ncols = 2
nrows = math.ceil(r / ncols)

plt.figure(figsize=(12, 3*nrows))

for i in range(r):

    plt.subplot(nrows, ncols, i+1)

    plt.plot(modes_check[i], 'b-', linewidth=1,
             label='Volterra')

    plt.plot(A_modes[:, i], 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# PCA
from pca import compute_pca, select_top_r

components, scores, ev, ev_ratio, mean_, scale_ = compute_pca(X, standardize=False)
comp_r, A_modes, ev_ratio_r = select_top_r(scores, components, ev_ratio, r=6)
# scores_r is (n_time, r) -- drop straight into run_volterra_identification_sequential


r = A_modes.shape[1]

h_modes = []
modes_check = []

for i in range(r):

    h_i = solve_full_regularized(
        A_123,
        A_modes[:, i],
        m,
        lambda_ridge=1e-3,
        lambda_smooth=1e-2,
        lambda_decay=1e-1,
        decay_rate=5.0
    )

    h_modes.append(h_i)

    modes_check.append(A_123 @ h_i)

h_modes = np.array(h_modes)
modes_check = np.array(modes_check)

import math

ncols = 2
nrows = math.ceil(r / ncols)

plt.figure(figsize=(12, 3*nrows))

for i in range(r):

    plt.subplot(nrows, ncols, i+1)

    plt.plot(modes_check[i], 'b-', linewidth=1,
             label='Volterra')

    plt.plot(A_modes[:, i], 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

