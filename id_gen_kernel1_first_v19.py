#!/usr/bin/env python3
"""
Volterra identification of the generalized forces (pitch / heave input -> CL, CM output)
for the data of Michael, looping over the angle of attack for ONE grid ("fine" or "medium").

    python identify_bscw.py                       # GRID and AOAS_BY_GRID below
    python identify_bscw.py --grid medium         # medium grid, AOA 1 ... 5
    python identify_bscw.py --grid fine --aoa 4   # one case only

For every AOA it
  - loads the training and the test set of the chosen grid,
  - identifies the kernels (regularized least squares, optional low-pass filter / transient skip),
  - saves the figures in OUT_DIR and the kernels in <home_pitch>/kernels_1_to_4.npz (same keys as before,
    plus some extra ones),
  and at the end it writes OUT_DIR/summary_<grid>.csv with the training and test errors.
"""

import os
import csv
import argparse

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, sosfiltfilt
from numpy.linalg import LinAlgError

# ADJUST: module that contains build_A123 and build_regularization_matrices
from regularized_mode_solve_v3 import (
    # regression matrices and block layout
    build_A123, build_A_band, build_A_band3,
    block_layout, block_layout3, block_slices,
    # regularization and fitting
    build_regularization_matrices, fit_blocks, solve_full_regularized,
    # windowed fits and errors
    window_bounds, fit_windows, window_predictions, cross_errors,
    plot_window_predictions,
    # other
    band_products, third_band_spec, third_band_products, noisy_signals,
)

from pathlib import Path

# ------------------------------------------------------------------ new dataset
NEW_DATA_DIR = Path("/Users/marcello/Documents/SU2/AePW4_TC3/identification/data")        # adjust (absolute path)
OUT_DIR = Path("/Users/marcello/Documents/SU2/AePW4_TC3/identification/kernels")  
FIG_FORMAT = "png" 
SHOW = False   

# NEW_AOAS     = [0, 1, 2, 3, 4, 5, 6] #[0, 1, 2, 3, 4, 5, 6]
NEW_AOAS     = [0] #, 1, 2] #[0, 1, 2, 3, 4, 5, 6]
# TURB_MODELS  = ["A", "B", "C", "D"]
# TURB_MODELS  = ["B", "C", "D"]
TURB_MODELS  = ["B"]
PITCH_IN_DEG = False        # CHECK: True if PITCH is in degrees -> converted to rad
# DELTA_T, V_INF, CHORD, M (top of the file) must be set for this dataset.
# ZDOTOV is assumed to be hdot/V already (same quantity as 'zdot' before).
MACH = 0.8

# optional: use only the first N time steps of the TRAINING set of a case, e.g. {("fine", 5): 3150}
NT_MAX = {}

V_INF = 123.453                                 # [m/s] -> 123.453 SU2, M=0.80 
RHO_INF = 0.832                                 # [kg/m^3]
DELTA_T = 0.0001                                # time step [s]
CHORD = 0.4064
RESAMPLE = 1                                    # use every RESAMPLE-th sample (identification AND test)

NORM_FACTOR = 1.0                               # input normalization (train and test)
CL_NORM_FACTOR = 1.0
CM_NORM_FACTOR = 1.0
DETREND_TEST = "own"                            # "own": subtract the means of the test set (as before)
                                                # "train": subtract the means of the training set

# --- regularization
LAMBDA_RIDGE = 1e-8 #1e-8
LAMBDA_SMOOTH = 1e-5 #1e-5 1e-4
LAMBDA_DECAY = 1e-4 #1e-5
DECAY_RATE = 9.0 #5.0
JITTERS = (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)   # relative ridge tried if Cholesky fails

# --- optional low-pass filter of CL and CM (training and test), e.g. shock buffet
FILTER_ON = True
CUTOFF = 80.0                                  # [Hz]
FILTER_ORDER = 4

# --- optional exclusion of the start-up transient of the training set (equations removed, inputs kept)
SKIP = 0                                        # first SKIP samples
RAMP = 5000                                        # half-Hann ramp 0 -> 1 over the next RAMP samples
SKIP_END = 0                                    # last SKIP_END samples
# ==========================================================================
BAND       = 4 #6        # 0 reproduces the old model; then try 2, 4, 8
LAMBDA_BAND   = 1e-3     # ridge growing with the lag distance |d|
BAND_DECAY = 3.0 #3.0
BAND3 = 4 #2 #8

INPUT_GAIN = 15. #18.

M = 64 #312                                         # memory length of each kernel [samples]
ORDER = 4 
SWAP_TRAIN_TEST = False      # with the other settings at the top

## ---------------------------------------------------------------- settings
# add next to the other settings (after SKIP, RAMP, SKIP_END)
USE_SMALL = True          # use "train_small" when True AND the file exists for the case; if it is
                          # missing the case falls back to the single fit on "train" (with a message)
SMALL_REQUIRED = False    # True: a missing "train_small" file skips the case instead of falling back
# --- fitting method (needs ident_extras.py in the same folder for anything but "classic")
FIT_METHOD = "gram"    # "classic": fit_blocks as before (absolute penalties)
                          # "gram"   : relative penalties, from the Gram matrix (ident_extras)
TUNE_LAMBDAS = True      # gram: one penalty multiplier per order, chosen by blocked cross-validation
CV_FOLDS = 4              # gram + tuning: contiguous folds (joint: see CV_BY_RECORD)
CV_BY_RECORD = False      # joint + tuning: one fold per record (small / large) instead of CV_FOLDS
GREEDY = True            # gram: forward selection of blocks instead of fitting all of them
GREEDY_FORCED = ("p", "z")  # blocks that are always in
GREEDY_MAX = None         # maximum number of blocks added (None = until GREEDY_TOL)
GREEDY_TOL = 1e-3         # stop when the best block reduces the error by less than this (relative)
PLOT_FRF = True           # plot H1(f), real and imaginary parts: kernel vs Y/U from the data
FRF_FMAX = None           # [Hz] upper limit of the plot; None = CUTOFF if FILTER_ON else Nyquist
FRF_WINDOW = "hann"       # window of the DFT of the fully weighted rows: "hann" or None
FRF_LINE_THR = 0.05       # a spectral line must exceed this fraction of the input's largest line
FRF_LINE_RATIO = 5.0      # ... and dominate the other input at that frequency by this factor
USE_TEST = True           # evaluate on "test" when True AND the file exists; otherwise the case is
                          # identified and plotted without any test comparison
SMALL_MODE = "sequential"      # "joint":      one regression on both sets, all coefficients shared
                          # "sequential": H1 from the small set, then orders >= 2 on the large set
SMALL_WEIGHT = 1.0        # joint only: extra factor on the small-set rows, on top of the automatic
                          # equalization of the relative errors (1 = both sets count equally)
SMALL_SKIP, SMALL_RAMP, SMALL_SKIP_END = SKIP, RAMP, SKIP_END
 
 
# ---------------------------------------------------------------- case_files
def case_files(aoa, model):
    d = NEW_DATA_DIR / f"AOA_{aoa}" / model
    train, test = d / "training.csv", d / "test.csv"
    if SWAP_TRAIN_TEST:
        train, test = test, train
    return dict(
        home=os.path.join(OUT_DIR, f"AOA_{aoa}", model),
        train=str(train),
        test=str(test),
        train_small=str(d / "training_small.csv"),
        steady=str(d / "steady.csv"),
    )

# load_dataset: no change needed. Its "kind == 'train'" truncation (NT_MAX) does not
# match "train_small", so the small set is read in full.

def load_dataset(aoa, model, kind, files):
    c = read_cols(files[kind], keys=("PITCH", "ZDOTOV", "CL", "CM"))
    n = len(c["CL"])
    if kind == "train":
        n = min(n, NT_MAX.get((aoa, model), n))

    pitch = c["PITCH"][:n] * (np.pi / 180 if PITCH_IN_DEG else 1.0)   # [rad]
    zdot  = c["ZDOTOV"][:n]                                           # hdot / V
    thetadot = np.gradient(pitch, DELTA_T) / V_INF    # only if condition()/others use it
    cl, cm = c["CL"][:n], c["CM"][:n]                 # already coefficients: no scaling
    t = np.arange(n) * DELTA_T

    sl = slice(None, None, RESAMPLE)
    return dict(t=t[sl], pitch=pitch[sl], zdot=zdot[sl], thetadot=thetadot[sl],
                cl=cl[sl], cm=cm[sl])

def butter_filter(data, btype, cutoff, fs, order=4):
    """Zero-phase Butterworth filter (second-order sections)."""
    nyq = 0.5 * fs
    Wn = [cutoff[0] / nyq, cutoff[1] / nyq] if btype == "band" else cutoff / nyq
    sos = butter(order, Wn, btype=btype, output="sos")
    return sosfiltfilt(sos, data)


def read_cols(path, keys):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}


def condition(ds, means=None):
    """Remove the means, apply the normalization factors, optional low-pass filter. Identical for
    training and test set (the means come from `means`, or from the set itself if None)."""
    if means is None:
        means = {k: float(np.mean(ds[k])) for k in ("pitch", "zdot", "cl", "cm")}
    out = dict(ds)
    out["pitch"] = (ds["pitch"] - means["pitch"]) * NORM_FACTOR
    out["zdot"] = (ds["zdot"] - means["zdot"]) * NORM_FACTOR
    out["cl_raw"] = (ds["cl"] - means["cl"]) * CL_NORM_FACTOR
    out["cm_raw"] = (ds["cm"] - means["cm"]) * CM_NORM_FACTOR
    if FILTER_ON:
        fs = 1.0 / (DELTA_T * RESAMPLE)
        out["cl"] = butter_filter(out["cl_raw"], "low", CUTOFF, fs, FILTER_ORDER)
        out["cm"] = butter_filter(out["cm_raw"], "low", CUTOFF, fs, FILTER_ORDER)
    else:
        out["cl"], out["cm"] = out["cl_raw"], out["cm_raw"]
    return out


def rel_err(pred, ref, mask=None):
    if mask is not None:
        pred, ref = pred[mask], ref[mask]
    return np.linalg.norm(pred - ref, axis=0) / np.linalg.norm(ref, axis=0)


# ------------------------------------------------------------------ figures
def finish(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"new_{name}.{FIG_FORMAT}")
    fig.savefig(path, dpi=130 if FIG_FORMAT == "png" else None)
    if SHOW:
        plt.show()
    plt.close(fig)
    return path

def plot_data(aoa, raw):
    tau = raw["t"] * 2 * V_INF / CHORD
    t = raw["t"]
    fig, ax = plt.subplots(6, 1, figsize=(10, 11))
    ax[0].plot(tau, raw["cm"]); ax[0].set(xlabel=r"$\tau$", ylabel=r"$C_m$")
    ax[1].plot(tau, raw["cl"]); ax[1].set(xlabel=r"$\tau$", ylabel=r"$C_l$")
    ax[2].plot(t, raw["zdot"] * 180 / np.pi); ax[2].set(xlabel="t [s]", ylabel=r"$\dot h / V$ [deg]")
    ax[3].plot(t, raw["pitch"] * 180 / np.pi); ax[3].set(xlabel="t [s]", ylabel=r"$\theta$ [deg]")
    ax[4].plot(t, raw["thetadot"] * 180 / np.pi); ax[4].set(xlabel="t [s]", ylabel=r"$\dot\theta / V$ [deg/m]")
    ax[5].plot(t, (raw["pitch"] + raw["zdot"]) * 180 / np.pi); ax[5].set(xlabel="t [s]", ylabel=r"$\alpha$ [deg]")
    for a in ax:
        a.grid(True)
    fig.suptitle(f"AOA {aoa}: training data")      # aoa is now the tag, e.g. "2_A"
    return finish(fig, f"AOA{aoa}_data")    


def plot_kernels(aoa, H):
    n_orders = H.shape[0] // M
    lag = np.arange(M) * DELTA_T * RESAMPLE * V_INF / (2 * CHORD)       # reduced time
    ncols = 4
    nrows = int(np.ceil(n_orders / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.0 * nrows), sharex=True, squeeze=False)
    for i in range(n_orders):
        ax = axes[i // ncols, i % ncols]
        ax.plot(lag, H[i * M:(i + 1) * M, 0], "-", lw=0.8, label="CL")
        ax.plot(lag, H[i * M:(i + 1) * M, 1], "-", lw=0.8, label="CM")
        ax.set_title(f"kernel {i + 1}", fontsize=9)
        ax.grid(True)
    axes[0, 0].legend(fontsize=7)
    for j in range(n_orders, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"AOA {aoa}: kernels (x: lag in reduced time)")
    return finish(fig, f"AOA{aoa}_kernels")


def plot_fit(aoa, tr, te, fit_tr, fit_te, w, e_tr, e_te):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for col, (ds, fit, e, name) in enumerate(((tr, fit_tr, e_tr, "training"), (te, fit_te, e_te, "test"))):
        tau = ds["t"] * 2 * V_INF / CHORD
        for row, (key, label) in enumerate((("cl", r"$C_L$"), ("cm", r"$C_M$"))):
            ax = axs[row, col]
            if FILTER_ON:
                ax.plot(tau, ds[f"{key}_raw"], color="0.75", lw=0.8, label="reference (raw)")
            ax.plot(tau, ds[key], "k-", lw=1.0, label="reference (low-pass)" if FILTER_ON else "reference")
            ax.plot(tau, fit[:, row], "b-", lw=1.0, label="Volterra")
            if name == "training" and np.any(w < 1.0):
                full = np.flatnonzero(w >= 1.0)
                if full.size:
                    ax.axvspan(tau[0], tau[full[0]], color="0.5", alpha=0.15, lw=0, label="not / partly used")
                    ax.axvspan(tau[full[-1]], tau[-1], color="0.5", alpha=0.15, lw=0)
            ys = [ds[key]] + ([ds[f"{key}_raw"]] if FILTER_ON else [])
            lo, hi = min(float(np.min(y)) for y in ys), max(float(np.max(y)) for y in ys)
            pad = 0.15 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)           # the Volterra curve is unconstrained in skipped rows
            ax.set(xlabel=r"$\tau$", ylabel=label, title=f"{name}: {label[1:-1]} error {e[row]:.2%}")
            ax.grid(True)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"AOA")
    return finish(fig, f"AOA{aoa}_fit")

# ---------------------------------------------------------------- make_weights
def make_weights(n, skip=None, ramp=None, skip_end=None):
    """As before, but the skipped / ramped lengths can be given per dataset."""
    skip = SKIP if skip is None else skip
    ramp = RAMP if ramp is None else ramp
    skip_end = SKIP_END if skip_end is None else skip_end
    w = np.ones(n)
    w[:skip] = 0.0
    if ramp > 0:
        x = np.arange(1, ramp + 1) / (ramp + 1)
        w[skip:skip + ramp] = 0.5 * (1.0 - np.cos(np.pi * x))
    if skip_end > 0:
        w[n - skip_end:] = 0.0
    return w
 
 
# ---------------------------------------------------------------- helpers
def first_order_scale(A_small, A_big, w_small, w_big, m):
    """(sequential mode) Ratio of the mean weighted energy of the first-order columns, small / large.
    The absolute lambdas are multiplied by it so that H1 is regularized as much, relatively, as in
    the full-record fit."""
    e_s = np.sum((w_small[:, None] * A_small[:, :2 * m]) ** 2, axis=0).mean()
    e_b = np.sum((w_big[:, None] * A_big[:, :2 * m]) ** 2, axis=0).mean()
    return e_s / e_b
 
 
def build_A1(pitch, zdot, m):
    """First-order columns only: identical to the first 2M columns of build_A123
    ([pitch lags | zdot lags], zero-padded before t = 0), without the other blocks."""
    n = len(pitch)
    A = np.zeros((n, 2 * m))
    for i in range(m):
        A[i:, i] = pitch[:n - i]
        A[i:, m + i] = zdot[:n - i]
    return A
 
 
def plot_two_sets(tag, sm, tr, fit_sm, fit_tr, w_sm, w, e_sm, e_tr, label, names, suptitle, fname):
    """Small set (left) and large set (right): CL on the top row, CM on the bottom row."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    cols = ((sm, fit_sm, w_sm, e_sm, names[0]), (tr, fit_tr, w, e_tr, names[1]))
    for col, (ds, fit, wt, e, name) in enumerate(cols):
        tau = ds["t"] * 2 * V_INF / CHORD
        for row, (key, ylabel) in enumerate((("cl", r"$C_L$"), ("cm", r"$C_M$"))):
            ax = axs[row, col]
            if FILTER_ON:
                ax.plot(tau, ds[f"{key}_raw"], color="0.75", lw=0.8, label="reference (raw)")
            ax.plot(tau, ds[key], "k-", lw=1.0, label="reference (low-pass)" if FILTER_ON else "reference")
            ax.plot(tau, fit[:, row], "b-", lw=1.0, label=label)
            if np.any(wt < 1.0):
                full = np.flatnonzero(wt >= 1.0)
                if full.size:
                    ax.axvspan(tau[0], tau[full[0]], color="0.5", alpha=0.15, lw=0, label="not / partly used")
                    ax.axvspan(tau[full[-1]], tau[-1], color="0.5", alpha=0.15, lw=0)
            ys = [ds[key]] + ([ds[f"{key}_raw"]] if FILTER_ON else [])
            lo, hi = min(float(np.min(y)) for y in ys), max(float(np.max(y)) for y in ys)
            pad = 0.15 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
            ax.set(xlabel=r"$\tau$", ylabel=ylabel, title=f"{name}: {ylabel[1:-1]} error {e[row]:.2%}")
            ax.grid(True)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(suptitle)
    return finish(fig, fname)
 
 
def plot_fit_train(tag, tr, fit_tr, w, e_tr):
    """Training-only version of plot_fit, used when there is no test set."""
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), squeeze=False)
    tau = tr["t"] * 2 * V_INF / CHORD
    for row, (key, label) in enumerate((("cl", r"$C_L$"), ("cm", r"$C_M$"))):
        ax = axs[row, 0]
        if FILTER_ON:
            ax.plot(tau, tr[f"{key}_raw"], color="0.75", lw=0.8, label="reference (raw)")
        ax.plot(tau, tr[key], "k-", lw=1.0, label="reference (low-pass)" if FILTER_ON else "reference")
        ax.plot(tau, fit_tr[:, row], "b-", lw=1.0, label="Volterra")
        if np.any(w < 1.0):
            full = np.flatnonzero(w >= 1.0)
            if full.size:
                ax.axvspan(tau[0], tau[full[0]], color="0.5", alpha=0.15, lw=0, label="not / partly used")
                ax.axvspan(tau[full[-1]], tau[-1], color="0.5", alpha=0.15, lw=0)
        ys = [tr[key]] + ([tr[f"{key}_raw"]] if FILTER_ON else [])
        lo, hi = min(float(np.min(y)) for y in ys), max(float(np.max(y)) for y in ys)
        pad = 0.15 * (hi - lo) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set(xlabel=r"$\tau$", ylabel=label, title=f"training: {label[1:-1]} error {e_tr[row]:.2%}")
        ax.grid(True)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{tag} (no test set)")
    return finish(fig, f"AOA{tag}_fit")
 
 
def kernel_frf(h, dt, nfft):
    """H1(f) = sum_i h[i] exp(-j 2 pi f i dt): frequency response of a first-order kernel
    (h[i] multiplies u(t - i dt), as in build_A123)."""
    return np.fft.rfftfreq(nfft, dt), np.fft.rfft(h, n=nfft)
 
 
def empirical_frf(ds, w, dt):
    """Y(f) / U(f) at the spectral lines of each input, from the fully weighted rows (w == 1),
    i.e. without start-up ramp and skipped rows. Pitch lines are bins where the pitch spectrum has a
    local maximum above FRF_LINE_THR of its largest line and dominates the zdot spectrum by
    FRF_LINE_RATIO (and vice versa), so the two inputs need different line sets. Returns
    {(input, output): (f_lines, H_lines)} in the conditioned (scaled) units of `ds`."""
    idx = np.flatnonzero(w >= 1.0)
    if idx.size < 16:
        return {}
    seg = slice(idx[0], idx[-1] + 1)
    n = idx[-1] + 1 - idx[0]
    win = np.hanning(n) if FRF_WINDOW == "hann" else np.ones(n)
 
    def spec(x):
        x = np.asarray(x[seg], dtype=float)
        return np.fft.rfft((x - x.mean()) * win)
 
    f = np.fft.rfftfreq(n, dt)
    S = {"p": spec(ds["pitch"]), "z": spec(ds["zdot"])}
    Y = {"cl": spec(ds["cl"]), "cm": spec(ds["cm"])}
    out = {}
    for a, b in (("p", "z"), ("z", "p")):
        amp, other = np.abs(S[a]), np.abs(S[b])
        pk = np.zeros(len(f), dtype=bool)
        pk[1:-1] = (amp[1:-1] >= amp[:-2]) & (amp[1:-1] >= amp[2:])
        sel = pk & (amp > FRF_LINE_THR * amp.max()) & (amp > FRF_LINE_RATIO * other)
        for o in ("cl", "cm"):
            out[(a, o)] = (f[sel], Y[o][sel] / S[a][sel])
    return out
 
 
def plot_frf(tag, H, sets):
    """Real and imaginary part of the first-order transfer functions: identified kernel (line)
    against Y/U at the excited lines of each record (markers). `sets` = [(label, marker, ds, w)].
    Both are converted to physical units (coefficient per rad). For a record with a large amplitude
    Y/U is an amplitude-dependent best linear approximation, so its distance from the kernel line
    shows how much nonlinearity the first-order kernel has to carry."""
    dt = DELTA_T * RESAMPLE
    fmax = FRF_FMAX if FRF_FMAX is not None else (CUTOFF if FILTER_ON else 0.5 / dt)
    nfft = 16 * M
    outs = (("cl", 0, CL_NORM_FACTOR, "C_L"), ("cm", 1, CM_NORM_FACTOR, "C_M"))
    ins = (("p", 0, r"\theta"), ("z", M, r"\dot h / V"))
 
    emp = []
    for label, marker, ds, w in sets:
        e = empirical_frf(ds, w, dt)
        print(f"FRF lines ({label}): pitch {len(e.get(('p', 'cl'), ([], []))[0])}, "
              f"zdot {len(e.get(('z', 'cl'), ([], []))[0])}")
        emp.append((label, marker, e))
 
    fig, axs = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    row = 0
    for yk, yj, ynorm, yn in outs:
        for xk, off, xs in ins:
            scale = NORM_FACTOR * INPUT_GAIN / ynorm        # scaled -> physical, as in the saved kernels
            f, Hf = kernel_frf(H[off:off + M, yj], dt, nfft)
            keep = f <= fmax
            for col, part, pname in ((0, np.real, "Re"), (1, np.imag, "Im")):
                ax = axs[row, col]
                ax.plot(f[keep], part(Hf[keep]) * scale, "b-", lw=1.2, label=r"$H_1$ (kernel)")
                for label, marker, e in emp:
                    fl, Hl = e.get((xk, yk), (np.array([]), np.array([], dtype=complex)))
                    k2 = fl <= fmax
                    ax.plot(fl[k2], part(Hl[k2]) * scale, marker, ms=5, ls="none", label=f"Y/U, {label}")
                ax.set_ylabel(rf"{pname} ${yn}\,/\,{xs}$ [1/rad]")
                ax.grid(True)
            row += 1
    for ax in axs[-1]:
        ax.set_xlabel("f [Hz]")
    axs[0, 0].legend(fontsize=8)
    for ax in axs[0]:
        top = ax.secondary_xaxis("top", functions=(lambda f: np.pi * f * CHORD / V_INF,
                                                   lambda k: k * V_INF / (np.pi * CHORD)))
        top.set_xlabel(r"reduced frequency $k = \pi f c / V$")
    fig.suptitle(f"First-order transfer functions, {tag}")
    fig.tight_layout()
    return finish(fig, f"AOA{tag}_frf")
 
 
def identify(A, Y, w, blocks, reg, holdout=None, bounds=None):
    """Single entry point for the fit of all blocks in `blocks` (columns of A).
    Returns (H, jitter, info); info holds the order multipliers and the selected blocks (gram)."""
    if FIT_METHOD == "classic":
        H, jitter = fit_blocks(A, Y, blocks, M, w, **reg)
        return H, jitter, {}
    assert FIT_METHOD == "gram", FIT_METHOD
    import ident_extras as ie
    G, r, yy = ie.gram_stats(A, Y, w)
    mult, info = None, {}
    if TUNE_LAMBDAS:
        stats = ie.segment_stats(A, Y, w, bounds if bounds is not None else ie.cv_bounds(w, CV_FOLDS))
        make = lambda d, om: ie.lag_groups(blocks, M, d, reg, om)
        mult, cv = ie.tune_order_lambdas(stats, make, orders=sorted({b[1] for b in blocks}))
        print(f"penalty multipliers per order: {mult}  (CV error {cv:.2%})")
        info["mult"] = mult
        del stats
    groups = ie.lag_groups(blocks, M, np.diag(G), reg, mult)
    if GREEDY:
        h, path = ie.greedy_select(G, r, yy, groups, forced=GREEDY_FORCED, max_groups=GREEDY_MAX,
                                   tol=GREEDY_TOL, holdout=holdout)
        info["selected"] = [e["name"] for e in path[1:]]
        return h, 0.0, info
    h, jitter = ie.solve_groups(G, r, groups)
    return h, jitter, info
 
 
# ---------------------------------------------------------------- run_case
def run_case(aoa, model):
    assert SMALL_MODE in ("joint", "sequential"), SMALL_MODE
    tag = f"{aoa}_{model}"
    files = case_files(aoa, model)
    missing = [files[k] for k in ("train", "steady") if not os.path.isfile(files[k])]
    if missing:
        print(f"AOA {aoa}, model {model}: skipped, not found: " + ", ".join(missing))
        return None
    os.makedirs(files["home"], exist_ok=True)
 
    # optional datasets: used only if requested AND present
    use_small = USE_SMALL and os.path.isfile(files["train_small"])
    if USE_SMALL and not use_small:
        if SMALL_REQUIRED:
            print(f"AOA {aoa}, model {model}: skipped, {files['train_small']} not found (SMALL_REQUIRED)")
            return None
        print(f"AOA {aoa}, model {model}: {files['train_small']} not found -> single fit on 'train'")
    use_test = USE_TEST and os.path.isfile(files["test"])
    if USE_TEST and not use_test:
        print(f"AOA {aoa}, model {model}: {files['test']} not found -> no test comparison")
    joint = use_small and SMALL_MODE == "joint"
 
    tr_raw = load_dataset(aoa, model, "train", files)
    te_raw = load_dataset(aoa, model, "test", files) if use_test else None
    sm_raw = load_dataset(aoa, model, "train_small", files) if use_small else None
    n = len(tr_raw["t"])
    n_te = len(te_raw["t"]) if use_test else 0
    n_sm = len(sm_raw["t"]) if use_small else 0
 
    st = read_cols(files["steady"], keys=("CL0", "CM0"))
    cl_0, cm_0 = float(st["CL0"][0]), float(st["CM0"][0])
 
    # same static reference for every set (valid only if all runs start from the same steady state)
    means = {"pitch": 0.0, "zdot": 0.0, "cl": cl_0, "cm": cm_0}
 
    tr = condition(tr_raw, means)
    te = condition(te_raw, means) if use_test else None
    sm = condition(sm_raw, means) if use_small else None
 
    for sig in (tr, te, sm):
        if sig is None:
            continue
        sig["pitch"] = sig["pitch"] * INPUT_GAIN
        sig["zdot"] = sig["zdot"] * INPUT_GAIN
 
    print(f"\n=== AOA {aoa}, model {model}: {n} training"
          + (f", {n_te} test" if use_test else ", no test")
          + (f", {n_sm} small-training ({SMALL_MODE})" if use_small else "") + f" samples, m = {M}")
 
    w = make_weights(n)
    n_used = int(np.count_nonzero(w))
    if SKIP + RAMP + SKIP_END >= n or n_used <= M:
        print(f"AOA {aoa} {model}: only {n_used} samples left after skipping -> skipped")
        return None
 
    if use_small:
        w_sm = make_weights(n_sm, SMALL_SKIP, SMALL_RAMP, SMALL_SKIP_END)
        n_sm_used = int(np.count_nonzero(w_sm))
        if n_sm_used <= 0:
            print(f"AOA {aoa} {model}: no samples left in the small set -> skipped")
            return None
        if not joint and n_sm_used < 2 * M:
            print(f"WARNING: small set: {n_sm_used} samples used < {2 * M} first-order coefficients; "
                  "H1 is regularization-dominated")
 
    # ---- layout and truncation to ORDER (unchanged)
    blocks = block_layout3(M, BAND, BAND3)
    sl = block_slices(blocks)
    n_coef_full = sum(length for _, _, length, _ in blocks)
    col_end = {1: 2 * M, 2: sl["p3"].start, 3: sl["p4"].start, 4: n_coef_full}[ORDER]
 
    kept, cum = [], 0
    for b in blocks:
        if cum + b[2] <= col_end:
            kept.append(b)
            cum += b[2]
    assert cum == col_end, (cum, col_end)
    blocks, n_coef = kept, col_end
 
    n_rows = n_used + (n_sm_used if joint else 0)
    if n_rows < n_coef:
        print(f"WARNING: {n_rows} samples used < {n_coef} coefficients: underdetermined, the regularization decides")
 
    # ---- design matrix. In joint mode the small set occupies the first n_sm rows of A_all and the
    # large set the rest; A is a view of the large-set rows. Lags are built per record, never across.
    Ytr = np.column_stack((tr["cl"], tr["cm"]))
    n_top = n_sm if joint else 0
    A_all = np.empty((n_top + n, col_end))
    A_all[n_top:] = build_A_band3(tr["pitch"], tr["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
    if joint:
        A_all[:n_top] = build_A_band3(sm["pitch"], sm["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
    A = A_all[n_top:]
    assert A.shape[1] == n_coef, (A.shape[1], n_coef)
 
    reg = dict(lam_ridge=LAMBDA_RIDGE, lam_smooth=LAMBDA_SMOOTH,
               lam_decay=LAMBDA_DECAY, decay_rate=DECAY_RATE,
               lam_band=LAMBDA_BAND, band_decay=BAND_DECAY, b=max(BAND, BAND3))
 
    # ---- identification
    e_sm, s1, fit_sm = None, float("nan"), None
    info = {}
    holdout = None
    if FIT_METHOD == "gram" and GREEDY and use_test:
        import ident_extras as ie
        A_te0 = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
        holdout = ie.gram_stats(A_te0, np.column_stack((te["cl"], te["cm"])), make_weights(n_te))
        del A_te0
    if not use_small:
        H, jitter, info = identify(A, Ytr, w, blocks, reg, holdout=holdout)
 
    elif joint:
        # One weighted least-squares problem over both records with shared coefficients.
        # The squared residuals of the large set would dominate the cost, so the rows of the small
        # set are scaled by gamma such that both sets contribute the same *relative* error:
        # gamma = ||w y_large|| / ||w y_small||  (times SMALL_WEIGHT).
        Y_sm = np.column_stack((sm["cl"], sm["cm"]))
        gamma = SMALL_WEIGHT * np.linalg.norm(w[:, None] * Ytr) / np.linalg.norm(w_sm[:, None] * Y_sm)
        print(f"joint fit: small-set rows scaled by {gamma:.3g}")
        w_all = np.concatenate((gamma * w_sm, w))
        bounds = [(0, n_sm), (n_sm, n_sm + n)] if CV_BY_RECORD else None
        H, jitter, info = identify(A_all, np.vstack((Y_sm, Ytr)), w_all, blocks, reg,
                                   holdout=holdout, bounds=bounds)
        fit_sm = A_all[:n_sm] @ H
        e_sm = rel_err(fit_sm, Y_sm, w_sm > 0)
        print(f"small set, full model: CL {e_sm[0]:.2%}, CM {e_sm[1]:.2%}")
 
    else:
        # stage 1: H1 from the small-amplitude set
        A_sm = build_A1(sm["pitch"], sm["zdot"], M)
        Y_sm = np.column_stack((sm["cl"], sm["cm"]))
        s1 = first_order_scale(A_sm, A[:, :2 * M], w_sm, w, M)
        reg1 = dict(reg, lam_ridge=reg["lam_ridge"] * s1,
                    lam_smooth=reg["lam_smooth"] * s1, lam_decay=reg["lam_decay"] * s1)
        print(f"first-order regularization scaled by {s1:.3g} (energy ratio small/large)")
        H1, j1 = fit_blocks(A_sm, Y_sm, blocks[:2], M, w_sm, **reg1)
 
        fit_sm = A_sm @ H1
        e_sm = rel_err(fit_sm, Y_sm, w_sm > 0)
        r = (Y_sm - fit_sm)[w_sm > 0]
        print(f"small set, H1 only: CL {e_sm[0]:.2%}, CM {e_sm[1]:.2%};  "
              f"residual mean / std: CL {r[:, 0].mean() / r[:, 0].std():+.2f}, "
              f"CM {r[:, 1].mean() / r[:, 1].std():+.2f}")
        del A_sm
 
        fit_tr_lin = A[:, :2 * M] @ H1
        e_tr_lin = rel_err(fit_tr_lin, Ytr, w > 0)
        os.makedirs(OUT_DIR, exist_ok=True)
        print("figure:", plot_two_sets(
            tag, sm, tr, fit_sm, fit_tr_lin, w_sm, w, e_sm, e_tr_lin, r"$H_1$ only",
            ("small set (fitted)", "large set (not fitted, H1 only)"),
            f"Step 1: first-order kernels only, {tag}", f"AOA{tag}_fit_step1"))
        del fit_tr_lin
 
        # stage 2: orders >= 2 on the large set, H1 held fixed
        if col_end > 2 * M:
            Yres = Ytr - A[:, :2 * M] @ H1
            H2, j2 = fit_blocks(A[:, 2 * M:], Yres, blocks[2:], M, w, **reg)
            H = np.vstack((H1, H2))
            jitter = max(j1, j2)
        else:
            H, jitter = H1, j1
 
    fit_tr = A @ H
    del A, A_all
    fit_te = None
    if use_test:
        A_te = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
        fit_te = A_te @ H
        del A_te
 
    # ---- back to the full coefficient layout
    H_full = np.zeros((n_coef_full, H.shape[1]))
    H_full[:col_end] = H
    H = H_full
 
    e_tr = rel_err(fit_tr, Ytr, w > 0)
    if use_test:
        mask_te = make_weights(n_te) > 0
        e_te = rel_err(fit_te, np.column_stack((te["cl"], te["cm"])), mask_te)
        print(f"relative error  training: CL {e_tr[0]:.2%}, CM {e_tr[1]:.2%}   "
              f"test: CL {e_te[0]:.2%}, CM {e_te[1]:.2%}")
    else:
        e_te = np.full(2, np.nan)
        print(f"relative error  training: CL {e_tr[0]:.2%}, CM {e_tr[1]:.2%}   (no test set)")
 
    os.makedirs(OUT_DIR, exist_ok=True)
    if joint:
        print("figure:", plot_two_sets(
            tag, sm, tr, fit_sm, fit_tr, w_sm, w, e_sm, e_tr, "Volterra (joint)",
            ("small set", "large set"),
            f"Joint fit on both training sets, {tag}", f"AOA{tag}_fit_joint"))
    fit_plot = (plot_fit(tag, tr, te, fit_tr, fit_te, w, e_tr, e_te) if use_test
                else plot_fit_train(tag, tr, fit_tr, w, e_tr))
    for p in (plot_data(aoa, tr_raw), plot_kernels(tag, H), fit_plot):
        print("figure:", p)
    if PLOT_FRF:
        frf_sets = [("large set", "s", tr, w)]
        if use_small:
            frf_sets.append(("small set", "o", sm, w_sm))
        print("figure:", plot_frf(tag, H, frf_sets))
 
    # ---- kernels
    h_cl, h_cm = H[:, 0], H[:, 1]
    i2a, i2b = 2 * M, sl["p3"].start
    i3a, i3b = sl["p3"].start, sl["p4"].start
    i4a = sl["p4"].start
 
    suffix = "" if not use_small else ("_joint" if joint else "_small")
    filename = os.path.join(files["home"], f"kernels_1_to_4_order{ORDER}{suffix}.npz")
    np.savez(
        filename,
        m=M, cl_0=cl_0, cm_0=cm_0,
        h_cl_1=np.concatenate([h_cl[:M], np.zeros(M), h_cl[M:2 * M]]) * NORM_FACTOR / CL_NORM_FACTOR * INPUT_GAIN,
        h_cm_1=np.concatenate([h_cm[:M], np.zeros(M), h_cm[M:2 * M]]) * NORM_FACTOR / CM_NORM_FACTOR * INPUT_GAIN,
        h_cl_2=h_cl[i2a:i2b] * (NORM_FACTOR ** 2) / CL_NORM_FACTOR * INPUT_GAIN**2,
        h_cm_2=h_cm[i2a:i2b] * (NORM_FACTOR ** 2) / CM_NORM_FACTOR * INPUT_GAIN**2,
        h_cl_3=h_cl[i3a:i3b] * (NORM_FACTOR ** 3) / CL_NORM_FACTOR * INPUT_GAIN**3,
        h_cm_3=h_cm[i3a:i3b] * (NORM_FACTOR ** 3) / CM_NORM_FACTOR * INPUT_GAIN**3,
        h_cl_4=h_cl[i4a:] * (NORM_FACTOR ** 4) / CL_NORM_FACTOR * INPUT_GAIN**4,
        h_cm_4=h_cm[i4a:] * (NORM_FACTOR ** 4) / CM_NORM_FACTOR * INPUT_GAIN**4,
        band=BAND, band3=BAND3,
        pitch_mean=means["pitch"], zdot_mean=means["zdot"], cl_mean=means["cl"], cm_mean=means["cm"],
        delta_t=DELTA_T, resample_steps=RESAMPLE, v_inf=V_INF, chord=CHORD,
        norm_factor=NORM_FACTOR, cl_norm_factor=CL_NORM_FACTOR, cm_norm_factor=CM_NORM_FACTOR,
        filtered=FILTER_ON, cutoff=CUTOFF, skip=SKIP, ramp=RAMP, skip_end=SKIP_END,
        err_train=e_tr, err_test=e_te, jitter=jitter, grid="csv", model=model, aoa=aoa,
        fit_method=FIT_METHOD, selected=np.array(info.get("selected", []), dtype=str),
        use_small=use_small, small_mode=(SMALL_MODE if use_small else "none"), use_test=use_test,
        err_small=(e_sm if e_sm is not None else np.nan), h1_reg_scale=s1,
    )
    print("kernels:", filename)
 
    return dict(model=model, aoa=aoa, n_train=n, n_test=n_te, m=M, n_used=n_used, jitter=jitter,
                train_err_cl=e_tr[0], train_err_cm=e_tr[1], test_err_cl=e_te[0], test_err_cm=e_te[1],
                mean_minus_static_cl=means["cl"] - cl_0, mean_minus_static_cm=means["cm"] - cm_0)

                
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoa", type=int, nargs="+", default=NEW_AOAS)
    ap.add_argument("--model", nargs="+", choices=TURB_MODELS, default=TURB_MODELS)
    args = ap.parse_args()

    if FILTER_ON and CUTOFF >= 0.5 / (DELTA_T * RESAMPLE):
        raise SystemExit(f"CUTOFF must be below the Nyquist frequency {0.5 / (DELTA_T * RESAMPLE):g} Hz")
    dtau = DELTA_T * V_INF / (2 * CHORD)
    print(f"AOA: {args.aoa}, models: {args.model}")
    print("Model memory: ", RESAMPLE * M * dtau, " (reduced time units)")

    rows = [r for r in (run_case(a, m) for a in args.aoa for m in args.model) if r is not None]
    if rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "summary_new.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsummary: {path}")

if __name__ == "__main__":
    main()