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
from regularized_mode_solve_v2 import (
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

NEW_AOAS     = [0, 1, 2] #[0, 1, 2, 3, 4, 5, 6]
# TURB_MODELS  = ["A", "B", "C", "D"]
# TURB_MODELS  = ["B", "C", "D"]
TURB_MODELS  = ["B"]
PITCH_IN_DEG = False        # CHECK: True if PITCH is in degrees -> converted to rad
# DELTA_T, V_INF, CHORD, M (top of the file) must be set for this dataset.
# ZDOTOV is assumed to be hdot/V already (same quantity as 'zdot' before).
MACH = 0.8

# optional: use only the first N time steps of the TRAINING set of a case, e.g. {("fine", 5): 3150}
NT_MAX = {}

V_INF = 131.41                                  # [m/s]
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
LAMBDA_SMOOTH = 1e-5 #1e-4
LAMBDA_DECAY = 1e-4 #1e-5
DECAY_RATE = 6.0 #5.0
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
BAND       = 2 #6        # 0 reproduces the old model; then try 2, 4, 8
LAMBDA_BAND   = 1e-3     # ridge growing with the lag distance |d|
BAND_DECAY = 3.0 #3.0
BAND3 = 2 #8

INPUT_GAIN = 25. #18.

M = 256 #312                                         # memory length of each kernel [samples]
ORDER = 4 
SWAP_TRAIN_TEST = False      # with the other settings at the top

# add next to the other settings
USE_SMALL = True     # H1 from "train_small", higher orders on "train" (False = old single fit)
# Weights for the small set. The default copies the large set (SKIP = 0, 5000-sample ramp).
# Keep the ramp if the small signal is also a ramped multisine; set SMALL_RAMP = 0 if it is a
# step (the ramp would down-weight the start, where the step response is).
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
        train_small=str(d / "training_small.csv"),      # NEW
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
 
 
# ---------------------------------------------------------------- helper
def first_order_scale(A_small, A_big, w_small, w_big, m):
    """Ratio of the mean (weighted) energy of the first-order columns, small / large set.
    LAMBDA_RIDGE, LAMBDA_SMOOTH and LAMBDA_DECAY are absolute, so on a lower-amplitude set the
    same values would regularize H1 much more strongly. Scaling them by this ratio keeps the
    *relative* regularization of H1 the same as in the full-record fit."""
    e_s = np.sum((w_small[:, None] * A_small[:, :2 * m]) ** 2, axis=0).mean()
    e_b = np.sum((w_big[:, None] * A_big[:, :2 * m]) ** 2, axis=0).mean()
    return e_s / e_b
 
 
def build_A1(pitch, zdot, m):
    """First-order columns only: identical to the first 2M columns of build_A123
    ([pitch lags | zdot lags], zero-padded before t = 0), without the 12 other blocks."""
    n = len(pitch)
    A = np.zeros((n, 2 * m))
    for i in range(m):
        A[i:, i] = pitch[:n - i]
        A[i:, m + i] = zdot[:n - i]
    return A
 
 
def plot_step1(tag, sm, tr, fit_sm, fit_tr_lin, w_sm, w, e_sm, e_tr_lin):
    """Fit after step 1 only (H1 from the small set, nothing else).
    Left: small set (the data H1 was fitted on). Right: large set, where the linear model is
    expected to miss the nonlinear part; this is what step 2 has to explain."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    cols = ((sm, fit_sm, w_sm, e_sm, "small set (fitted)"),
            (tr, fit_tr_lin, w, e_tr_lin, "large set (not fitted, H1 only)"))
    for col, (ds, fit, wt, e, name) in enumerate(cols):
        tau = ds["t"] * 2 * V_INF / CHORD
        for row, (key, label) in enumerate((("cl", r"$C_L$"), ("cm", r"$C_M$"))):
            ax = axs[row, col]
            if FILTER_ON:
                ax.plot(tau, ds[f"{key}_raw"], color="0.75", lw=0.8, label="reference (raw)")
            ax.plot(tau, ds[key], "k-", lw=1.0, label="reference (low-pass)" if FILTER_ON else "reference")
            ax.plot(tau, fit[:, row], "b-", lw=1.0, label=r"$H_1$ only")
            if np.any(wt < 1.0):
                full = np.flatnonzero(wt >= 1.0)
                if full.size:
                    ax.axvspan(tau[0], tau[full[0]], color="0.5", alpha=0.15, lw=0, label="not / partly used")
                    ax.axvspan(tau[full[-1]], tau[-1], color="0.5", alpha=0.15, lw=0)
            ys = [ds[key]] + ([ds[f"{key}_raw"]] if FILTER_ON else [])
            lo, hi = min(float(np.min(y)) for y in ys), max(float(np.max(y)) for y in ys)
            pad = 0.15 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + pad)
            ax.set(xlabel=r"$\tau$", ylabel=label, title=f"{name}: {label[1:-1]} error {e[row]:.2%}")
            ax.grid(True)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Step 1: first-order kernels only, {tag}")
    return finish(fig, f"AOA{tag}_fit_step1")
 
 
# ---------------------------------------------------------------- run_case
def run_case(aoa, model):
    tag = f"{aoa}_{model}"
    files = case_files(aoa, model)
    needed = ["train", "test", "steady"] + (["train_small"] if USE_SMALL else [])
    missing = [files[k] for k in needed if not os.path.isfile(files[k])]
    if missing:
        print(f"AOA {aoa}, model {model}: skipped, not found: " + ", ".join(missing))
        return None
    os.makedirs(files["home"], exist_ok=True)
 
    tr_raw = load_dataset(aoa, model, "train", files)
    te_raw = load_dataset(aoa, model, "test", files)
    sm_raw = load_dataset(aoa, model, "train_small", files) if USE_SMALL else None
    n, n_te = len(tr_raw["t"]), len(te_raw["t"])
    n_sm = len(sm_raw["t"]) if USE_SMALL else 0
 
    st = read_cols(files["steady"], keys=("CL0", "CM0"))
    cl_0, cm_0 = float(st["CL0"][0]), float(st["CM0"][0])
 
    # same static reference for every set (valid only if all runs start from the same steady state)
    means = {"pitch": 0.0, "zdot": 0.0, "cl": cl_0, "cm": cm_0}
 
    tr = condition(tr_raw, means)
    te = condition(te_raw, means)
    sm = condition(sm_raw, means) if USE_SMALL else None
 
    for sig in (tr, te, sm):
        if sig is None:
            continue
        sig["pitch"] = sig["pitch"] * INPUT_GAIN
        sig["zdot"] = sig["zdot"] * INPUT_GAIN
 
    print(f"\n=== AOA {aoa}, model {model}: {n} training, {n_te} test"
          + (f", {n_sm} small-training" if USE_SMALL else "") + f" samples, m = {M}")
 
    w = make_weights(n)
    n_used = int(np.count_nonzero(w))
    if SKIP + RAMP + SKIP_END >= n or n_used <= M:
        print(f"AOA {aoa} {model}: only {n_used} samples left after skipping -> skipped")
        return None
 
    if USE_SMALL:
        w_sm = make_weights(n_sm, SMALL_SKIP, SMALL_RAMP, SMALL_SKIP_END)
        n_sm_used = int(np.count_nonzero(w_sm))
        if n_sm_used < 2 * M:
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
 
    if n_used < n_coef:
        print(f"WARNING: {n_used} samples used < {n_coef} coefficients: underdetermined, the regularization decides")
 
    Ytr = np.column_stack((tr["cl"], tr["cm"]))
    A = build_A_band3(tr["pitch"], tr["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
    assert A.shape[1] == n_coef, (A.shape[1], n_coef)
 
    reg = dict(lam_ridge=LAMBDA_RIDGE, lam_smooth=LAMBDA_SMOOTH,
               lam_decay=LAMBDA_DECAY, decay_rate=DECAY_RATE,
               lam_band=LAMBDA_BAND, band_decay=BAND_DECAY, b=max(BAND, BAND3))
 
    # ---- identification
    e_sm, s1 = None, float("nan")
    if not USE_SMALL:
        H, jitter = fit_blocks(A, Ytr, blocks, M, w, **reg)
    else:
        # stage 1: H1 from the small-amplitude set (first 2M columns only: [pitch lags | zdot lags])
        A_sm = build_A1(sm["pitch"], sm["zdot"], M)
        Y_sm = np.column_stack((sm["cl"], sm["cm"]))
        s1 = first_order_scale(A_sm, A[:, :2 * M], w_sm, w, M)
        reg1 = dict(reg, lam_ridge=reg["lam_ridge"] * s1,
                    lam_smooth=reg["lam_smooth"] * s1, lam_decay=reg["lam_decay"] * s1)
        print(f"first-order regularization scaled by {s1:.3g} (energy ratio small/large)")
        H1, j1 = fit_blocks(A_sm, Y_sm, blocks[:2], M, w_sm, **reg1)
 
        fit_sm = A_sm @ H1
        e_sm = rel_err(fit_sm, Y_sm, w_sm > 0)
        # a mean offset in the residual means the small run does not start from the steady state
        # used as reference (or that second-order mean shifts are not negligible)
        r = (Y_sm - fit_sm)[w_sm > 0]
        print(f"small set, H1 only: CL {e_sm[0]:.2%}, CM {e_sm[1]:.2%};  "
              f"residual mean / std: CL {r[:, 0].mean() / r[:, 0].std():+.2f}, "
              f"CM {r[:, 1].mean() / r[:, 1].std():+.2f}")
        del A_sm
 
        # plot of the fit after step 1 only (H1 on both the small and the large set)
        fit_tr_lin = A[:, :2 * M] @ H1
        e_tr_lin = rel_err(fit_tr_lin, Ytr, w > 0)
        os.makedirs(OUT_DIR, exist_ok=True)
        print("figure:", plot_step1(tag, sm, tr, fit_sm, fit_tr_lin, w_sm, w, e_sm, e_tr_lin))
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
    del A
    A_te = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123)[:, :col_end]
    fit_te = A_te @ H
    del A_te
 
    # ---- back to the full coefficient layout
    H_full = np.zeros((n_coef_full, H.shape[1]))
    H_full[:col_end] = H
    H = H_full
 
    w_te = make_weights(n_te)
    mask_te = w_te > 0
    e_tr = rel_err(fit_tr, Ytr, w > 0)
    e_te = rel_err(fit_te, np.column_stack((te["cl"], te["cm"])), mask_te)
    print(f"relative error  training: CL {e_tr[0]:.2%}, CM {e_tr[1]:.2%}   "
          f"test: CL {e_te[0]:.2%}, CM {e_te[1]:.2%}")
 
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (plot_data(aoa, tr_raw), plot_kernels(tag, H), plot_fit(tag, tr, te, fit_tr, fit_te, w, e_tr, e_te)):
        print("figure:", p)
 
    # ---- kernels
    h_cl, h_cm = H[:, 0], H[:, 1]
    i2a, i2b = 2 * M, sl["p3"].start
    i3a, i3b = sl["p3"].start, sl["p4"].start
    i4a = sl["p4"].start
 
    suffix = "_small" if USE_SMALL else ""
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
        use_small=USE_SMALL, err_small=(e_sm if e_sm is not None else np.nan), h1_reg_scale=s1,
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