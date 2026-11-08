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

NEW_AOAS     = [0, 1, 2, 3, 4, 5, 6]
TURB_MODELS  = ["A", "C"]
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
M = 264                                         # memory length of each kernel [samples]

NORM_FACTOR = 1.0                               # input normalization (train and test)
CL_NORM_FACTOR = 1.0
CM_NORM_FACTOR = 1.0
DETREND_TEST = "own"                            # "own": subtract the means of the test set (as before)
                                                # "train": subtract the means of the training set

# --- regularization
LAMBDA_RIDGE = 1e-8 #1e-8
LAMBDA_SMOOTH = 1e-8 #1e-4
LAMBDA_DECAY = 1e-5 #1e-5
DECAY_RATE = 5.0 #5.0
JITTERS = (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)   # relative ridge tried if Cholesky fails

# --- optional low-pass filter of CL and CM (training and test), e.g. shock buffet
FILTER_ON = True
CUTOFF = 120.0                                  # [Hz]
FILTER_ORDER = 4

# --- optional exclusion of the start-up transient of the training set (equations removed, inputs kept)
SKIP = 120                                        # first SKIP samples
RAMP = 0                                        # half-Hann ramp 0 -> 1 over the next RAMP samples
SKIP_END = 20                                    # last SKIP_END samples
# ==========================================================================
BAND       = 0        # 0 reproduces the old model; then try 2, 4, 8
LAMBDA_BAND   = 1e-3     # ridge growing with the lag distance |d|
BAND_DECAY = 3.0
BAND3 = 4

INPUT_GAIN = 25.

def read_cols(path, keys):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}

def case_files(aoa, model):
    d = NEW_DATA_DIR / f"AOA_{aoa}" / model
    return dict(
        home=os.path.join(OUT_DIR, f"AOA_{aoa}", model),     # kernels are saved here
        train=str(d / "training.csv"),
        test=str(d / "test.csv"),
        steady=str(d / "steady.csv"),
    )


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


# ------------------------------------------------------------------ identification
def make_weights(n):
    """Row weights: 0 for the first SKIP / last SKIP_END samples, half-Hann ramp over RAMP samples
    after the skipped part, 1 elsewhere. The inputs are never cut, only equations are removed."""
    w = np.ones(n)
    w[:SKIP] = 0.0
    if RAMP > 0:
        x = np.arange(1, RAMP + 1) / (RAMP + 1)
        w[SKIP:SKIP + RAMP] = 0.5 * (1.0 - np.cos(np.pi * x))
    if SKIP_END > 0:
        w[n - SKIP_END:] = 0.0
    return w


def fit_volterra(A, Y, m, w):
    """Weighted, regularized least squares  min sum_k w_k^2 |A h - Y|_k^2 + regularization.
    Normal equations scaled to unit diagonal (same solution, better conditioned); if Cholesky still
    fails, a small relative ridge delta*diag(M) is added. All columns of Y share one factorization."""
    n_coeffs = A.shape[1]
    D2, L = build_regularization_matrices(n_coeffs, m, decay_rate=DECAY_RATE)
    Aw = A if np.all(w == 1.0) else A * w[:, None]
    Mmat = (Aw.T @ Aw + LAMBDA_RIDGE * np.eye(n_coeffs)
            + LAMBDA_SMOOTH * (D2.T @ D2) + LAMBDA_DECAY * L)
    del Aw
    s = 1.0 / np.sqrt(np.maximum(np.diag(Mmat), np.finfo(float).tiny))
    Ms = Mmat * s[:, None] * s[None, :]
    eye = np.eye(n_coeffs)
    factor, jitter = None, None
    for delta in JITTERS:
        try:
            factor, jitter = cho_factor(Ms + delta * eye), delta
            break
        except LinAlgError:
            continue
    if factor is None:
        raise LinAlgError("normal equations not positive definite even with relative ridge "
                          f"{JITTERS[-1]:g}: reduce M, increase LAMBDA_RIDGE or use more samples")
    if jitter > 0:
        print(f"    note: normal equations numerically singular; added relative ridge {jitter:g}")
    rhs = A.T @ ((w ** 2)[:, None] * Y)
    return s[:, None] * cho_solve(factor, s[:, None] * rhs), jitter


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


# ------------------------------------------------------------------ one AOA
def run_case(aoa, model):
    tag = f"{aoa}_{model}"                       # used wherever 'aoa' was used for names
    files = case_files(aoa, model)
    missing = [files[k] for k in ("train", "test", "steady") if not os.path.isfile(files[k])]
    if missing:
        print(f"AOA {aoa}, model {model}: skipped, not found: " + ", ".join(missing))
        return None
    os.makedirs(files["home"], exist_ok=True)

    tr_raw = load_dataset(aoa, model, "train", files)
    te_raw = load_dataset(aoa, model, "test", files)
    n, n_te = len(tr_raw["t"]), len(te_raw["t"])

    # sd = max(np.std(tr_raw["pitch"]), np.std(tr_raw["zdot"]))
    # print("SD :", sd)

    st = read_cols(files["steady"], keys=("CL0", "CM0"))
    cl_0, cm_0 = float(st["CL0"][0]), float(st["CM0"][0])

    means = {k: float(np.mean(tr_raw[k])) for k in ("pitch", "zdot", "cl", "cm")}
    tr = condition(tr_raw, means)
    te = condition(te_raw, means if DETREND_TEST == "train" else None)

    for sig in (tr, te):
        sig["pitch"] = sig["pitch"] * INPUT_GAIN
        sig["zdot"]  = sig["zdot"]  * INPUT_GAIN

    print(f"\n=== AOA {aoa}, model {model}: {n} training and {n_te} test samples, m = {M}")
    print(f"static Cl_0, Cm_0 = {cl_0:.5f}, {cm_0:.5f};  mean of the training signals Cl, Cm = "
          f"{means['cl']:.5f}, {means['cm']:.5f}  (difference {means['cl'] - cl_0:+.5f}, {means['cm'] - cm_0:+.5f})")

    w = make_weights(n)
    n_used = int(np.count_nonzero(w))
    if SKIP + RAMP + SKIP_END >= n or n_used <= M:
        print(f"AOA {aoa} {model}: only {n_used} samples left after skipping -> skipped")
        return None
    if n_used < 14 * M:
        print(f"WARNING: {n_used} samples used < {14 * M} coefficients: underdetermined, the regularization decides")

    # NEW
    blocks = block_layout3(M, BAND, BAND3)
    sl     = block_slices(blocks)
    n_coef = sum(length for _, _, length, _ in blocks)

    if n_used < n_coef:
        print(f"WARNING: {n_used} samples used < {n_coef} coefficients: underdetermined, the regularization decides")

    Ytr = np.column_stack((tr["cl"], tr["cm"]))
    A = build_A_band3(tr["pitch"], tr["zdot"], M, BAND, BAND3, build_A123)
    assert A.shape[1] == n_coef, (A.shape[1], n_coef)       # catches any layout mismatch early

    for name, s in sl.items():
        print(name, np.linalg.norm(A[:, s]) / np.sqrt(A.shape[0]))

    bounds = window_bounds(w, n_win=2, overlap=0.65)
    Hs = fit_windows(A, Ytr, blocks, M, w, bounds,
                    lam_ridge=LAMBDA_RIDGE, lam_smooth=LAMBDA_SMOOTH,
                    lam_decay=LAMBDA_DECAY, decay_rate=DECAY_RATE,
                    lam_band=LAMBDA_BAND, band_decay=BAND_DECAY,
                    b=max(BAND, BAND3))

    E_cl, E_cm = cross_errors(A, Ytr, Hs, bounds)

    print(np.round(E_cl, 3))
    print(np.round(E_cm, 3))

    #
    A_te = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123)
    Yte  = np.column_stack((te["cl"], te["cm"]))

    fit_te_win = [A_te @ Hk for Hk in Hs]
    for k, f in enumerate(fit_te_win):
        e = rel_err(f, Yte)
        print(f"test error, kernels of window {k}: CL {e[0]:.2%}, CM {e[1]:.2%}")

    # fit_te = A_te @ H                                  # full-record kernels (H must already exist here)
    # e_te   = rel_err(fit_te, Yte)                      # defined here, so later uses of e_te still work
    # print(f"test error, full-record kernels:   CL {e_te[0]:.2%}, CM {e_te[1]:.2%}")
    # #


    H, jitter = fit_blocks(A, Ytr, blocks, M, w,
                        lam_ridge=LAMBDA_RIDGE, lam_smooth=LAMBDA_SMOOTH,
                        lam_decay=LAMBDA_DECAY, decay_rate=DECAY_RATE,
                        lam_band=LAMBDA_BAND, band_decay=BAND_DECAY,
                        b=max(BAND, BAND3))
    fit_tr = A @ H
    A_te   = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123)
    Yte    = np.column_stack((te["cl"], te["cm"]))
    fit_te = A_te @ H

    P = window_predictions(A, Hs, bounds)
    plot_window_predictions(tr_raw["t"], Ytr, P, bounds,fname=os.path.join(OUT_DIR, f"window_pred_AOA{tag}.png"))
    del A
    fit_te = build_A_band3(te["pitch"], te["zdot"], M, BAND, BAND3, build_A123) @ H

    # ---- noise-augmented fits (inputs perturbed, targets clean)
    # REL_NOISE, N_REP = 0.05, 1
    # rng = np.random.default_rng(0)
    # H_reps = []
    # for r in range(N_REP):
    #     s  = noisy_signals(tr, REL_NOISE, rng)
    #     Ar = build_A_band3(s["pitch"], s["zdot"], M, BAND, BAND3, build_A123)
    #     Hr, _ = fit_blocks(Ar, Ytr, blocks, M, w,
    #                        lam_ridge=LAMBDA_RIDGE, lam_smooth=LAMBDA_SMOOTH,
    #                        lam_decay=LAMBDA_DECAY, decay_rate=DECAY_RATE,
    #                        lam_band=LAMBDA_BAND, band_decay=BAND_DECAY,
    #                        b=max(BAND, BAND3))
    #     H_reps.append(Hr)
    #     del Ar
    # H_bag = np.mean(H_reps, axis=0)

    # e_te_base = rel_err(A_te @ H,     Yte)
    # e_te_bag  = rel_err(A_te @ H_bag, Yte)

    # print(f"test error  baseline: CL {e_te_base[0]:.2%}, CM {e_te_base[1]:.2%}   "
    #       f"bagged (noise {REL_NOISE:.1%}): CL {e_te_bag[0]:.2%}, CM {e_te_bag[1]:.2%}")

    e_tr = rel_err(fit_tr, np.column_stack((tr["cl"], tr["cm"])), w > 0)
    e_te = rel_err(fit_te, np.column_stack((te["cl"], te["cm"])))
    print(f"relative error  training: CL {e_tr[0]:.2%}, CM {e_tr[1]:.2%}   test: CL {e_te[0]:.2%}, CM {e_te[1]:.2%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (plot_data(aoa, tr_raw), plot_kernels(tag, H), plot_fit(tag, tr, te, fit_tr, fit_te, w, e_tr, e_te)):
        print("figure:", p)

    # ---- kernels (same keys / layout as before: first order = [pitch, zeros, heave], then orders 2, 3, 4)

    h_cl, h_cm = H[:, 0], H[:, 1]
    i2a, i2b = 2 * M, sl["p3"].start              # whole second-order block (banded if BAND > 0)
    i3a, i3b = sl["p3"].start, sl["p4"].start     # p3 z3 pz2 p2z
    i4a      = sl["p4"].start                     # p4 z4 p2z2 p3z pz3

    filename = os.path.join(files["home"], "kernels_1_to_4.npz")
    np.savez(
        filename,
        m=M, cl_0=cl_0, cm_0=cm_0,
        h_cl_1=np.concatenate([h_cl[:M], np.zeros(M), h_cl[M:2 * M]]) * NORM_FACTOR / CL_NORM_FACTOR * INPUT_GAIN,
        h_cm_1=np.concatenate([h_cm[:M], np.zeros(M), h_cm[M:2 * M]]) * NORM_FACTOR / CM_NORM_FACTOR * INPUT_GAIN,
        h_cl_2=h_cl[i2a:i2b] * (NORM_FACTOR ** 2) / CL_NORM_FACTOR * INPUT_GAIN**2,
        h_cm_2=h_cm[i2a:i2b] * (NORM_FACTOR ** 2) / CM_NORM_FACTOR * INPUT_GAIN**2,
        h_cl_3=h_cl[i3a:i3b] * (NORM_FACTOR ** 3) / CL_NORM_FACTOR * INPUT_GAIN**3,
        h_cm_3=h_cm[i3a:i3b] * (NORM_FACTOR ** 3) / CM_NORM_FACTOR * INPUT_GAIN**3,
        h_cl_4=h_cl[i4a:]    * (NORM_FACTOR ** 4) / CL_NORM_FACTOR * INPUT_GAIN**4,
        h_cm_4=h_cm[i4a:]    * (NORM_FACTOR ** 4) / CM_NORM_FACTOR * INPUT_GAIN**4,
        band=BAND, band3=BAND3,                                 # new key: the integration needs it
        # extras: what is needed to apply the kernels consistently later
        pitch_mean=means["pitch"], zdot_mean=means["zdot"], cl_mean=means["cl"], cm_mean=means["cm"],
        delta_t=DELTA_T, resample_steps=RESAMPLE, v_inf=V_INF, chord=CHORD,
        norm_factor=NORM_FACTOR, cl_norm_factor=CL_NORM_FACTOR, cm_norm_factor=CM_NORM_FACTOR,
        filtered=FILTER_ON, cutoff=CUTOFF, skip=SKIP, ramp=RAMP, skip_end=SKIP_END,
        err_train=e_tr, err_test=e_te, jitter=jitter, grid="csv", model=model, aoa=aoa,
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