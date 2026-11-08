import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
import matplotlib.pyplot as plt


"""
Second-order Volterra terms with a BANDED memory.

Your current second-order terms are products of two inputs at the SAME lag i:
    y2(t) = sum_i  h_pp(i) p(t-i)^2 + h_zz(i) z(t-i)^2 + h_pz(i) p(t-i) z(t-i)        (band b = 0)
With band b they also contain products of two DIFFERENT lags, |lag difference| <= b:
    y2(t) = sum_i sum_d  c_pp(i,d) p(t-i) p(t-i-d)        d = 0 ... b     (symmetric: each pair once)
                       + c_zz(i,d) z(t-i) z(t-i-d)        d = 0 ... b
                       + c_pz(i,d) p(t-i) z(t-i-d)        d = -b ... b
Every "diagonal" d is one block of coefficients of length m-|d|. First, third and fourth order
are exactly those of build_A123 (so b = 0 reproduces the old model).

Column layout (b = 0 is identical to build_A123):
    p, z | pp_d0..pp_db | zz_d0..zz_db | pz_d-b..pz_db | p3, z3, pz2, p2z | p4, z4, p2z2, p3z, pz3
Memory warning: the design matrix has n * n_coeffs entries; for m = 240 and b = 10 about 12,700 columns.
"""

from scipy.linalg.blas import dsyrk
from numpy.linalg import LinAlgError

ORDER3 = ("p3", "z3", "pz2", "p2z")
ORDER4 = ("p4", "z4", "p2z2", "p3z", "pz3")


def block_layout(m, b):
    """List of blocks (name, order, length, dist) in column order; dist = |d|."""
    blocks = [("p", 1, m, 0), ("z", 1, m, 0)]
    blocks += [(f"pp_d{d}", 2, m - d, d) for d in range(0, b + 1)]
    blocks += [(f"zz_d{d}", 2, m - d, d) for d in range(0, b + 1)]
    blocks += [(f"pz_d{d}", 2, m - abs(d), abs(d)) for d in range(-b, b + 1)]
    blocks += [(name, 3, m, 0) for name in ORDER3]
    blocks += [(name, 4, m, 0) for name in ORDER4]
    return blocks


def block_slices(blocks):
    """name -> slice of the coefficient vector."""
    out, start = {}, 0
    for name, _, length, _ in blocks:
        out[name] = slice(start, start + length)
        start += length
    return out


def band_products(U, V, b, symmetric):
    """Columns u(t-i) v(t-i-d) for the diagonals d = 0..b (symmetric) or -b..b.
    U, V are lag matrices (n, m) with U[t, i] = u(t-i). Returns one array (n, m-|d|) per diagonal."""
    m = U.shape[1]
    ds = range(0, b + 1) if symmetric else range(-b, b + 1)
    return [U[:, max(0, -d):m - max(0, d)] * V[:, max(0, d):m - max(0, -d)] for d in ds]


def build_A_band(pitch, zdot, m, b, build_A123):
    """Design matrix with the banded second-order part. Uses the lag matrices that build_A123
    already contains (its first-order block is [pitch lags | heave lags])."""
    A0 = build_A123(pitch, zdot, m)                    # [p z | p2 z2 pz | p3 z3 pz2 p2z | p4 z4 p2z2 p3z pz3]
    if b == 0:
        return A0
    assert A0.shape[1] == 14 * m, "unexpected layout of build_A123"
    U, V = A0[:, :m], A0[:, m:2 * m]
    blocks = block_layout(m, b)
    A = np.empty((A0.shape[0], sum(length for _, _, length, _ in blocks)))
    A[:, :2 * m] = A0[:, :2 * m]
    col = 2 * m
    for group in (band_products(U, U, b, True), band_products(V, V, b, True), band_products(U, V, b, False)):
        for blk in group:
            A[:, col:col + blk.shape[1]] = blk
            col += blk.shape[1]
    assert col + 9 * m == A.shape[1]
    A[:, col:] = A0[:, 5 * m:]                         # third and fourth order, unchanged
    return A

def third_band_spec(b3):
    spec = []
    for d in range(1, b3 + 1):
        for a, c in (("p", "p"), ("p", "z"), ("z", "z")):
            for s in ("p", "z"):
                spec.append((f"n_{a}{c}_{s}_d{d}", d, "n", a, c, s))  # (a c)(i) * s(i+d)
                spec.append((f"o_{s}_{a}{c}_d{d}", d, "o", a, c, s))  # s(i) * (a c)(i+d)
    return spec

def third_band_products(X, spec, m):
    """X = {'p': ..., 'z': ...}. Works on lag matrices (n, m) AND on lag vectors (m,)."""
    out = []
    for name, d, kind, a, c, s in spec:
        if kind == "n":
            out.append(X[a][..., :m-d] * X[c][..., :m-d] * X[s][..., d:])
        else:
            out.append(X[s][..., :m-d] * X[a][..., d:] * X[c][..., d:])
    return np.concatenate(out, axis=-1)

def block_layout3(m, b, b3):
    base = block_layout(m, b)
    k = [bl[0] for bl in base].index("p4")                 # insert before the 4th order
    extra = [(name, 3, m - d, d) for name, d, *_ in third_band_spec(b3)]
    return base[:k] + extra + base[k:]

def build_A_band3(pitch, zdot, m, b, b3, build_A123):
    A = build_A_band(pitch, zdot, m, b, build_A123)
    if b3 == 0:
        return A
    X = {"p": A[:, :m], "z": A[:, m:2*m]}                  # lag matrices
    extra = third_band_products(X, third_band_spec(b3), m)
    return np.hstack((A[:, :-5*m], extra, A[:, -5*m:]))    # before the 5 fourth-order blocks

def fit_blocks(A, Y, blocks, m, w, lam_ridge, lam_smooth, lam_decay, decay_rate,
               lam_band, band_decay, b, jitters=(0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8),
               chunk=2048):
    """Weighted, regularized least squares for the block layout. Per block (as before):
    smoothness (second difference along the lag), decay with the OLDER lag of the pair (i + |d|);
    in addition for off-diagonal blocks a ridge lam_band * column energy * (exp(band_decay |d|/b) - 1)
    that grows with the distance |d| between the two lags. For b = 0 this is exactly the old problem.
    Memory-light: Gram matrix accumulated with rank-k updates (upper triangle), penalties added
    blockwise, scaling and the ridge fallback in place."""
    n, nc = A.shape
    M = np.zeros((nc, nc), order="F")
    for r0 in range(0, n, chunk):
        Ac = A[r0:r0 + chunk]
        if not np.all(w[r0:r0 + chunk] == 1.0):
            Ac = Ac * w[r0:r0 + chunk, None]
        M = dsyrk(1.0, Ac, beta=1.0, c=M, trans=1, lower=0, overwrite_c=1)
    energy = np.diag(M).copy()                         # column energies of the (weighted) design matrix

    di = np.diag_indices(nc)
    M[di] += lam_ridge
    start = 0
    for name, order, length, dist in blocks:
        sl = slice(start, start + length)
        if length >= 3 and lam_smooth > 0:
            D2 = np.diff(np.eye(length), n=2, axis=0)
            M[sl, sl] += lam_smooth * (D2.T @ D2)
        add = lam_decay * (np.exp(decay_rate * (np.arange(length) + dist) / (m - 1)) - 1.0)
        if dist > 0 and b > 0:
            add = add + lam_band * energy[sl] * (np.exp(band_decay * dist / b) - 1.0)
        idx = (di[0][sl], di[1][sl])
        M[idx] += add
        start += length

    s = 1.0 / np.sqrt(np.maximum(np.diag(M), np.finfo(float).tiny))
    M *= s[:, None]
    M *= s[None, :]
    applied, factor, used = 0.0, None, None
    for delta in jitters:
        M[di] += delta - applied
        applied = delta
        try:
            factor, used = cho_factor(M, lower=False), delta
            break
        except LinAlgError:
            continue
    if factor is None:
        raise LinAlgError("normal equations not positive definite even with relative ridge "
                          f"{jitters[-1]:g}: reduce m or the band, increase the ridge or use more samples")
    if used > 0:
        print(f"    note: normal equations numerically singular; added relative ridge {used:g}")
    rhs = A.T @ ((w ** 2)[:, None] * Y)
    return s[:, None] * cho_solve(factor, s[:, None] * rhs), used

def window_bounds(w, n_win=4, overlap=0.5):
    used = np.flatnonzero(w > 0)
    s0, N = used[0], len(used)
    L = int(N / (n_win * (1 - overlap) + overlap))       # window length
    step = int(L * (1 - overlap))
    return [(s0 + k * step, s0 + k * step + L) for k in range(n_win)]

def fit_windows(A, Y, blocks, M, w, bounds, **reg):
    Hs = []
    for s, e in bounds:
        H, _ = fit_blocks(A[s:e], Y[s:e], blocks, M, w[s:e], **reg)
        Hs.append(H)
    return Hs
    
def cross_errors(A, Y, Hs, bounds):
    E = np.zeros((len(Hs), len(bounds)))
    E_cl = np.zeros((len(Hs), len(bounds)))
    E_cm = np.zeros((len(Hs), len(bounds)))
    for k, H in enumerate(Hs):
        for j, (s, e) in enumerate(bounds):
            r = A[s:e] @ H - Y[s:e]
            # E[k, j] = np.linalg.norm(r) / np.linalg.norm(Y[s:e])
            E_cl[k, j] = np.linalg.norm(r[:, 0]) / np.linalg.norm(Y[s:e, 0])
            E_cm[k, j] = np.linalg.norm(r[:, 1]) / np.linalg.norm(Y[s:e, 1])
    return E_cl, E_cm                         # row = kernels, column = window evaluated on

def window_predictions(A, Hs, bounds):
    """P[j][k] = prediction of kernels k on window j, shape (e_j - s_j, 2)."""
    return [[A[s:e] @ H for H in Hs] for (s, e) in bounds]

def plot_window_predictions(t, Y, P, bounds, fname=None):
    n_win = len(bounds)
    fig, ax = plt.subplots(n_win, 2, figsize=(13, 2.6 * n_win), sharex='col', squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_win))
    for j, (s, e) in enumerate(bounds):
        for c, name in enumerate(("CL", "CM")):
            a = ax[j, c]
            a.plot(t[s:e], Y[s:e, c], 'k', lw=1.6, label="measured")
            for k in range(n_win):
                own = (k == j)
                a.plot(t[s:e], P[j][k][:, c], color=colors[k],
                       lw=1.4 if own else 0.8, ls='-' if own else '--',
                       label=f"kernels {k}" + (" (own)" if own else ""))
            a.set_ylabel(f"{name}, window {j}")
            a.grid(True)
    ax[0, 0].legend(ncol=3, fontsize=7)
    ax[-1, 0].set_xlabel("t [s]"); ax[-1, 1].set_xlabel("t [s]")
    plt.tight_layout()
    if fname: fig.savefig(fname, dpi=130)
    plt.close(fig)

def noisy_signals(tr, rel, rng):
    s = {k: v.copy() for k, v in tr.items()}
    for k in ("pitch", "zdot"):
        s[k] += rel * np.std(tr[k]) * rng.standard_normal(len(tr[k]))
    return s
    
from scipy.linalg import qr, lstsq

def solve_full_regularized(A, ys, m,
                              lambda_ridge=1e-8, lambda_smooth=1e-8,
                              lambda_decay=1e-5, decay_rate=5.0,
                              qr_cache=None, sqrtw=None):
    """Same problem as solve_full_regularized, without the normal equations.
    sqrtw: optional sqrt of the sample weights (rows of A and ys are scaled by it)."""
    if sqrtw is not None:
        A = A * sqrtw[:, None]
        ys = ys * (sqrtw[:, None] if ys.ndim == 2 else sqrtw)

    n = A.shape[1]
    D2, L = build_regularization_matrices(n, m, decay_rate=decay_rate)
    D2 = D2.toarray() if hasattr(D2, "toarray") else np.asarray(D2)
    L  = L.toarray()  if hasattr(L,  "toarray") else np.asarray(L)

    # square root of L (symmetric positive semidefinite)
    ev, V = np.linalg.eigh(L)
    Lh = (V * np.sqrt(np.clip(ev, 0.0, None))).T          # Lh.T @ Lh = L

    if qr_cache is None:                                   # reuse across lambdas
        qr_cache = qr(A, mode="economic")
    Q, R = qr_cache
    Aug = np.vstack((R,
                     np.sqrt(lambda_ridge) * np.eye(n),
                     np.sqrt(lambda_smooth) * D2,
                     np.sqrt(lambda_decay) * Lh))
    rhs = np.vstack((Q.T @ ys.reshape(len(ys), -1),
                     np.zeros((n + D2.shape[0] + Lh.shape[0], ys.reshape(len(ys), -1).shape[1]))))
    h = lstsq(Aug, rhs, lapack_driver="gelsd")[0]          # SVD-based, tolerates rank deficiency
    return h if ys.ndim == 2 else h[:, 0]


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


def build_A123(pitch_input, heave_input_grad, m):
    """
    Build the stacked Volterra design matrix A_123 (linear through 4th-order
    terms, including delayed cross-terms) from the two input signals.
 
    Parameters
    ----------
    pitch_input : ndarray, shape (ntsteps,)
    heave_input_grad : ndarray, shape (ntsteps,)
    m : int
        Memory depth (number of lag columns per basic Toeplitz block).
 
    Returns
    -------
    A_123 : ndarray, shape (ntsteps, n_coeffs)
        Horizontally stacked design matrix: A (1st order) + A2 (2nd order,
        including delayed cross-terms) + A3 (3rd order) + A4 (4th order).
    """
    ntsteps = len(pitch_input)
 
    # ------------------------------------------------------------------
    # Delayed signals (m//4 steps behind)
    # ------------------------------------------------------------------
    half = m // 4
 
    pitch_delayed = np.zeros(ntsteps)
    pitch_delayed[: ntsteps - half] = pitch_input[half:]
 
    heave_delayed = np.zeros(ntsteps)
    heave_delayed[: ntsteps - half] = heave_input_grad[half:]
 
    # ------------------------------------------------------------------
    # First-order (Toeplitz lag) matrices
    # ------------------------------------------------------------------
    A_pitch = np.zeros((ntsteps, m))
    A_zdot = np.zeros((ntsteps, m))
    A_pitch_delayed = np.zeros((ntsteps, m))
    A_zdot_delayed = np.zeros((ntsteps, m))
 
    for icolumn in range(m):
        A_pitch[icolumn:, icolumn] = pitch_input[: ntsteps - icolumn]
        A_zdot[icolumn:, icolumn] = heave_input_grad[: ntsteps - icolumn]
        A_pitch_delayed[icolumn:, icolumn] = pitch_delayed[: ntsteps - icolumn]
        A_zdot_delayed[icolumn:, icolumn] = heave_delayed[: ntsteps - icolumn]
 
    A = np.hstack((A_pitch, A_zdot))
 
    # ------------------------------------------------------------------
    # Second order (including delayed cross-terms)
    # ------------------------------------------------------------------
    # A_pitch2 = np.abs(A_pitch) * A_pitch
    # A_zdot2 = np.abs(A_zdot) * A_zdot
    A_pitch2 = A_pitch * A_pitch
    A_zdot2 = A_zdot * A_zdot
    
    A_pitch_zdot = A_pitch * A_zdot
 
    A2 = np.hstack((
        A_pitch2, A_zdot2, A_pitch_zdot,
        #A_pitch * A_pitch_delayed, A_zdot * A_zdot_delayed, 
    ))
 
    # ------------------------------------------------------------------
    # Third order
    # ------------------------------------------------------------------
    A_pitch3 = A_pitch * A_pitch * A_pitch
    A_zdot3 = A_zdot * A_zdot * A_zdot
    A_pitch_zdot2 = A_pitch * A_zdot2
    A_zdot_pitch2 = A_zdot * A_pitch2
 
    A3 = np.hstack((A_pitch3, A_zdot3, A_pitch_zdot2, A_zdot_pitch2))
 
    # ------------------------------------------------------------------
    # Fourth order
    # ------------------------------------------------------------------
    A_pitch4 = A_pitch * A_pitch * A_pitch * A_pitch
    A_zdot4 = A_zdot * A_zdot * A_zdot * A_zdot
    A_pitch2_zdot2 = A_pitch * A_pitch * A_zdot * A_zdot
    A_pitch_zdot3 = A_pitch * A_zdot3
    A_pitch3_zdot = A_zdot * A_pitch3
 
    A4 = np.hstack((A_pitch4, A_zdot4, A_pitch2_zdot2, A_pitch3_zdot, A_pitch_zdot3))
 
    # ------------------------------------------------------------------
    # Stack everything
    # ------------------------------------------------------------------
    A_123 = np.hstack((A, A2, A3, A4))
    return A_123

