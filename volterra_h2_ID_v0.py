"""
Volterra 2nd-order kernel identification
=========================================
Solves:
    f_k = A_k @ h1 + A_k @ H2 @ A_k.T          for k = 1..n

where:
    A   : (n, p)   regressor matrix, p = 2*m (two inputs, memory m each)
    h1  : (p,)     first-order kernel
    H2  : (p, p)   SYMMETRIC second-order kernel
    f   : (n,)     training output

Strategy
--------
Exploit symmetry of H2: only the upper-triangular half (including
diagonal) is independent. Build a quadratic feature matrix B whose
columns are A_i * A_j (i<=j), with a factor 2 for off-diagonal terms
to account for the symmetric contribution. Then:

    f = A @ h1  +  B @ vech(H2)

which is a single linear least-squares problem in [h1 ; vech(H2)].

Two modes are provided:
  1. joint_identification()  — identify h1 and H2 simultaneously
  2. residual_identification() — assume h1 already known, identify H2
                                   on the residual f - A @ h1
"""

import numpy as np
from itertools import combinations_with_replacement


# ════════════════════════════════════════════════════════════════════════
# VECH INDEXING UTILITIES
# ════════════════════════════════════════════════════════════════════════

def vech_indices(p):
    """
    Return the list of (i, j) index pairs for the upper-triangular
    half of a (p, p) symmetric matrix, INCLUDING the diagonal,
    in the order used by vech(H2).

    Returns
    -------
    idx_pairs : list of (i, j) tuples, i <= j,  length p*(p+1)//2
    """
    return list(combinations_with_replacement(range(p), 2))


def vech_to_full(h2_vech, p):
    """
    Reconstruct the full symmetric (p, p) matrix H2 from its
    vech representation.

    Parameters
    ----------
    h2_vech : (p*(p+1)//2,) array — output of identification
    p       : int — kernel dimension

    Returns
    -------
    H2 : (p, p) symmetric matrix
    """
    idx_pairs = vech_indices(p)
    H2 = np.zeros((p, p))
    for (i, j), val in zip(idx_pairs, h2_vech):
        H2[i, j] = val
        H2[j, i] = val          # symmetric fill
    return H2


def full_to_vech(H2):
    """Inverse of vech_to_full — extract vech from a full symmetric matrix."""
    p = H2.shape[0]
    idx_pairs = vech_indices(p)
    return np.array([H2[i, j] for i, j in idx_pairs])


# ════════════════════════════════════════════════════════════════════════
# BUILD QUADRATIC FEATURE MATRIX B
# ════════════════════════════════════════════════════════════════════════

def build_quadratic_features(A):
    """
    Build the quadratic feature matrix B such that

        A_k @ H2 @ A_k.T  =  B_k @ vech(H2)

    for every row k, exploiting the symmetry of H2.

    Parameters
    ----------
    A : (n, p) regressor matrix

    Returns
    -------
    B         : (n, p*(p+1)//2) quadratic feature matrix
    idx_pairs : list of (i, j) pairs matching B's column order
                (needed to map back to H2 entries)
    """
    n, p = A.shape
    idx_pairs = vech_indices(p)
    n_feat = len(idx_pairs)

    B = np.empty((n, n_feat))
    for col, (i, j) in enumerate(idx_pairs):
        if i == j:
            B[:, col] = A[:, i] * A[:, i]
        else:
            # factor 2: off-diagonal term appears twice in A H2 A^T
            B[:, col] = 2.0 * A[:, i] * A[:, j]

    return B, idx_pairs


# ════════════════════════════════════════════════════════════════════════
# OPTION 1 — JOINT IDENTIFICATION OF h1 AND H2
# ════════════════════════════════════════════════════════════════════════

def joint_identification(A, f, reg=0.0):
    """
    Identify h1 and H2 simultaneously by solving the single
    linear least-squares problem:

        f = [A | B] @ [h1 ; vech(H2)]

    Parameters
    ----------
    A   : (n, p) regressor matrix (as you already build it)
    f   : (n,)   training output
    reg : float  optional Tikhonov regularisation strength
                (0.0 = plain least squares; >0 = ridge)
                Recommended if p*(p+1)//2 approaches n in size.

    Returns
    -------
    h1  : (p,) first-order kernel
    H2  : (p, p) second-order kernel (symmetric, full matrix)
    info: dict with r2, condition number, design matrix shape
    """
    n, p = A.shape
    f = np.asarray(f).reshape(-1)
    assert len(f) == n, "f must have the same length as A's rows"

    B, idx_pairs = build_quadratic_features(A)

    # full design matrix: [A | B]
    D = np.hstack([A, B])           # shape (n, p + p*(p+1)//2)

    print(f"Joint identification: A is ({n},{p}), "
          f"H2 has {len(idx_pairs)} unique entries")
    print(f"Full design matrix D: {D.shape}")
    if D.shape[1] > n:
        print(f"  WARNING: more unknowns ({D.shape[1]}) than data points "
              f"({n}) — system is UNDERDETERMINED. Use reg>0 or a "
              f"sparse solver (see sparse_identification below).")

    if reg > 0:
        # Ridge: (D^T D + reg*I) x = D^T f
        DtD = D.T @ D
        Dtf = D.T @ f
        x   = np.linalg.solve(DtD + reg*np.eye(D.shape[1]), Dtf)
    else:
        x, _, _, _ = np.linalg.lstsq(D, f, rcond=None)

    h1      = x[:p]
    h2_vech = x[p:]
    H2      = vech_to_full(h2_vech, p)

    f_pred = D @ x
    r2     = 1 - np.var(f - f_pred) / np.var(f)
    cond   = np.linalg.cond(D) if D.shape[0] <= 5000 else np.nan  # skip if huge

    info = dict(r2=r2, cond=cond, n_unknowns=D.shape[1],
                n_samples=n, f_pred=f_pred)
    return h1, H2, info


# ════════════════════════════════════════════════════════════════════════
# OPTION 2 — H2 ONLY, GIVEN A KNOWN h1
# ════════════════════════════════════════════════════════════════════════

def residual_identification(A, f, h1, reg=0.0):
    """
    Identify H2 only, assuming h1 has already been identified
    (e.g. from a separate linear-only Volterra fit).

    Solves on the RESIDUAL:
        r = f - A @ h1  =  B @ vech(H2)

    Parameters
    ----------
    A   : (n, p) regressor matrix
    f   : (n,)   training output
    h1  : (p,)   already-known first-order kernel
    reg : float  optional Tikhonov regularisation

    Returns
    -------
    H2   : (p, p) second-order kernel
    info : dict with r2, residual variance explained, etc.
    """
    n, p = A.shape
    f = np.asarray(f).reshape(-1)
    h1 = np.asarray(h1).reshape(-1)
    assert len(h1) == p, "h1 length must match A's column count"

    residual = f - A @ h1
    B, idx_pairs = build_quadratic_features(A)

    print(f"Residual identification: H2 has {len(idx_pairs)} unique entries")
    print(f"B matrix: {B.shape}")
    if B.shape[1] > n:
        print(f"  WARNING: underdetermined ({B.shape[1]} unknowns, "
              f"{n} samples). Use reg>0 or sparse_identification.")

    if reg > 0:
        BtB = B.T @ B
        Btr = B.T @ residual
        h2_vech = np.linalg.solve(BtB + reg*np.eye(B.shape[1]), Btr)
    else:
        h2_vech, _, _, _ = np.linalg.lstsq(B, residual, rcond=None)

    H2 = vech_to_full(h2_vech, p)

    residual_pred = B @ h2_vech
    r2 = 1 - np.var(residual - residual_pred) / np.var(residual)

    info = dict(r2=r2, n_unknowns=B.shape[1], n_samples=n,
                residual=residual, residual_pred=residual_pred)
    return H2, info


# ════════════════════════════════════════════════════════════════════════
# OPTION 3 — SPARSE IDENTIFICATION (when p*(p+1)/2 >> n)
# ════════════════════════════════════════════════════════════════════════

def sparse_identification(A, f, h1=None, method='lasso', **kwargs):
    """
    For large m (hence large p = 2m), the number of unique H2 entries
    p*(p+1)/2 can vastly exceed the number of training samples n.
    In that regime, use a sparse solver (LASSO / STLSQ / OMP) instead
    of plain least squares — exactly the tools from the earlier
    volterra_sparse_ls.py / volterra_sparsity_control.py scripts.

    Parameters
    ----------
    A      : (n, p) regressor matrix
    f      : (n,)   training output
    h1     : (p,) or None
             if None: jointly identify [h1 ; vech(H2)] sparsely
             if given: identify only vech(H2) on the residual, sparsely
    method : 'lasso', 'stlsq', or 'omp'
    kwargs : passed to the chosen sparse solver

    Returns
    -------
    h1   : (p,) — returned unchanged if it was provided, else identified
    H2   : (p, p)
    info : dict
    """
    n, p = A.shape
    f = np.asarray(f).reshape(-1)
    B, idx_pairs = build_quadratic_features(A)

    if h1 is None:
        D = np.hstack([A, B])
        target = f
    else:
        h1 = np.asarray(h1).reshape(-1)
        D = B
        target = f - A @ h1

    # import your earlier sparse solvers
    # (assumes volterra_sparse_ls.py is importable / in the same folder)
    try:
        from volterra_sparse_ls import solve_sparse
    except ImportError:
        raise ImportError(
            "sparse_identification requires volterra_sparse_ls.py "
            "(solve_lasso / solve_stlsq) to be importable.")

    x, sparse_info = solve_sparse(D, target, method=method, **kwargs)

    if h1 is None:
        h1_out      = x[:p]
        h2_vech_out = x[p:]
    else:
        h1_out      = h1
        h2_vech_out = x

    H2 = vech_to_full(h2_vech_out, p)

    info = dict(r2=sparse_info['r2'], nnz=sparse_info['nnz'],
                method=method, n_unknowns=D.shape[1], n_samples=n)
    return h1_out, H2, info


# ════════════════════════════════════════════════════════════════════════
# PREDICTION (for validation on test signals)
# ════════════════════════════════════════════════════════════════════════

def predict(A, h1, H2):
    """
    Evaluate f = A @ h1 + diag(A @ H2 @ A.T) efficiently
    (avoids forming the full A @ H2 @ A.T matrix, which would be
    (n, n) and wasteful — only the diagonal is needed).

    Parameters
    ----------
    A   : (n, p)
    h1  : (p,)
    H2  : (p, p) symmetric

    Returns
    -------
    f_pred : (n,)
    """
    linear_term    = A @ h1
    # efficient diagonal of A @ H2 @ A.T :  sum over columns of (A @ H2) * A
    quadratic_term = np.sum((A @ H2) * A, axis=1)
    return linear_term + quadratic_term


# ════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # ── synthetic problem ────────────────────────────────────────────
    m  = 15            # memory depth per input
    p  = 2 * m          # two inputs
    n  = 3000           # samples (plenty, p=30 << n=3000)

    # build A exactly as in your snippet
    ntsteps = n + m
    pitch_input = rng.standard_normal(ntsteps) * 0.05
    zdot_input  = rng.standard_normal(ntsteps) * 0.03

    A_pitch = np.zeros((n, m))
    A_zdot  = np.zeros((n, m))
    for icolumn in range(m):
        A_pitch[icolumn:, icolumn] = pitch_input[:n-icolumn]
        A_zdot[icolumn:, icolumn]  = zdot_input[:n-icolumn]
    A = np.hstack([A_pitch, A_zdot])

    # true kernels (sparse-ish, decaying)
    h1_true = np.concatenate([
        0.5*np.exp(-0.3*np.arange(m)),
        0.3*np.exp(-0.5*np.arange(m))
    ])
    H2_true = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            H2_true[i,j] = 0.1*np.exp(-0.2*abs(i-j)) * np.exp(-0.1*(i+j))
    H2_true = 0.5*(H2_true + H2_true.T)   # ensure symmetric

    f_true  = predict(A, h1_true, H2_true)
    f_noisy = f_true + 0.001 * rng.standard_normal(n)

    print("="*60)
    print(f"Problem: m={m}, p={p}, n={n}, "
          f"H2 unique entries={p*(p+1)//2}")
    print("="*60)

    # ── Option 1: joint identification ────────────────────────────────
    print("\n--- Joint identification (h1 and H2 together) ---")
    h1_id, H2_id, info1 = joint_identification(A, f_noisy, reg=1e-6)
    print(f"  R² = {info1['r2']:.6f}")
    print(f"  corr(h1_true, h1_id) = {np.corrcoef(h1_true, h1_id)[0,1]:.5f}")
    print(f"  corr(H2_true, H2_id) = "
          f"{np.corrcoef(H2_true.ravel(), H2_id.ravel())[0,1]:.5f}")

    # ── Option 2: H2 only, h1 known ───────────────────────────────────
    print("\n--- Residual identification (h1 known) ---")
    H2_res, info2 = residual_identification(A, f_noisy, h1_true, reg=1e-6)
    print(f"  R² (on residual) = {info2['r2']:.6f}")
    print(f"  corr(H2_true, H2_res) = "
          f"{np.corrcoef(H2_true.ravel(), H2_res.ravel())[0,1]:.5f}")

    # ── validate prediction on a fresh test signal ────────────────────
    print("\n--- Validation on unseen test signal ---")
    n_test = 1000
    ntsteps_t = n_test + m
    pitch_t = rng.standard_normal(ntsteps_t) * 0.05
    zdot_t  = rng.standard_normal(ntsteps_t) * 0.03
    A_pitch_t = np.zeros((n_test, m)); A_zdot_t = np.zeros((n_test, m))
    for icolumn in range(m):
        A_pitch_t[icolumn:, icolumn] = pitch_t[:n_test-icolumn]
        A_zdot_t[icolumn:, icolumn]  = zdot_t[:n_test-icolumn]
    A_test = np.hstack([A_pitch_t, A_zdot_t])

    f_test_true = predict(A_test, h1_true, H2_true)
    f_test_pred = predict(A_test, h1_id, H2_id)
    r2_test = 1 - np.var(f_test_true - f_test_pred)/np.var(f_test_true)
    print(f"  Test R² (joint-identified model) = {r2_test:.6f}")

    print("\nDone.")