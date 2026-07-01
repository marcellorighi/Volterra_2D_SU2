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
# STRATEGY 3 — ORTHOGONAL MATCHING PURSUIT  (exact nnz control)
# ════════════════════════════════════════════════════════════════════════
 
def solve_omp(A, y, n_nonzero, normalise_cols=True):
    """
    Orthogonal Matching Pursuit: greedily selects exactly n_nonzero
    columns of A that best explain y, then solves OLS on that support.
 
    This gives you EXACT control over the number of non-zero
    kernel coefficients — the most direct answer to "I want exactly
    k terms in my kernel".
 
    Parameters
    ----------
    A          : (n, p)
    y          : (n,)
    n_nonzero  : int    exact number of non-zero coefficients desired
 
    Returns
    -------
    h    : (p,) sparse kernel
    info : dict (method, r2, nnz, support)
    """
    from sklearn.linear_model import OrthogonalMatchingPursuit
 
    n, p = A.shape
    n_nonzero = min(n_nonzero, p)
 
    if normalise_cols:
        col_std = A.std(axis=0)
        col_std[col_std < 1e-14] = 1.0
        A_fit = A / col_std
    else:
        A_fit = A
        col_std = np.ones(p)
 
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero,
                                      fit_intercept=False)
    model.fit(A_fit, y)
    h_norm = model.coef_
    h      = h_norm / col_std
 
    y_pred  = A @ h
    r2      = _r2(y, y_pred)
    support = np.where(h != 0)[0]
 
    info = dict(method='omp', n_nonzero=n_nonzero,
                r2=r2, nnz=int(np.sum(h != 0)),
                support=support, y_pred=y_pred)
    return h, info
 
 

# ════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════
 
def _r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-14
    return 1.0 - ss_res / ss_tot
 
 # ════════════════════════════════════════════════════════════════════════
# OPTION 3 — SPARSE IDENTIFICATION (when p*(p+1)/2 >> n)
# ════════════════════════════════════════════════════════════════════════


def lasso_path_nnz(A, y, target_nnz, n_alphas=50, normalise_cols=True):
    """
    Compute the full LASSO regularisation path and return the solution
    with the number of non-zero coefficients closest to target_nnz.
 
    This is useful when you want to say "I expect about k non-zero
    kernel coefficients" without knowing the right alpha in advance.
 
    Parameters
    ----------
    A          : (n, p)
    y          : (n,)
    target_nnz : int    desired number of non-zero coefficients
    n_alphas   : int    number of alpha values on the path
 
    Returns
    -------
    h          : (p,) kernel at the alpha closest to target_nnz
    info       : dict (method, alpha_used, r2, nnz, path_nnz, path_r2)
    """
    from sklearn.linear_model import lasso_path
    from sklearn.preprocessing import StandardScaler
 
    n, p = A.shape
 
    if normalise_cols:
        scaler  = StandardScaler(with_mean=False)
        A_fit   = scaler.fit_transform(A)
        col_std = scaler.scale_
    else:
        A_fit   = A
        col_std = np.ones(p)
 
    # compute path: returns (alphas, coefs, _)
    # coefs shape: (p, n_alphas)
    alphas, coefs, _ = lasso_path(A_fit, y, n_alphas=n_alphas)
 
    # for each alpha: count nnz and compute R²
    path_nnz = np.sum(coefs != 0, axis=0)        # (n_alphas,)
    path_r2  = np.array([
        _r2(y, A_fit @ coefs[:, k]) for k in range(n_alphas)
    ])
 
    # find alpha where nnz is closest to target
    best_k    = np.argmin(np.abs(path_nnz - target_nnz))
    best_alpha = alphas[best_k]
    h_norm    = coefs[:, best_k]
    h         = h_norm / col_std
 
    y_pred = A_fit @ h_norm
    r2     = _r2(y, y_pred)
 
    print(f"  LASSO path: target_nnz={target_nnz}  "
          f"found nnz={path_nnz[best_k]}  alpha={best_alpha:.3e}  R²={r2:.4f}")
 
    info = dict(method='lasso_path', alpha_used=best_alpha,
                r2=r2, nnz=int(path_nnz[best_k]),
                path_nnz=path_nnz, path_r2=path_r2,
                alphas=alphas, y_pred=A @ h)
    return h, info

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
    # try:
    #     from volterra_sparse_ls import solve_sparse
    # except ImportError:
    #     raise ImportError(
    #         "sparse_identification requires volterra_sparse_ls.py "
    #         "(solve_lasso / solve_stlsq) to be importable.")

    # x, sparse_info = solve_sparse(D, target, method=method, **kwargs)

    # OMP
    # x, sparse_info = solve_omp(D, target, n_nonzero=264)

    # STLSQ
    # x, sparse_info = stlsq_min_nnz(D, target)

    # LASSO
    x, sparse_info = lasso_path_nnz(D, target,264)

    # print(f"  R²={i3['r2']:.4f}  nnz={i3['nnz']}")
    # corr = np.corrcoef(h_true, h3)[0,1]
    # print(f"  corr(h_true, h_omp)={corr:.5f}")


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
# STRATEGY 1 — STLSQ WITH MINIMUM NNZ GUARANTEE
# ════════════════════════════════════════════════════════════════════════
 
def stlsq_min_nnz(A, y, threshold=0.01, min_nnz=10,
                   max_iter=20, normalise_cols=True):
    """
    STLSQ with a hard lower bound on the number of non-zero coefficients.
 
    The threshold is applied normally at each iteration, but if the
    active set would fall below min_nnz, the weakest coefficients are
    kept until the minimum is satisfied.
 
    Parameters
    ----------
    A             : (n, p)  regressor matrix
    y             : (n,)    output vector
    threshold     : float   sparsity threshold on normalised coefficients
    min_nnz       : int     minimum number of coefficients to keep
                            set to 0 for standard STLSQ behaviour
    max_iter      : int
    normalise_cols: bool
 
    Returns
    -------
    h    : (p,) kernel in physical units
    info : dict  (method, r2, nnz, n_iter, threshold_used)
    """
    n, p = A.shape
    min_nnz = max(1, min(min_nnz, p))   # clamp to valid range
 
    if normalise_cols:
        col_std = A.std(axis=0)
        col_std[col_std < 1e-14] = 1.0
        A_fit = A / col_std
    else:
        A_fit = A
        col_std = np.ones(p)
 
    h_norm = np.linalg.lstsq(A_fit, y, rcond=None)[0]
    active = np.ones(p, dtype=bool)
    n_iter = 0
 
    for it in range(max_iter):
        n_iter = it + 1
 
        # standard threshold
        small  = np.abs(h_norm) < threshold
        n_keep = int(active.sum()) - int(small.sum())
 
        if n_keep < min_nnz:
            # not enough coefficients survive — keep the min_nnz largest
            abs_h   = np.abs(h_norm)
            abs_h[~active] = 0.0           # only among currently active
            cutoff_idx = np.argsort(abs_h)[::-1][:min_nnz]
            small      = np.ones(p, dtype=bool)
            small[cutoff_idx] = False
            print(f"  [iter {it+1}] min_nnz={min_nnz} active, "
                  f"threshold relaxed to keep {min_nnz} terms")
 
        h_norm[small] = 0.0
        active        = ~small
 
        h_new = np.zeros(p)
        h_new[active] = np.linalg.lstsq(A_fit[:, active], y, rcond=None)[0]
 
        # convergence: active set unchanged
        if np.array_equal(active, np.abs(h_new) >= threshold):
            h_norm = h_new
            break
        h_norm = h_new
 
    h      = h_norm / col_std
    y_pred = A @ h
    r2     = _r2(y, y_pred)
 
    info = dict(method='stlsq_min_nnz', threshold=threshold,
                min_nnz=min_nnz, r2=r2,
                nnz=int(np.sum(h != 0)), n_iter=n_iter,
                y_pred=y_pred)
    return h, info
 


# ════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    Mach = 0.8      # freestream Mach number
    AOA = 2    # wind off angle-of-attack [deg]

    # home_pitch = "/Users/marcello/Documents/Ael/bscw/Data_from_Michael/AOA0%s" % AOA
    home_pitch = "/Users/marcello/Documents/Ael/bscw/Data_from_Michael/AOA0%s_fine" % AOA

    # FINE 3 deg: 10900, 4 deg: 9000. 5 deg: 3150
    # MEDIUM  5 deg: 1500. #4 deg: 7800. #3 deg: 10600. #2 deg: 9350. # 1 deg: 9350. #0 deg: 9400. 

    # MEDIUM: 0DEG 8600 1DEG 8400 2deg 8900  3deg 10300 

    v_inf = 131.41  # freestream velocity from CFD simulation [m/s]
    rho_inf = 0.832 # freestream density from CFD simulation [kg/m^3]
    q_inf = 0.5*rho_inf*v_inf**2 
    delta_t = 0.0002 # transient time step [seconds]
    chord = 0.41

    dtau = delta_t * v_inf /(2 * chord)
    m = 96 #64 #320 #320 #240 #160 #80 #80

    gen_static_forces = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/gen_static_force_AOA%s_q150psf_SST_med_train.txt" % AOA) # generalised static force normalised by q_inf

    # medium mesh 
    gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_train.txt"%AOA) # INPUTS: generalized structural displacements
    gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_train.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

    # fine mesh 
    # gen_disp = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_fine_train.txt"%AOA) # INPUTS: generalized structural displacements
    # gen_force = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_fine_train.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

    pitch_input           = 1e0*gen_disp[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg

    ntsteps_CFD            = len(pitch_input)

    time_vector           = np.linspace(0.,ntsteps_CFD*delta_t, ntsteps_CFD)

    tau_vector = time_vector*2*v_inf/chord

    heave_input_int = - gen_disp[0,:]*0.106654261 # heave input if you want to convert to m
    heave_input_grad = np.gradient(heave_input_int,time_vector)/v_inf

    pitch_input_grad = np.gradient(gen_disp[1,:]*29.85872 /180*np.pi,time_vector)/v_inf

    cl_output = gen_force[0,:] / 0.106654 / (chord*chord*2)

    cm_output = gen_force[1,:]/ (29.86*np.pi/180) / (chord*chord*chord*2)

    cm_0 = gen_static_forces[1] / (29.86*np.pi/180) / (chord*chord*chord*2)
    cl_0 = gen_static_forces[0] / 0.106654 / (chord*chord*2)

    cl0 = np.average(cl_output)
    cm0 = np.average(cm_output)

    cl_output -= cl0
    cm_output -= cm0 

    rng = np.random.default_rng(0)

    # ── synthetic problem ────────────────────────────────────────────
    # m  = 15            # memory depth per input
    p  = 2 * m          # two inputs
    n  = ntsteps_CFD # 3000           # samples (plenty, p=30 << n=3000)

    # build A exactly as in your snippet
    ntsteps = n + m
    # pitch_input = rng.standard_normal(ntsteps) * 0.05
    zdot_input  = 1.0 * heave_input_grad #rng.standard_normal(ntsteps) * 0.03

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

    f_true  = 1.0 * cl_output #predict(A, h1_true, H2_true)
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
    # print("\n--- Residual identification (h1 known) ---")
    # H2_res, info2 = residual_identification(A, f_noisy, h1_true, reg=1e-6)
    # print(f"  R² (on residual) = {info2['r2']:.6f}")
    # print(f"  corr(H2_true, H2_res) = "
    #       f"{np.corrcoef(H2_true.ravel(), H2_res.ravel())[0,1]:.5f}")

    # # -- sparse identification -----------------------------------------
    h1_id, H2_id, info2 = sparse_identification(A, f_noisy, h1=None, method='lasso') #, **kwargs)

    # ── validate prediction on a fresh test signal ────────────────────
    print("\n--- Validation on unseen test signal ---")

    print(h1_id.shape)
    print(H2_id.shape)




    # ======== TEST SIGNAL 
    gen_disp_test = np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/input_AOA%s_q150psf_SST_med_test.txt"%AOA) # INPUTS: generalized structural displacements
    gen_force_test = 1.0*np.loadtxt("/Users/marcello/Documents/Ael/bscw/Data_from_Michael/output_AOA%s_q150psf_SST_med_test.txt"%AOA) # OUTPUT: generalised dynamic force normalised by q_inf

    ntsteps_test            = len(pitch_input)
    print("TEST", ntsteps_test)

    pitch_input_test           = 1e0*gen_disp_test[1,:]*29.85872 /180*np.pi # pitch input if you want to convert to deg

    heave_input_int = - gen_disp_test[0,:]*0.106654261 # heave input if you want to convert to m
    heave_input_grad_test = np.gradient(heave_input_int,time_vector)/v_inf

    cl_output_test = gen_force_test[0,:] / 0.106654 / (chord*chord*2)
    cm_output_test = gen_force_test[1,:]/ (29.86*np.pi/180) / (chord*chord*chord*2)

    # de trend

    pitch_avg = np.average(pitch_input_test)
    heave_avg = np.average(heave_input_grad_test)

    pitch_input_test -= pitch_avg
    heave_input_grad_test -= heave_avg

    cl0 = np.average(cl_output_test)
    cm0 = np.average(cm_output_test)

    cl_output_test -= cl0
    cm_output_test -= cm0 





    # n_test = 1000
    ntsteps_t = ntsteps_test + m
    pitch_t = 1.0 * pitch_input_test
    zdot_t  = 1.0 * heave_input_grad_test
    A_pitch_t = np.zeros((ntsteps_test, m)); A_zdot_t = np.zeros((ntsteps_test, m))
    for icolumn in range(m):
        A_pitch_t[icolumn:, icolumn] = pitch_t[:ntsteps_test-icolumn]
        A_zdot_t[icolumn:, icolumn]  = zdot_t[:ntsteps_test-icolumn]
    A_test = np.hstack([A_pitch_t, A_zdot_t])

    f_test_true = 1.0 * cl_output_test # predict(A_test, h1_true, H2_true)
    f_test_pred = predict(A_test, h1_id, H2_id)
    r2_test = 1 - np.var(f_test_true - f_test_pred)/np.var(f_test_true)
    print(f"  Test R² (joint-identified model) = {r2_test:.6f}")

    print("\nDone.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(f_test_true, 'b-', linewidth=2,label="TRUE")
    plt.plot(f_test_pred, 'k-', linewidth=2,label="PRED")
    plt.xlabel('t')
    plt.ylabel('Cl')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(h1_id, '.', linewidth=2)
    plt.plot(np.diag(H2_id), 'o', linewidth=2)
    plt.xlabel('Lag index')
    plt.ylabel('Coefficient')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(7, 5))
    # contour = plt.contourf(H2_id, cmap='viridis', levels=20)

    # # 3. Add a colorbar to show the scale
    # plt.colorbar(contour)

    # plt.title("Easiest Matrix Contour Map")
    # plt.show()

    rows, cols = 2*m, 2*m
    x = np.linspace(-3, 3, cols)
    y = np.linspace(-3, 3, rows)
    X, Y = np.meshgrid(x, y)      # Generates matching coordinate grids
    matrix = H2_id# Your matrix data goes here

    # 2. Set up the 3D plotting environment
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot the matrix as a 3D surface
    # rstride/cstride adjust step size; cmap sets the colors
    surface = ax.plot_surface(X, Y, matrix, cmap='viridis', 
                            linewidth=0, antialiased=True)

    # 4. Add a colorbar legend
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

    # 5. Label your axes
    ax.set_xlabel('X Axis (Columns)')
    ax.set_ylabel('Y Axis (Rows)')
    ax.set_zlabel('Z Value (Matrix Data)')
    ax.set_title('3D Surface Map of a Matrix')

    plt.show()