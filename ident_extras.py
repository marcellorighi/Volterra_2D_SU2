"""Gram-based helpers for the Volterra identification.

  1. gram_stats / solve_groups   : everything runs on G = A'W^2A, r = A'W^2Y, yy = sum W^2 Y^2
  2. lag_groups                  : your penalties, made relative to the block energy, with a
                                   multiplier per order  (point 2)
     tune_order_lambdas          : choose those multipliers by blocked cross-validation, computed
                                   from per-segment Gram matrices only (no refit on the data)
  3. greedy_select               : forward selection of blocks (block-level OMP) on the Gram matrix
                                   (point 3)
  4. laguerre_basis, lv_*        : Laguerre-Volterra model, i.e. kernels expanded in a basis along the
                                   lag axis, all products (cross terms included) of the basis outputs
                                   (point 4)

Conventions follow your script: `blocks` = [(name, order, length, dist), ...] from block_layout3,
`reg` = the dict passed to fit_blocks (lam_ridge, lam_smooth, lam_decay, decay_rate, lam_band,
band_decay, b), outputs Y = columns (CL, CM), weights w as from make_weights.

Run `python ident_extras.py` for a self-test on synthetic data.
"""

import itertools

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.blas import dsyrk
from scipy.signal import fftconvolve, lfilter

JITTERS = (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
TINY = np.finfo(float).tiny


# ====================================================================== Gram matrix

def gram_stats(A, Y, w, chunk=2048):
    """G = A' W^2 A, r = A' W^2 Y, yy = sum(W^2 Y^2) per output.
    A is an ndarray, or a callable build(r0, r1) returning rows r0:r1 of the design matrix
    (so the full matrix never has to exist). len(Y) is the number of rows."""
    n = len(Y)
    nc = (A(0, 1) if callable(A) else A[:1]).shape[1]
    G = np.zeros((nc, nc), order="F")
    r = np.zeros((nc, Y.shape[1]))
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        Ac = A(r0, r1) if callable(A) else A[r0:r1]
        wc = w[r0:r1]
        if not np.all(wc == 1.0):
            Ac = Ac * wc[:, None]
        G = dsyrk(1.0, Ac, beta=1.0, c=G, trans=1, lower=0, overwrite_c=1)
        r += Ac.T @ (wc[:, None] * Y[r0:r1])
    G = np.triu(G)
    G = G + np.triu(G, 1).T
    yy = np.sum((w ** 2)[:, None] * Y ** 2, axis=0)
    return G, r, yy


def cv_bounds(w, k):
    """k contiguous row ranges (start, stop) covering all rows, each holding about the same number
    of rows with non-zero weight. For a joint fit use your own list, e.g. one range per record."""
    rows = np.flatnonzero(w > 0)
    starts = [0] + [int(p[0]) for p in np.array_split(rows, k)[1:]]
    return list(zip(starts, starts[1:] + [len(w)]))


def segment_stats(A, Y, w, bounds):
    """Gram statistics of each segment. Their sum is the Gram matrix of the whole record."""
    out = []
    for a, b in bounds:
        Ak = (lambda r0, r1, a=a: A(a + r0, a + r1)) if callable(A) else A[a:b]
        out.append(gram_stats(Ak, Y[a:b], w[a:b]))
    return out


def sum_stats(stats):
    return (sum(s[0] for s in stats), sum(s[1] for s in stats), sum(s[2] for s in stats))


def error_from_gram(G, r, yy, h):
    """Relative error ||W(y - A h)|| / ||W y|| per output, from Gram statistics only."""
    sse = yy - 2 * np.sum(h * r, axis=0) + np.einsum("ij,ij->j", h, G @ h)
    return np.sqrt(np.maximum(sse, 0.0) / yy)


# ====================================================================== solving

def solve_spd(M, rhs, jitters=JITTERS):
    """Solve M x = rhs, M symmetric positive (semi)definite. M is overwritten (Jacobi scaling),
    pass a copy. Relative ridge from `jitters` if Cholesky fails, as in fit_blocks."""
    s = 1.0 / np.sqrt(np.maximum(np.diag(M), TINY))
    M *= s[:, None]
    M *= s[None, :]
    di = np.diag_indices_from(M)
    applied = 0.0
    for delta in jitters:
        M[di] += delta - applied
        applied = delta
        try:
            factor = cho_factor(M, lower=False)
        except LinAlgError:
            continue
        return s[:, None] * cho_solve(factor, s[:, None] * rhs), delta
    raise LinAlgError(f"not positive definite even with relative ridge {jitters[-1]:g}")


def solve_groups(G, r, groups, chosen=None):
    """Penalized solution restricted to the groups in `chosen` (default: all). Returns (h, jitter),
    h has the full length (zero outside the chosen groups)."""
    chosen = groups if chosen is None else chosen
    idx = np.concatenate([g["idx"] for g in chosen])
    M = G[np.ix_(idx, idx)]                              # copy
    pos = 0
    for g in chosen:
        k = len(g["idx"])
        M[pos:pos + k, pos:pos + k] += g["P"]
        pos += k
    c, used = solve_spd(M, r[idx])
    h = np.zeros((G.shape[0], r.shape[1]))
    h[idx] = c
    return h, used


# ====================================================================== point 2: relative penalties

def lag_groups(blocks, m, diag, reg, order_mult=None, relative=True):
    """One group per block, with the same penalties as fit_blocks (smoothness, decay with the older
    lag, band ridge) plus a ridge. With relative=True, ridge, smoothness and decay are multiplied
    by the mean column energy of the block, so a lambda is a fraction of the data energy and means
    the same for every order. order_mult = {order: factor} scales all penalties of that order.
    `diag` = diagonal of the Gram matrix."""
    om = order_mult or {}
    groups, start = [], 0
    for name, order, length, dist in blocks:
        sl = slice(start, start + length)
        mult = om.get(order, 1.0)
        scale = max(float(np.mean(diag[sl])), TINY) if relative else 1.0
        P = np.eye(length) * reg["lam_ridge"]
        if length >= 3 and reg["lam_smooth"] > 0:
            D2 = np.diff(np.eye(length), n=2, axis=0)
            P += reg["lam_smooth"] * (D2.T @ D2)
        di = np.diag_indices(length)
        P[di] += reg["lam_decay"] * (np.exp(reg["decay_rate"] * (np.arange(length) + dist) / (m - 1)) - 1.0)
        P *= scale * mult
        if dist > 0 and reg["b"] > 0:                    # already relative (uses the column energy)
            P[di] += mult * reg["lam_band"] * diag[sl] * (np.exp(reg["band_decay"] * dist / reg["b"]) - 1.0)
        groups.append(dict(name=name, order=order, idx=np.arange(start, start + length), P=P))
        start += length
    return groups


def block_energies(blocks, diag):
    """Mean column energy of each block (diag of the Gram matrix). A relative lambda is the absolute
    value of your current setting divided by this number: lam_rel = lam_abs / energy."""
    out, start = {}, 0
    for name, order, length, dist in blocks:
        out[name] = float(np.mean(diag[start:start + length]))
        start += length
    return out


def fit_gram(G, r, blocks, m, reg, order_mult=None, relative=True):
    """Drop-in replacement of fit_blocks working from Gram statistics: returns (H, jitter)."""
    groups = lag_groups(blocks, m, np.diag(G), reg, order_mult, relative)
    return solve_groups(G, r, groups)


def cv_score(stats, make_groups, order_mult=None):
    """Blocked leave-one-segment-out error. `stats` = segment_stats(...). make_groups(diag, order_mult)
    -> groups. The held-out error of a fold is computed from that segment's Gram matrix only:
    SSE = h'G_k h - 2 h'r_k + yy_k, so no pass over the data is needed. Returns the relative error
    per output."""
    G, r, yy = sum_stats(stats)
    num = 0.0
    for Gk, rk, yyk in stats:
        Gt, rt = G - Gk, r - rk
        h, _ = solve_groups(Gt, rt, make_groups(np.diag(Gt), order_mult))
        num = num + np.einsum("ij,ij->j", h, Gk @ h) - 2 * np.sum(h * rk, axis=0) + yyk
    return np.sqrt(np.maximum(num, 0.0) / yy)


def tune_order_lambdas(stats, make_groups, orders=(1, 2, 3, 4), factors=(0.01, 0.1, 1.0, 10.0, 100.0),
                       passes=1, verbose=True):
    """Coordinate search of one penalty multiplier per order by blocked cross-validation.
    Cost: passes * len(orders) * len(factors) * len(stats) solves of the full system."""
    mult = {o: 1.0 for o in orders}
    best = float(np.mean(cv_score(stats, make_groups, mult)))
    if verbose:
        print(f"CV error with all multipliers 1: {best:.2%}")
    for _ in range(passes):
        for o in orders:
            results = []
            for f in factors:
                trial = dict(mult)
                trial[o] = mult[o] * f
                results.append((float(np.mean(cv_score(stats, make_groups, trial))), trial[o]))
            score, val = min(results)
            if verbose:
                print(f"  order {o}: " + ", ".join(f"x{v:g}: {s:.2%}" for s, v in results)
                      + f"  -> multiplier {val:g}")
            mult[o], best = val, score
    return mult, best


# ====================================================================== point 3: greedy selection

def greedy_select(G, r, yy, groups, forced=(), max_groups=None, tol=1e-3, holdout=None, verbose=True):
    """Forward selection of groups (block-level orthogonal matching pursuit), on the Gram matrix.
    Each step fits every remaining group to the current residual, adds the one that reduces the
    squared error most (exactly, penalties included), and refits all selected groups together.
    Stops when the best relative reduction (mean over outputs) falls below `tol` or after
    `max_groups` additions. `forced` = names that are always in (e.g. ("p", "z")).
    holdout = (G_te, r_te, yy_te) from gram_stats on test rows gives a test error per step at no
    extra cost. Returns (h, path)."""
    by_name = {g["name"]: g for g in groups}
    chosen = [by_name[n] for n in forced]
    remaining = [g for g in groups if g["name"] not in forced]
    h = solve_groups(G, r, groups, chosen)[0] if chosen else np.zeros_like(r)

    def record(step, name):
        e = dict(step=step, name=name, train=error_from_gram(G, r, yy, h))
        if holdout is not None:
            e["test"] = error_from_gram(*holdout, h)
        return e

    path = [record(0, "+".join(forced) if forced else "-")]
    added = 0
    while remaining and (max_groups is None or added < max_groups):
        res = r - G @ h
        sse_now = yy * error_from_gram(G, r, yy, h) ** 2
        best = None
        for g in remaining:
            idx = g["idx"]
            Gbb = G[np.ix_(idx, idx)]
            c, _ = solve_spd(Gbb + g["P"], res[idx])
            gain = 2 * np.sum(c * res[idx], axis=0) - np.einsum("ij,ij->j", c, Gbb @ c)
            score = float(np.mean(gain / np.maximum(sse_now, TINY)))
            if best is None or score > best[0]:
                best = (score, g)
        score, g = best
        if score < tol:
            if verbose:
                print(f"stop: best remaining group '{g['name']}' would reduce the error by only {score:.2e} (tol {tol:g})")
            break
        chosen.append(g)
        remaining = [x for x in remaining if x is not g]
        h = solve_groups(G, r, groups, chosen)[0]
        added += 1
        path.append(record(added, g["name"]))
    if verbose:
        print(f"{'step':>4} {'added':<14} {'train CL':>9} {'train CM':>9}" + (f" {'test CL':>9} {'test CM':>9}" if holdout else ""))
        for e in path:
            line = f"{e['step']:>4} {e['name']:<14} {e['train'][0]:>9.2%} {e['train'][1]:>9.2%}"
            if holdout is not None:
                line += f" {e['test'][0]:>9.2%} {e['test'][1]:>9.2%}"
            print(line)
    return h, path


# ====================================================================== point 4: Laguerre-Volterra

def laguerre_basis(L, M, a):
    """M x L matrix with orthonormal discrete Laguerre functions (pole a, 0 < a < 1) in its columns,
    sampled at lags 0..M-1 and re-orthonormalized after truncation to M samples. Larger a = longer
    memory with fewer functions."""
    n = np.arange(M)
    phi = np.empty((M, L))
    phi[:, 0] = np.sqrt(1 - a * a) * a ** n
    for j in range(1, L):
        phi[:, j] = lfilter([-a, 1.0], [1.0, -a], phi[:, j - 1])
    Q, R = np.linalg.qr(phi)
    return Q * np.sign(np.diag(R))


def basis_capture(B, h):
    """Relative error of projecting a kernel h (length M, e.g. your current H1) on the basis span.
    Use it to choose L and a before identifying anything."""
    return float(np.linalg.norm(h - B @ (B.T @ h)) / np.linalg.norm(h))


def lag_features(u, B):
    """x_j(t) = sum_i B[i, j] u(t - i), zero before t = 0 (same convention as build_A123)."""
    n = len(u)
    return np.column_stack([fftconvolve(u, B[:, j])[:n] for j in range(B.shape[1])])


def lv_features(pitch, zdot, B):
    """Basis outputs of both inputs: columns 0..L-1 pitch, L..2L-1 zdot."""
    return np.hstack((lag_features(pitch, B), lag_features(zdot, B)))


def lv_terms(L, order):
    """All monomials of the 2L basis outputs up to `order` (cross terms included), as tuples of
    variable indices."""
    return [t for o in range(1, order + 1) for t in itertools.combinations_with_replacement(range(2 * L), o)]


def lv_matrix(X, terms, abs_terms=False):
    """Columns = products of the basis outputs listed in `terms`; optional extra columns x|x| per
    variable (the diagonal second-order form your build_A123 uses)."""
    nt = len(terms)
    cols = np.empty((X.shape[0], nt + (X.shape[1] if abs_terms else 0)))
    by_order = {}
    for j, t in enumerate(terms):
        by_order.setdefault(len(t), []).append(j)
    for o, js in by_order.items():
        cols[:, js] = np.prod(X[:, np.array([terms[j] for j in js])], axis=2)
    if abs_terms:
        cols[:, nt:] = np.abs(X) * X
    return cols


def lv_builder(X, terms, abs_terms=False):
    """Row-chunk builder for gram_stats: the design matrix is never stored."""
    return lambda r0, r1: lv_matrix(X[r0:r1], terms, abs_terms)


def lv_groups(terms, L, diag, lam, order_mult=None, abs_terms=False):
    """Groups by type (order, number of pitch factors), named p<k>z<l> like your blocks (p2z0 = pp,
    p1z1 = pz, ...). Ridge only (smoothness comes from the basis), relative to the group energy."""
    om = order_mult or {}
    types = {}
    for j, t in enumerate(terms):
        types.setdefault((len(t), sum(v < L for v in t)), []).append(j)
    groups = []
    for (o, npit), js in sorted(types.items()):
        idx = np.array(js)
        scale = max(float(np.mean(diag[idx])), TINY)
        groups.append(dict(name=f"p{npit}z{o - npit}", order=o, idx=idx,
                           P=lam * scale * om.get(o, 1.0) * np.eye(len(idx))))
    if abs_terms:
        idx = np.arange(len(terms), len(terms) + 2 * L)
        scale = max(float(np.mean(diag[idx])), TINY)
        groups.append(dict(name="abs2", order=2, idx=idx, P=lam * scale * om.get(2, 1.0) * np.eye(len(idx))))
    return groups


def lv_predict(X, terms, h, abs_terms=False, chunk=512):
    out = np.empty((X.shape[0], h.shape[1]))
    for r0 in range(0, X.shape[0], chunk):
        out[r0:r0 + chunk] = lv_matrix(X[r0:r0 + chunk], terms, abs_terms) @ h
    return out


def lv_kernels(h, terms, L, B):
    """First- and second-order kernels in the lag domain, for one output (h = one column).
    Returns k1 = {'p': (M,), 'z': (M,)} and k2 = {'pp', 'pz', 'zz': (M, M)}; pp and zz are symmetrized,
    pz[i, j] multiplies pitch(t-i) zdot(t-j). Compare k2 with the diagonals of your banded blocks."""
    M = B.shape[0]
    k1 = {"p": np.zeros(M), "z": np.zeros(M)}
    k2 = {k: np.zeros((M, M)) for k in ("pp", "pz", "zz")}

    def split(v):
        return ("p", v) if v < L else ("z", v - L)

    for c, t in zip(h, terms):
        if c == 0.0:
            continue
        if len(t) == 1:
            i, j = split(t[0])
            k1[i] += c * B[:, j]
        elif len(t) == 2:
            (ia, ja), (ib, jb) = split(t[0]), split(t[1])
            k2[ia + ib] += c * np.outer(B[:, ja], B[:, jb])
    for key in ("pp", "zz"):
        k2[key] = 0.5 * (k2[key] + k2[key].T)
    return k1, k2


# ====================================================================== self-test

def _lagmat(u, m):
    n = len(u)
    A = np.zeros((n, m))
    for i in range(m):
        A[i:, i] = u[:n - i]
    return A


def _smooth_noise(rng, n, k=15):
    return np.convolve(rng.standard_normal(n + k), np.ones(k) / k, "same")[:n]


def selftest():
    rng = np.random.default_rng(1)

    # ---- lag-domain machinery on a small problem with a known sparse truth
    m, n = 16, 4000
    u, v = _smooth_noise(rng, n), _smooth_noise(rng, n)
    U, V = _lagmat(u, m), _lagmat(v, m)
    cols = [U, V, U * U, V * V, U * V, U[:, :m - 1] * U[:, 1:], U[:, :m - 1] * V[:, 1:]]
    blocks = [("p", 1, m, 0), ("z", 1, m, 0), ("pp_d0", 2, m, 0), ("zz_d0", 2, m, 0),
              ("pz_d0", 2, m, 0), ("pp_d1", 2, m - 1, 1), ("pz_d1", 2, m - 1, 1)]
    A = np.hstack(cols)
    Ht = np.zeros((A.shape[1], 2))
    sl = {b[0]: slice(sum(x[2] for x in blocks[:i]), sum(x[2] for x in blocks[:i + 1])) for i, b in enumerate(blocks)}
    Ht[sl["p"], 0] = np.exp(-np.arange(m) / 4.0)
    Ht[sl["z"], 1] = 0.5 * np.exp(-np.arange(m) / 6.0)
    Ht[sl["pz_d0"], 0] = 0.4 * np.exp(-np.arange(m) / 5.0)
    Ht[sl["pz_d0"], 1] = 0.3 * np.exp(-np.arange(m) / 5.0)
    Y = A @ Ht
    Y = Y + 0.01 * Y.std(axis=0) * rng.standard_normal(Y.shape)
    w = np.ones(n)
    w[:200] = np.linspace(0, 1, 200)
    reg = dict(lam_ridge=1e-9, lam_smooth=1e-9, lam_decay=0.0, decay_rate=1.0, lam_band=1e-3, band_decay=1.0, b=1)

    G, r, yy = gram_stats(A, Y, w)
    # the Gram route equals the explicit weighted normal equations
    assert np.allclose(G, (A * w[:, None]).T @ (A * w[:, None])) and np.allclose(r, A.T @ ((w ** 2)[:, None] * Y))
    # callable builder gives the same statistics
    G2, r2, yy2 = gram_stats(lambda a, b: A[a:b], Y, w, chunk=700)
    assert np.allclose(G, G2) and np.allclose(r, r2) and np.allclose(yy, yy2)

    # blocked CV: Gram-only held-out error equals the explicit one
    bounds = cv_bounds(w, 4)
    stats = segment_stats(A, Y, w, bounds)
    assert all(np.allclose(a, b) for a, b in zip(sum_stats(stats), (G, r, yy)))
    make = lambda diag, om: lag_groups(blocks, m, diag, reg, om)
    cv = cv_score(stats, make)
    num, den = 0.0, 0.0
    for (a, b) in bounds:
        mask = np.ones(n, bool)
        mask[a:b] = False
        Gt, rt, _ = gram_stats(A[mask], Y[mask], w[mask])
        h, _ = solve_groups(Gt, rt, make(np.diag(Gt), None))
        num = num + np.sum((w[a:b, None] * (Y[a:b] - A[a:b] @ h)) ** 2, axis=0)
        den = den + np.sum((w[a:b, None] * Y[a:b]) ** 2, axis=0)
    assert np.allclose(cv, np.sqrt(num / den), rtol=1e-6), (cv, np.sqrt(num / den))

    # greedy selection finds the true blocks first
    groups = make(np.diag(G), None)
    h, path = greedy_select(G, r, yy, groups, forced=("p", "z"), max_groups=3, tol=1e-6, verbose=False)
    assert path[1]["name"] == "pz_d0", [p["name"] for p in path]
    assert path[1]["train"].max() < 0.03, path[1]["train"]

    # order multipliers can be tuned
    mult, best = tune_order_lambdas(stats, make, orders=(1, 2), factors=(0.1, 1, 10), verbose=False)
    assert best <= float(np.mean(cv)) + 1e-12

    # ---- Laguerre-Volterra: known system with a sub-kernel outside the basis span.
    # Broadband inputs: with only a few spectral lines the first-order kernel is unconstrained
    # between the lines (the prediction error would still be small).
    n, M, L = 20000, 128, 12
    pitch = 0.5 * _smooth_noise(rng, n, 5)
    zdot = 0.5 * _smooth_noise(rng, n, 5)
    i = np.arange(M)
    h1 = np.exp(-i / 15.0) * np.cos(0.2 * i)
    g = np.exp(-i / 10.0)
    xg = np.convolve(pitch, g)[:n]
    y = np.convolve(pitch, h1)[:n] + 0.4 * np.convolve(zdot, 0.5 * h1)[:n] + 0.5 * xg ** 2
    Yl = np.column_stack((y, 0.2 * y))
    B = laguerre_basis(L, M, 0.8)
    assert np.allclose(B.T @ B, np.eye(L), atol=1e-10)
    print(f"basis capture of h1: {basis_capture(B, h1):.2%}")
    X = lv_features(pitch, zdot, B)
    terms = lv_terms(L, 2)
    wl = np.ones(n)
    wl[:M] = 0.0
    cut = 14000
    G, r, yy = gram_stats(lv_builder(X[:cut], terms), Yl[:cut], wl[:cut], chunk=1024)
    groups = lv_groups(terms, L, np.diag(G), 1e-9)
    h, _ = solve_groups(G, r, groups)
    err_te = np.linalg.norm((lv_predict(X[cut:], terms, h) - Yl[cut:]), axis=0) / np.linalg.norm(Yl[cut:], axis=0)
    print(f"LV test error: {err_te}")
    assert err_te.max() < 0.05
    k1, k2 = lv_kernels(h[:, 0], terms, L, B)
    e1 = np.linalg.norm(k1["p"] - h1) / np.linalg.norm(h1)
    print(f"first-order pitch kernel error: {e1:.2%}")
    assert e1 < 0.1
    k2e = np.linalg.norm(k2['pp'] - 0.5 * np.outer(g, g)) / np.linalg.norm(0.5 * np.outer(g, g))
    print(f"second-order pp kernel error: {k2e:.2%}")
    assert k2e < 0.15
    print("self-test passed")


if __name__ == "__main__":
    selftest()