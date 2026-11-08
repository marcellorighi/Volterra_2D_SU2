#!/usr/bin/env python3
"""Plunge motions, mirroring the per-step logic of CVolumetricMovement::Rigid_Plunging.

Per iteration the solver gets deltaX (displacement over the step) and xDot (grid velocity at t_new).
The position is the running sum of deltaX. In all cases the total displacement tends to -AMPL.

    exp   : x = A*(exp(-w1*t) - 1)                                   xDot(0) = -A*w1
    zv    : x = A*((1 + w1*t)*exp(-w1*t) - 1)                        xDot(0) = 0
    bi    : x = A*((w2*exp(-w1*t) - w1*exp(-w2*t))/(w2 - w1) - 1)    xDot(0) = 0
            xDot = -A*w1*w2/(w2 - w1) * (exp(-w1*t) - exp(-w2*t)),  w2 > w1
            bi-exponential: slow decay w1, fast rise w2. Initial acceleration -A*w1*w2.
            w2 -> w1 gives zv, w2 -> infinity gives exp.
            Velocity spectrum: w1*w2 / ((w1 + i w)(w2 + i w)):
            flat up to w1, -20 dB/dec between w1 and w2, -40 dB/dec above w2.
    gauss : Gaussian velocity pulse, erf-shaped displacement (C-infinity smooth).
    ms    : multisine with Schroeder phases, x = -sum a_k sin(w_k t + ph_k), xDot = -sum a_k w_k cos(w_k t + ph_k),
            ph_k = -pi*k*(k-1)/N (k = 1..N), same sign convention as the original sine motion.
            Not a step: oscillates about the start position, peak displacement from the start = AMPL.
"""
import numpy as np
from math import erf, sqrt, pi
import matplotlib.pyplot as plt

# ---- user settings (nondimensional, as in the solver) ----
AMPL = 0.05        # Ampl / Lref  (total displacement)
OMEGA = 20.0       # slow rate w1 / Omega_ref  (exp, zv, bi)
OMEGA2 = 60.0      # fast rate w2 (bi only), must be > OMEGA
SIGMA = 0.05       # pulse width (gauss)
T0 = 4.0 * SIGMA   # pulse centre (gauss)
DT = 0.0001        # Delta_UnstTimeND
N_ITER = 22000      # record length T = N_ITER*DT = 3.0

# multisine: frequencies in the same units as the frequency axis (cycles per nondimensional time, w = 2*pi*f).
# Schroeder phases assume EQUALLY SPACED lines (for other spacings they can align all lines by coincidence:
# 3, 5.5, 10.5, 20 gave the worst-case crest factor). Here: 3, 8.67, 14.33, 20 = 9, 26, 43, 60 times 1/T (leakage-free).
MS_FREQS = np.linspace(3.0, 20.0, 7)
MS_FLAT_VELOCITY = True   # True: equal xDot amplitude per line (a_k ~ 1/w_k); False: equal position amplitude
MS_RAMP_T = 0.0           # > 0: smooth ramp-in (3s^2 - 2s^3) over this time, so that xDot(0) = 0

if OMEGA2 * DT > 0.5:
    print(f"WARNING: OMEGA2*DT = {OMEGA2 * DT:.2f}, the fast rise of 'bi' is under-resolved (aim for < 0.3)")
if 1.0 / (MS_FREQS.max() * DT) < 20:
    print(f"WARNING: only {1.0 / (MS_FREQS.max() * DT):.0f} time steps per period at the highest multisine "
          f"frequency (aim for >= 20)")
if SIGMA / DT < 3:
    print(f"WARNING: only {SIGMA / DT:.1f} time steps per sigma, the pulse is under-resolved")

verf = np.vectorize(erf)


def pos_gauss(t):
    """Displacement relative to t = 0 (so the position starts exactly at 0)."""
    return -AMPL / 2 * (verf((t - T0) / (SIGMA * sqrt(2))) - erf(-T0 / (SIGMA * sqrt(2))))


_k = np.arange(1, len(MS_FREQS) + 1)
MS_PHASE = -pi * _k * (_k - 1) / len(MS_FREQS)     # Schroeder phases (equal-power components)
_W = 2 * pi * MS_FREQS


def ms_raw(t, v0=1.0):
    """Multisine position and velocity at time(s) t, optionally with a smooth ramp-in."""
    t = np.atleast_1d(np.asarray(t, float))
    amp = v0 / _W if MS_FLAT_VELOCITY else v0 * np.ones_like(_W)
    arg = np.outer(t, _W) + MS_PHASE
    x = -(amp * np.sin(arg)).sum(axis=1)
    xd = -(amp * _W * np.cos(arg)).sum(axis=1)
    if MS_RAMP_T > 0:
        sr = np.clip(t / MS_RAMP_T, 0, 1)
        r = 3 * sr ** 2 - 2 * sr ** 3
        rd = np.where((sr > 0) & (sr < 1), 6 * sr * (1 - sr) / MS_RAMP_T, 0.0)
        x, xd = r * x, rd * x + r * xd
    return x, xd


_tf = np.arange(0, N_ITER * DT, DT / 10)
_xf, _ = ms_raw(_tf)
MS_V0 = AMPL / np.max(np.abs(_xf - _xf[0]))       # scale: peak displacement from the start position = AMPL


def crest(phase):
    v = np.cos(np.outer(_tf, _W) + phase).sum(axis=1)
    return np.max(np.abs(v)) / np.sqrt(np.mean(v ** 2))


def step(iter_, kind):
    """deltaX and xDot for one iteration, same time_new / time_old logic as the C++ (direct problem)."""
    t_new = iter_ * DT
    t_old = (iter_ - 1) * DT if iter_ != 0 else t_new
    if kind == "exp":
        dx = AMPL * (np.exp(-OMEGA * t_new) - np.exp(-OMEGA * t_old))
        xdot = -AMPL * OMEGA * np.exp(-OMEGA * t_new)
    elif kind == "zv":
        a_new, a_old = OMEGA * t_new, OMEGA * t_old
        dx = AMPL * ((1 + a_new) * np.exp(-a_new) - (1 + a_old) * np.exp(-a_old))
        xdot = -AMPL * OMEGA * a_new * np.exp(-a_new)
    elif kind == "bi":
        w1, w2 = OMEGA, OMEGA2
        k = 1.0 / (w2 - w1)
        f = lambda t: k * (w2 * np.exp(-w1 * t) - w1 * np.exp(-w2 * t))
        dx = AMPL * (f(t_new) - f(t_old))
        xdot = -AMPL * w1 * w2 * k * (np.exp(-w1 * t_new) - np.exp(-w2 * t_new))
    elif kind == "ms":
        xn, vn = ms_raw(t_new, MS_V0)
        xo, _ = ms_raw(t_old, MS_V0)
        dx, xdot = (xn - xo)[0], vn[0]
    else:  # gauss
        dx = pos_gauss(t_new) - pos_gauss(t_old)
        xdot = -AMPL / (SIGMA * sqrt(2 * pi)) * np.exp(-(t_new - T0) ** 2 / (2 * SIGMA ** 2))
    return dx, xdot


KINDS = (("exp", "exponential step"), ("zv", "zero initial velocity"),
         ("bi", "bi-exponential (zero initial velocity, steeper)"), ("gauss", "Gaussian velocity pulse"),
         ("ms", "multisine (Schroeder, 4 frequencies)"))

fig, (ax_x, ax_v, ax_f, ax_p) = plt.subplots(4, 1, figsize=(8, 12))
t = np.arange(N_ITER) * DT
n_fft = 2 ** 14
f = np.fft.rfftfreq(n_fft, DT)
w = 2 * pi * f

print(f"{'motion':<12}{'peak |xDot|':>14}{'xDot(0)':>12}{'xDDot(0)':>12}")
for kind, label in KINDS:
    dx, xdot = np.array([step(i, kind) for i in range(N_ITER)]).T
    line, = ax_x.plot(t, np.cumsum(dx), label=label)                      # position = sum of deltaX
    ax_v.plot(t, xdot, color=line.get_color(), label=label)              # grid velocity
    ax_v.plot(t[1:], dx[1:] / DT, ".", ms=3, color=line.get_color(), alpha=0.5)  # mean velocity deltaX/dt
    if kind == "ms" and MS_RAMP_T == 0:
        # integer number of periods in the record: exact lines, no leakage -> show only the lines
        sp = np.abs(np.fft.rfft(xdot)) * DT / AMPL
        fr = np.fft.rfftfreq(len(xdot), DT)
        m = (sp > 1e-6 * sp.max()) & (fr > 0)
        ax_f.plot(fr[m], 20 * np.log10(sp[m]), "o", color=line.get_color(), label=label)
        ax_p.loglog(fr[m], sp[m] / (2 * pi * fr[m]), "o", color=line.get_color(), label=label)
    else:
        spec = np.abs(np.fft.rfft(xdot, n_fft)) * DT / AMPL                  # DC value = 1 for steps
        ax_f.plot(f, 20 * np.log10(spec + 1e-12), color=line.get_color(), label=label)
        # position spectrum X(w) = V(w)/(i w): computed from the velocity spectrum, which avoids the
        # spurious 1/f leakage an FFT of x(t) would get from the jump at the end of the finite record
        ax_p.loglog(f[1:], spec[1:] / w[1:], color=line.get_color(), label=label)
    if kind == "ms":
        for fk in MS_FREQS:
            for ax in (ax_f, ax_p):
                ax.axvline(fk, color=line.get_color(), ls=":", lw=0.6, alpha=0.7)
    print(f"{kind:<12}{np.abs(xdot).max():>14.4f}{xdot[0]:>12.4f}{(xdot[1] - xdot[0]) / DT:>12.2f}")

ax_f.plot(f, -20 / np.log(10) * (w * SIGMA) ** 2 / 2, "k--", lw=0.8, label="analytic (gauss, bi)")
ax_f.plot(f, 20 * np.log10(OMEGA * OMEGA2 / np.abs((OMEGA + 1j * w) * (OMEGA2 + 1j * w))), "k--", lw=0.8)

ax_p.loglog(f[1:], 1 / w[1:], "k:", lw=1.2, label="ideal step: 1/(2 pi f)")

print(f"crest factor of xDot (multisine): Schroeder {crest(MS_PHASE):.2f}, zero phases {crest(0 * MS_PHASE):.2f}")

ax_x.axhline(-AMPL, color="k", ls="--", lw=0.8)
ax_x.set_ylabel("position  x(t)")
ax_v.set_ylabel("velocity  xDot")
ax_v.set_xlabel("t (nondimensional)")
ax_v.plot([], [], ".", color="gray", label="deltaX / dt (mean over step)")
ax_f.set_xscale("log")
F_MAX = min(0.5 / DT, 100.0)
ax_f.set_xlim(0.5, F_MAX)
ax_f.set_ylim(-70, 30)
ax_f.set_xlabel("frequency (1/time, nondimensional)")
ax_f.set_ylabel("|FFT(xDot)| / Ampl  [dB]")
ax_p.set_xlim(0.5, F_MAX)
ax_p.set_ylim(1e-6, 3)
ax_p.set_xlabel("frequency (1/time, nondimensional)")
ax_p.set_ylabel("|X(f)| / Ampl = |FFT(xDot)| / (2 pi f Ampl)")
for ax in (ax_x, ax_v):
    ax.grid(alpha=0.3)
for ax in (ax_f, ax_p):
    ax.grid(alpha=0.3, which="both")
ax_x.legend(fontsize=8)
ax_v.legend(fontsize=8)
ax_f.legend(fontsize=8)
ax_p.legend(fontsize=8)
fig.tight_layout()
fig.savefig("smoothed_step.png", dpi=120)
plt.show()