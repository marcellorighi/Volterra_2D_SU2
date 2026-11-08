#!/usr/bin/env python3
"""
Zippered Schroeder multisine with a smooth ramp-in (v3: fixed PERIOD): the signal family of
Rigid_Plunging / Rigid_Pitching, plotted and written to a formatted text file for the identification software.

  x(t)    = r(t) * xm(t),                    xm(t)    = SIGN * sum_k a_k sin(w_k t + ph_k)
  xDot(t) = r'(t) * xm(t) + r(t) * xmDot(t), xmDot(t) = SIGN * sum_k a_k w_k cos(w_k t + ph_k)

r(t) is a quintic smootherstep, 0 -> 1 over [HOLD_T, HOLD_T + RAMP_T] (zero velocity AND acceleration at both
ends), so the motion starts at rest and the flow can settle while the amplitude is still small. The same ramp
multiplies pitch and plunge. For t >= T_START = HOLD_T + RAMP_T the signal is the pure multisine.

Comb: 2*N_LINES equally spaced lines from Omega/(2 pi) to Omega/(2 pi) + RANGE_HZ.
Odd lines (1st, 3rd, ...) -> pitch, even lines (2nd, 4th, ...) -> plunge.

v3: with SNAP_TO_RECORD the lines are snapped to multiples of 1/T_PERIOD, where T_PERIOD is a HARD-WIRED period
(not the run length, as in v2). After the ramp the signal is then exactly periodic with period T_PERIOD, and
it does NOT depend on N_ITER (the number of samples) any more. Pitch and plunge are exactly orthogonal over any
whole number of periods, so a run that stops early still contains usable whole periods; the periods can be
averaged. T_PERIOD = None gives the v2 behaviour (snap to the whole run after the ramp).
The same T_PERIOD (in seconds) has to be hard-wired in the C++ functions (it replaces the run-length window).
Quantities are in the units of the SU2 config (the nondimensionalisation cancels: w_nd * t_nd = w * t).
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ============================ settings ============================
SIGN = +1.0                  # sign of the motion: -1.0 = convention of the C++ reference (x = -sum a_k sin(...)),
                             #                     +1.0 = flipped. Position and velocity flip together.
# --- as in the SU2 config ---
PLUNGING_OMEGA = [0.0, 20.9, 2 * pi * 3.0]   # rad/s, [x, y, z]; only [1] (y) and [2] (z) are used
PLUNGING_AMPL = [0.0, 0.1, 0.0]                      # length units; position amplitude at Omega (0 = axis off)
PITCHING_OMEGA = 20.9                        # rad/s; use the SAME lowest frequency as the plunge axis in use
PITCHING_AMPL = 0.90                                  # degrees
FS_VELOCITY = 130.0
SUM_SIGN_PLUNGE = +1.0       # relative sign of the plunge term in the sum: theta + SUM_SIGN_PLUNGE * ydot/V

# --- zippered multisine (hard-wired in the C++) ---
N_LINES = 8                  # lines per motion
RANGE_HZ = 19.0              # width of the full comb [Hz]
SNAP_TO_RECORD = True        # lines on whole numbers of periods of T_PERIOD
PLUNGE_FLAT_VELOCITY = True  # equal xDot amplitude on every line (a_k = Ampl*Omega/w_k)
PITCH_FLAT_VELOCITY = False  # False: equal angle amplitude on every line
INCLUDE_PITCH = True

# --- period of the multisine after the ramp (hard-wired in the C++ as well) ---
T_PERIOD = 1.0               # [s] must be >= (2*N_LINES-1)/RANGE_HZ = 0.79 s and a multiple of DT; a longer
                             # period gives a finer line grid (2-4 s is a good range); None = v2 behaviour

# --- ramp-in ---
HOLD_T = 0.0                 # [s] rest before the ramp starts (0 = start ramping immediately)
RAMP_T = 0.5                 # [s] duration of the smooth ramp, 0 = no ramp (position then relative to t = 0)

# --- time ---
DT = 1.0e-4                  # physical time step [s]
N_ITER = 32000               # number of samples to generate; the signal does not depend on it (if T_PERIOD is set)

WRITE_FILE = False           # True: write OUT_FILE
OUT_FILE = "multisine_inputs.dat"
OUT_PNG = "multisine_inputs.png"
# ==================================================================

T_START = HOLD_T + RAMP_T if RAMP_T > 0 else 0.0     # full amplitude from here on
T_REC = N_ITER * DT                                  # length of the generated record
T_SNAP = T_PERIOD if T_PERIOD is not None else T_REC - T_START   # period of the signal after the ramp
N_PER = int(np.floor((T_REC - T_START) / T_SNAP + 1e-9))         # whole periods after the ramp
I0 = int(round(T_START / DT))                                    # analysis window: samples I0 ... I1-1
I1 = I0 + int(round(N_PER * T_SNAP / DT))
P_SAMPLES = int(round(T_SNAP / DT))

D_OMEGA = 2 * pi * RANGE_HZ / (2 * N_LINES - 1)
if T_PERIOD is not None and abs(T_PERIOD / DT - round(T_PERIOD / DT)) > 1e-6:
    print(f"WARNING: T_PERIOD ({T_PERIOD} s) is not a multiple of DT: the sampled signal is not exactly periodic")
if SNAP_TO_RECORD and D_OMEGA * T_SNAP < 2 * pi:
    print(f"WARNING: period ({T_SNAP:.3f} s) shorter than {(2 * N_LINES - 1) / RANGE_HZ:.2f} s: "
          f"lines are NOT snapped, pitch and plunge are only approximately orthogonal")
if not SNAP_TO_RECORD:
    print("WARNING: SNAP_TO_RECORD = False: the signal is not periodic and the orthogonality is only approximate")
if N_PER < 1:
    print("WARNING: the record is shorter than the ramp plus one period: no whole period available for analysis")


def ramp(t):
    """Quintic smootherstep r(t) and dr/dt: 0 for t < HOLD_T, 1 for t > HOLD_T + RAMP_T."""
    if RAMP_T <= 0:
        return np.ones_like(t), np.zeros_like(t)
    s = np.clip((t - HOLD_T) / RAMP_T, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2), 30 * s ** 2 * (1 - s) ** 2 / RAMP_T


def lines_and_phases(omega, zip_offset):
    """Angular frequencies [rad/s] and Schroeder phases of the lines of one motion."""
    n = N_LINES
    k = np.arange(1, n + 1)
    w = omega + (2 * k - 2 + zip_offset) * D_OMEGA        # line 2k-1+zip_offset of the full comb
    if SNAP_TO_RECORD and D_OMEGA * T_SNAP >= 2 * pi:     # period long enough: comb lines stay on distinct bins
        w = 2 * pi * np.floor(w * T_SNAP / (2 * pi) + 0.5) / T_SNAP
    return w, -pi * k * (k - 1) / n


def multisine(omega, ampl, zip_offset, flat_velocity, t):
    """Position and velocity as the solver sees them (ramp included). The sign of the motion is SIGN."""
    w, ph = lines_and_phases(omega, zip_offset)
    a = ampl * omega / w if flat_velocity else ampl * np.ones_like(w)
    arg = np.outer(t, w) + ph
    xm = SIGN * (a * np.sin(arg)).sum(axis=1)
    vm = SIGN * (a * w * np.cos(arg)).sum(axis=1)
    if RAMP_T > 0:
        r, dr = ramp(t)
        return r * xm, dr * xm + r * vm                   # r(0) = 0: starts at rest at the reference position
    return xm - SIGN * (a * np.sin(ph)).sum(), vm         # no ramp: position relative to t = 0


t = np.arange(N_ITER) * DT                    # t_i = i*DT, as time_new in the solver
cols, names, units = [t], ["time"], ["s"]
lines = {}

for iax, name in ((1, "y"), (2, "z")):
    if PLUNGING_AMPL[iax] != 0.0:
        x, xd = multisine(PLUNGING_OMEGA[iax], PLUNGING_AMPL[iax], 1, PLUNGE_FLAT_VELOCITY, t)
        cols += [x, xd]; names += [name, name + "dot"]; units += ["length", "length/s"]
        lines[name] = lines_and_phases(PLUNGING_OMEGA[iax], 1)[0] / (2 * pi)
        if INCLUDE_PITCH and abs(PLUNGING_OMEGA[iax] - PITCHING_OMEGA) > 1e-9:
            print(f"WARNING: PLUNGING_OMEGA[{iax}] != PITCHING_OMEGA, the {name} and pitch combs are not interleaved")

if INCLUDE_PITCH:
    th, thd = multisine(PITCHING_OMEGA, PITCHING_AMPL, 0, PITCH_FLAT_VELOCITY, t)
    cols += [th, thd]; names += ["theta", "thetadot"]; units += ["deg", "deg/s"]
    lines["theta"] = lines_and_phases(PITCHING_OMEGA, 0)[0] / (2 * pi)

# ------------------------------ file ------------------------------
data = np.column_stack(cols)
header = [
    "Zippered Schroeder multisine with ramp-in and fixed period (SU2 rigid plunge / pitch)",
    f"SIGN = {SIGN:+.0f}, N_LINES = {N_LINES}, RANGE_HZ = {RANGE_HZ}, SNAP_TO_RECORD = {SNAP_TO_RECORD}",
    f"DT = {DT} s, N_ITER = {N_ITER}, record length = {T_REC} s",
    f"HOLD_T = {HOLD_T} s, RAMP_T = {RAMP_T} s (quintic smootherstep), T_PERIOD = {T_PERIOD} s (period used: {T_SNAP} s)",
    f"analysis window: t = {T_START} s ... {I1 * DT} s = {N_PER} whole period(s), samples {I0} ... {I1 - 1}",
    f"PLUNGING_OMEGA = {PLUNGING_OMEGA} rad/s, PLUNGING_AMPL = {PLUNGING_AMPL}, flat velocity = {PLUNGE_FLAT_VELOCITY}",
    f"PITCHING_OMEGA = {PITCHING_OMEGA} rad/s, PITCHING_AMPL = {PITCHING_AMPL} deg, flat velocity = {PITCH_FLAT_VELOCITY}",
]
for key, f in lines.items():
    header.append(f"lines [Hz] of '{key}': " + " ".join(f"{v:.6f}" for v in f))
header.append("".join(f"{n:>18s}" for n in names))
header.append("".join(f"{'[' + u + ']':>18s}" for u in units))
if WRITE_FILE:
    np.savetxt(OUT_FILE, data, fmt="%18.10e", header="\n".join(header), comments="# ")
    print(f"wrote {OUT_FILE}: {data.shape[0]} rows, columns = {names}")

# ------------------------------ checks ------------------------------
win = np.zeros(N_ITER, dtype=bool)
win[I0:I1] = True                                           # analysis window: whole periods after the ramp
f_low = min(PLUNGING_OMEGA[i] for i in (1, 2) if PLUNGING_AMPL[i] != 0) / (2 * pi) if lines else 0
print(f"ramp: HOLD_T = {HOLD_T} s, RAMP_T = {RAMP_T} s -> full amplitude from t = {T_START} s")
print(f"period {T_SNAP:.4f} s; {N_PER} whole period(s) in the record -> analysis window t = {T_START} ... {I1 * DT:.4f} s "
      f"= {(I1 - I0) * DT * f_low:.1f} periods of the lowest comb frequency "
      f"(samples {I0} ... {I1 - 1}; {N_ITER - I1} samples after it are not used)")
print(f"C++: hard-wire periodTime = {T_SNAP} s (seconds) in both functions, in place of the run-length window")
if N_PER < 1:
    print("WARNING: no whole period available")
for key, f in lines.items():
    print(f"lines of {key:<6}[Hz]:", np.round(f, 3))
for name, col in zip(names[1:], cols[1:]):
    cw = col[win]
    per = (f", periodicity error = {np.abs(col[I0 + P_SAMPLES:I1] - col[I0:I1 - P_SAMPLES]).max():.1e}"
           if N_PER >= 2 else "")
    print(f"{name:<9} first sample = {col[0]:+.2e}, window: peak = {np.abs(cw).max():.5g}, mean = {cw.mean():.2e}, "
          f"crest = {np.abs(cw).max() / np.sqrt(np.mean(cw ** 2)):.2f}{per}")
if INCLUDE_PITCH and len(cols) > 3:
    a, b = cols[1][win] - cols[1][win].mean(), cols[-2][win] - cols[-2][win].mean()
    c_all = np.corrcoef(cols[1] - cols[1].mean(), cols[-2] - cols[-2].mean())[0, 1]
    print(f"correlation {names[1]} vs theta, zero lag: analysis window {(a @ b) / np.sqrt((a @ a) * (b @ b)):.1e}, "
          f"whole record (ramp included) {c_all:.1e}")

# plunge-equivalent angle of attack [deg]; FS_VELOCITY in the same length/time units as ydot, zdot
alpha_plunge = {nm[0]: np.rad2deg(col / FS_VELOCITY)          # 'ydot' -> 'y', 'zdot' -> 'z'
                for nm, col in zip(names, cols) if nm in ("ydot", "zdot")}
for ax_name, a in alpha_plunge.items():
    print(f"plunge-equivalent angle of attack ({ax_name}): peak = {np.abs(a[win]).max():.3f} deg")

# sum of the pitch angle and the plunge-equivalent angle: theta + ydot/V  [deg]
alpha_sum = {}
if INCLUDE_PITCH:
    th_col = cols[names.index("theta")]
    for ax_name, a in alpha_plunge.items():
        alpha_sum[ax_name] = th_col + SUM_SIGN_PLUNGE * a
        sw = alpha_sum[ax_name][win]
        print(f"theta + ydot/V ({ax_name}): peak = {np.abs(sw).max():.3f} deg, rms = {sw.std():.3f} deg "
              f"(pitch rms {th_col[win].std():.3f}, plunge rms {alpha_plunge[ax_name][win].std():.3f})")

# ------------------------------ plot ------------------------------
fig = plt.figure(figsize=(10, 11.5))
gs = fig.add_gridspec(4, 2)
ax_p, ax_v = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_t, ax_tr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_sum = fig.add_subplot(gs[2, :])
ax_s = fig.add_subplot(gs[3, :])
for nm, col in zip(names[1:], cols[1:]):
    if nm in ("y", "z"):
        ax_p.plot(t, col, label=nm)
    elif nm in ("ydot", "zdot"):
        ax_v.plot(t, col, label=nm)
    elif nm == "theta":
        ax_t.plot(t, col, color="C2", label=nm)
    elif nm == "thetadot":
        ax_tr.plot(t, col, color="C2", label=nm)

for ax_name, a in alpha_plunge.items():
    ax_t.plot(t, a, color="C3" if ax_name == "y" else "C1", lw=1,
              label=f"plunge-equivalent AoA ({ax_name})")

for ax_name, sm in alpha_sum.items():
    ax_sum.plot(t, th_col, color="C2", lw=0.8, alpha=0.5, label="pitch angle theta")
    ax_sum.plot(t, alpha_plunge[ax_name] * SUM_SIGN_PLUNGE, color="C3", lw=0.8, alpha=0.5,
                label=f"plunge-equivalent AoA ({ax_name}dot/V)")
    ax_sum.plot(t, sm, color="k", lw=1.2, label=f"sum: theta {'+' if SUM_SIGN_PLUNGE > 0 else '-'} {ax_name}dot/V")

for ax, ttl in ((ax_p, "plunge position"), (ax_v, "plunge velocity"),
                (ax_t, "pitch angle and plunge-equivalent AoA [deg]"), (ax_tr, "pitch rate [deg/s]"),
                (ax_sum, "sum of pitch angle and plunge-equivalent angle [deg]")):
    if T_START > 0:
        ax.axvspan(0, T_START, color="0.85", zorder=0)              # hold + ramp
    if I1 < N_ITER:
        ax.axvspan(I1 * DT, T_REC, color="0.93", zorder=0)          # after the last whole period
    for k in range(1, N_PER):
        ax.axvline(T_START + k * T_SNAP, color="0.6", lw=0.6, ls=":", zorder=0)   # period boundaries
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("t [s]"); ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

freqs = np.fft.rfftfreq(win.sum(), DT)                              # spectrum of the analysis window only
for nm, col in zip(names[1:], cols[1:]):
    if nm in ("ydot", "zdot", "thetadot"):
        sp = np.abs(np.fft.rfft(col[win])); sp /= sp.max()
        m = sp > 1e-6
        ax_s.vlines(freqs[m], -80, 20 * np.log10(sp[m]), color="C2" if nm == "thetadot" else "C0",
                    lw=2, alpha=0.8, label=nm)
ax_s.set_ylim(-60, 3); ax_s.set_xlim(0, RANGE_HZ + max(PLUNGING_OMEGA) / (2 * pi) + 2)
ax_s.set_xlabel("frequency [Hz]"); ax_s.set_ylabel("rel. velocity/rate spectrum [dB]")
ax_s.set_title("analysis window (whole periods after the ramp): interleaved lines, pitch (odd) and plunge (even)",
               fontsize=10)
ax_s.grid(alpha=0.3); ax_s.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.show()