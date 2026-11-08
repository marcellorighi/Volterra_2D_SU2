import numpy as np
import openmdao.api as om
from scipy.special import eval_laguerre
from glob import glob
import pyvista as pv
from tqdm import tqdm
import csv 
from scipy.linalg import block_diag, cho_factor, cho_solve

 
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

def read_csv(filename):
    with open(filename, newline='') as f_input:
        next(f_input)
        next(f_input)
        return [list(map(float, row)) for row in csv.reader(f_input)]


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



def plot_aero_responses(h_cl_theta, h_cl_zdot, h_cl_thetadot, 
                        h_cm_theta, h_cm_zdot, h_cm_thetadot, 
                        omega, nt_int, dt):
    """
    Calculates and plots the aerodynamic lift (Cl) and moment (Cm) responses 
    for pitch, heave, and theta-dot components based on a sinusoidal AoA.
    """
    
    # --- 1. Define Time and Input Signal ---
    time_test = dt * np.arange(nt_int)
    # Sinusoidal Angle of Attack (1 degree amplitude)
    aoa = np.deg2rad(1.) * np.sin(omega * time_test)
    aoa_2 = np.deg2rad(1.) * np.sin(2 * omega * time_test)
    aoa_3 = np.deg2rad(1.) * np.sin(3 * omega * time_test)

    # --- 2. Calculate Responses ---
    # Assuming 'response' is a pre-defined function in your workspace
    cl_pitch    = response(h_cl_theta, aoa, nt_int)
    cl_heave    = response(h_cl_zdot, aoa, nt_int)
    cl_thetadot = response(h_cl_thetadot, aoa, nt_int)

    cm_pitch    = response(h_cm_theta, aoa, nt_int)
    cm_heave    = response(h_cm_zdot, aoa, nt_int)
    cm_thetadot = response(h_cm_thetadot, aoa, nt_int)

    cl_pitch_2    = response(h_cl_theta, aoa_2, nt_int)
    cm_pitch_2    = response(h_cm_theta, aoa_2, nt_int)

    cl_pitch_3    = response(h_cl_theta, aoa_3, nt_int)
    cm_pitch_3    = response(h_cm_theta, aoa_3, nt_int)

    # --- 3. Visualization ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Subplot 1: Input AoA
    ax[0].plot(time_test, aoa, label="AOA")
    ax[0].set_xlabel(r'$\tau\,[\,]$')
    ax[0].set_ylabel(r'$w\, [\,]$')
    ax[0].legend()
    ax[0].grid(True)

    # Subplot 2: Cl Components
    ax[1].plot(time_test, cl_pitch, label="Cl PITCH")
    ax[1].plot(time_test, cl_pitch_2, label="Cl PITCH 2")
    ax[1].plot(time_test, cl_pitch_3, label="Cl PITCH 3")
    ax[1].plot(time_test, cl_heave, label="Cl HEAVE")
    ax[1].plot(time_test, cl_thetadot, label="Cl Thetadot")
    ax[1].set_xlabel(r'$\tau\,[\,]$')
    ax[1].set_ylabel(r'$Cl\, [\,]$')
    ax[1].legend()
    ax[1].grid(True)

    # Subplot 3: Cm Components
    ax[2].plot(time_test, cm_pitch, label="Cm PITCH")
    ax[2].plot(time_test, cm_pitch_2, label="Cm PITCH 2")
    ax[2].plot(time_test, cm_pitch_3, label="Cm PITCH 3")
    ax[2].plot(time_test, cm_heave, label="Cm HEAVE")
    ax[2].plot(time_test, cm_thetadot, label="Cm Thetadot")
    ax[2].set_xlabel(r'$\tau\,[\,]$')
    ax[2].set_ylabel(r'$Cm\, [\,]$')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Optional: Return data for further analysis
    return time_test, aoa
    

def response(kernels, aoa, nt):
    ndof = kernels.shape[0] 
    
    AA = np.diag(np.ones(ndof-1),-1)
    BB = np.zeros((ndof))
    BB[0] = 1.
    CC = kernels
    
    q = np.zeros((ndof, nt))
    cn = np.zeros(nt)
    
    for it in range(1,nt):
        q[:,it] = AA@q[:,it-1] + BB*aoa[it] 
        cn[it] = CC@q[:,it]    
    
    return cn 
    
AOA = 5
MACH = 80
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_linear_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test_2" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_verylarge" % AOA
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine" % AOA
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_surfaceprint_test" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_3" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/M0%s_veryfine" % MACH
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_veryfine" % AOA
filename = home_pitch + "/history_05_00002.dat"    

chord = 0.4064 #1. 
v_inf = 123. * MACH / 80 # * 277.24 # 123. 
print("V freestream", v_inf)

# mach = 0.8
# v_inf = mach * 10 / 8 * 277.24 # 123. 
# chord = 1. 

tref = chord / (2 * v_inf)
subs = 48 
# subs = 76 

# dt = 7.5e-4 
dt = 2.5e-4 

iCd = 11
iCl = 12
iCm = 16

print(filename)
data = np.array(read_csv(filename)) 
cd = (data[0*subs::subs,iCd] ) #/ daoa_plunge
cl_output = (data[0*subs::subs,iCl] ) #/ daoa_plunge
cm_output = - (data[0::subs,iCm] ) # sign differs in 2 and 3D?
it_05 = data[0*subs::subs,0]
tau_vector = (it_05-2)*dt/tref
tt_05 = (it_05-2)*dt

print(cl_output.shape,it_05.shape,it_05[0],it_05[-1])

"""
cl_0 and cm_0 values 
AOA.    Cl          Cd          Cm
0       0.477260|    0.026601|    0.032471
1       0.669071|    0.041779|   -0.004728
2       0.742936|    0.058152|   -0.032949|
3       0.867393|    0.079169|   -0.062860
4       0.942390|    0.101759|   -0.081949|
5       1.042210|    0.126311|   -0.102394
6       1.116924|    0.155741|   -0.116118|

"""

cl0 = 0.942390
cm0 = -0.081949

# cm_output = -cm_output + 0.25*cl_output

ntsteps = len(cl_output)

# filter 
from scipy import signal

def butter_filter(data, btype, cutoff, fs, order=4):
    """
    btype   : 'low', 'high', or 'band'
    cutoff  : scalar for low/high, [low, high] for band  [Hz]
    fs      : sampling frequency [Hz]
    order   : filter order (4 is a safe default)
 
    Uses second-order sections (sosfilt) for numerical stability —
    always prefer this over ba-form for order > 2.
    """
    nyq = 0.5 * fs
    if btype == 'band':
        Wn = [cutoff[0] / nyq, cutoff[1] / nyq]
    else:
        Wn = cutoff / nyq
 
    sos = signal.butter(order, Wn, btype=btype, output='sos')
    return signal.sosfiltfilt(sos, data)   # zero-phase (forward + backward)


cl_output = butter_filter(cl_output, 'low', cutoff = 120., fs = 1./dt)
cm_output = butter_filter(cm_output, 'low', cutoff = 120., fs = 1./dt)


# test 

target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.85 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.08)

# veryfine 

# target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.8849]) 
# target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.95 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.065)

# verylarge
# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.90 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.130)

# test 2

# target_freqs_pitching = 20.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
# target_freqs_plunging = 14.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.20 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.12)


# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 1.00 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

norm_factor = 20. 

zdot_input = np.gradient(plunge_input,tt_05)/v_inf * norm_factor 
thetadot_input = np.gradient(pitch_input, tt_05) /v_inf * norm_factor 
pitch_input *= norm_factor 

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

X = Cp.copy()

Cp_mean = Cp.mean(axis=0)
X = Cp - Cp_mean

U, S, Vt = np.linalg.svd(X, full_matrices=False)
Phi = Vt.T
print("SVD complete")
#

# Use ALL modes
A_full = X @ Phi
Cp_full = Cp_mean + A_full @ Phi.T

error_full = np.linalg.norm(Cp - Cp_full) / np.linalg.norm(Cp)

print("Cp shape:", Cp.shape)
print("Phi shape:", Phi.shape)
print("A_full shape:", A_full.shape)
print("Full-rank reconstruction error:", error_full)
# 

energy = S**2
energy_fraction = energy / np.sum(energy)
cumulative_energy = np.cumsum(energy_fraction)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(
    np.arange(1, len(cumulative_energy)+1),
    100*cumulative_energy,
    '-o'
)

plt.xlabel('Number of modes')
plt.ylabel('Captured energy [%]')
plt.grid(True)

plt.show()

# plot modes 

r = 6

x_ref = x[0, :]

plt.figure(figsize=(8, 5))

for j in range(r):
    plt.plot(x_ref, Phi[:, j], label=f"Mode {j+1}")

plt.xlabel("x")
plt.ylabel("POD mode")
plt.grid(True)
plt.legend()
plt.show()

# for i in range(Cp.shape[0]):
#     plt.plot(x[i,:], Cp[i,:])
# plt.show()

# for i in range(Cp.shape[0]):
#     plt.plot(x[i,:], y[i,:])
# plt.show()

# spot check 

Phi_r = Phi[:, :r]

A_modes = X @ Phi_r

Cp_r = Cp_mean + A_modes @ Phi_r.T

print("Cp:     ", Cp.shape)
print("Phi_r:  ", Phi_r.shape)
print("A:      ", A_modes.shape)
print("Cp_r:   ", Cp_r.shape)

print("t:      ", t.shape)
print("pitch_input:   ", pitch_input.shape)

i = 7300

x_ref = x[i, :]

plt.figure()
plt.plot(x_ref, Cp[i, :], label="CFD")
plt.plot(x_ref, Cp_r[i, :], "--", label=f"POD ({r} modes)")

plt.xlabel("x")
plt.ylabel("$C_p$")
plt.legend()
plt.grid()
plt.show()

# time histories modes coeffs

plt.figure(figsize=(8, 5))

for j in range(r):
    plt.plot(A_modes[:, j], label=f"Mode {j+1}")

plt.xlabel("x")
plt.ylabel("POD mode")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================================
# ID
# ============================================================================



# n_discard = 200 
# ntsteps -= n_discard
# cl_output = cl_output[n_discard:]
# cm_output = cm_output[n_discard:]
# zdot_input = zdot_input[n_discard:]
# thetadot_input = thetadot_input[n_discard:]
# pitch_input = pitch_input[n_discard:]


# -----
# m = 240 #180 #180 #160 #80 #80
m = 96 #320 

# cl0 = cl_output[0]
# cm0 = cm_output[0]

# detrend

cl_output -= cl0
cm_output -= cm0 

cl_avg = np.average(cl_output)
cm_avg = np.average(cm_output)

cl_output -= cl_avg
cm_output -= cm_avg 

A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
    A_thetadot[icolumn:,icolumn] = thetadot_input[:ntsteps-icolumn] 
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

A = np.hstack( (A_pitch, A_zdot))

A_pitch2 = A_pitch*A_pitch
# A_pitch2 = np.abs(A_pitch)*A_pitch
A_zdot2 = A_zdot*A_zdot
#A_zdot2 = np.abs(A_zdot)*A_zdot
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

# TEMP
cl_norm_factor = 4. #4. #5. #5.
cm_norm_factor = 30. #30. # 20. 

delta_cl_norm_factor = 1. #30. 
delta_cm_norm_factor = 1. #30. 

cl_output *= cl_norm_factor
cm_output *= cm_norm_factor 

# A_123 = np.hstack( (A, A2, A3) )
A_123 = np.hstack( (A, A2, A3, A4) )

# A_123 = np.hstack( (A, A2, A3))

h_cl = solve_full_regularized(A_123, cl_output, m,
                                     lambda_ridge=1e-3,
                                     lambda_smooth=1e-2,
                                     lambda_decay=1e-1,
                                     decay_rate=5.0)

h_cm = solve_full_regularized(A_123, cm_output, m,
                                     lambda_ridge=1e-3,
                                     lambda_smooth=1e-2,
                                     lambda_decay=1e-1,
                                     decay_rate=5.0)

# h_mode_5_full, h_mode_6_full = h_full[:, 0], h_full[:, 1]

cl_check = A_123@h_cl / cl_norm_factor
cm_check = A_123@h_cm / cm_norm_factor

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cl_output / cl_norm_factor, 'b-', linewidth=1)
plt.plot(cl_check, 'k-', linewidth=1)
plt.xlabel(f't')
plt.ylabel(f'C_l')
#plt.title(f'h_cl - Order {order}')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cm_output / cm_norm_factor, 'b-', linewidth=1)
plt.plot(cm_check, 'k-', linewidth=1)
plt.xlabel(f't')
plt.ylabel(f'C_m')
#plt.title(f'h_cm - Order {order}')
plt.grid(True)

plt.tight_layout()
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

# h_mode_1 = solve_full_regularized(A_123, A_modes[:,0], m,
#                                      lambda_ridge=1e-3,
#                                      lambda_smooth=1e-2,
#                                      lambda_decay=1e-1,
#                                      decay_rate=5.0)

# h_mode_2 = solve_full_regularized(A_123, A_modes[:,1], m,
#                                      lambda_ridge=1e-3,
#                                      lambda_smooth=1e-2,
#                                      lambda_decay=1e-1,
#                                      decay_rate=5.0)

# h_mode_3 = solve_full_regularized(A_123, A_modes[:,2], m,
#                                      lambda_ridge=1e-3,
#                                      lambda_smooth=1e-2,
#                                      lambda_decay=1e-1,
#                                      decay_rate=5.0)

# h_mode_4 = solve_full_regularized(A_123, A_modes[:,3], m,
#                                      lambda_ridge=1e-3,
#                                      lambda_smooth=1e-2,
#                                      lambda_decay=1e-1,
#                                      decay_rate=5.0)

# mode_1_check = A_123@h_mode_1                                     
# mode_2_check = A_123@h_mode_2                                     
# mode_3_check = A_123@h_mode_3                                     
# mode_4_check = A_123@h_mode_4


# plt.figure(figsize=(12, 5))

# plt.subplot(2, 2, 1)
# plt.plot(mode_1_check, 'b-', linewidth=1)
# plt.plot(A_modes[:,0], 'k-', linewidth=1)
# plt.xlabel(f't')
# plt.ylabel(f'q_1')
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(mode_2_check, 'b-', linewidth=1)
# plt.plot(A_modes[:,1], 'k-', linewidth=1)
# plt.xlabel(f't')
# plt.ylabel(f'q_2')
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(mode_3_check, 'b-', linewidth=1)
# plt.plot(A_modes[:,2], 'k-', linewidth=1)
# plt.xlabel(f't')
# plt.ylabel(f'q_3')
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(mode_4_check, 'b-', linewidth=1)
# plt.plot(A_modes[:,3], 'k-', linewidth=1)
# plt.xlabel(f't')
# plt.ylabel(f'q_4')
# plt.grid(True)

# plt.tight_layout()
# plt.show()
