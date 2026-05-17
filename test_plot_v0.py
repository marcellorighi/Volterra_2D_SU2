import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import csv 
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag


def l_curve_analysis(A, y, lambdas=None):
    """
    Find optimal lambda via L-curve corner detection
    """
    if lambdas is None:
        lambdas = np.logspace(-6, 2, 50)  # Scan wide range
    
    residuals = []
    solution_norms = []
    
    for lam in lambdas:
        h = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        residuals.append(np.linalg.norm(A @ h - y))
        solution_norms.append(np.linalg.norm(h))
    
    residuals = np.array(residuals)
    solution_norms = np.array(solution_norms)
        
    # Find corner using curvature
    # Convert to log scale for curvature calculation
    log_res = np.log(residuals)
    log_sol = np.log(solution_norms)
    
    # Curvature formula for parametric curve
    curvature = np.zeros(len(lambdas) - 2)
    for i in range(1, len(lambdas) - 1):
        dx1 = log_res[i] - log_res[i-1]
        dy1 = log_sol[i] - log_sol[i-1]
        dx2 = log_res[i+1] - log_res[i]
        dy2 = log_sol[i+1] - log_sol[i]
        
        # Curvature approximation
        curvature[i-1] = abs(dx1*dy2 - dx2*dy1) / (dx1**2 + dy1**2)**1.5
    
    # Find maximum curvature (corner)
    corner_idx = np.argmax(curvature) + 1
    optimal_lambda = lambdas[corner_idx]
        
    return optimal_lambda, lambdas, residuals, solution_norms

def read_csv(filename):
    with open(filename, newline='') as f_input:
        next(f_input)
        next(f_input)
        return [list(map(float, row)) for row in csv.reader(f_input)]


def apply_smooth_start(t, signal, rise_time=0.1):
    # k=50 provides a very sharp but smooth rise
    # t0=rise_time/2 centers the ramp
    k = 50 / rise_time 
    t0 = rise_time / 2
    
    window = 1 / (1 + np.exp(-k * (t - t0)))
    
    # Force the very first point to be absolute zero
    window -= window[0]
    window /= window.max()
    
    return signal * window

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


mach = 0.7
chord = 1. 

v_inf = mach * 10 / 8 * 272
tref = chord / (2 * v_inf)
subs = 48 

# dt = 7.5e-4 
dt = 2.5e-4 

iCd = 11
iCl = 12
iCm = 16

home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA05_4/" #PAOA05/"
filename = home_pitch + "/history_05_00002.dat"    

print(filename)
data = np.array(read_csv(filename)) 
cd = (data[0*subs::subs,iCd] ) #/ daoa_plunge
cl_output = (data[0*subs::subs,iCl] ) #/ daoa_plunge
cm_output = (data[0::subs,iCm] ) #/ daoa_plunge # positive nose up
it_05 = data[0*subs::subs,0]
tau_vector = (it_05-2)*dt/tref
tt_05 = (it_05-2)*dt


ntsteps = len(cl_output)


#target_freqs = [2.9, 5.1, 8.0, 11.5] #, 15.9] 
# target_freqs_pitching = 11. / (2.*np.pi) * np.array([1.0, 1.47, 2.31, 3.57]) 
# target_freqs_plunging = 12. / (2.*np.pi) * np.array([1.0, 1.4142, 2.41, 3.77])

# target_freqs_pitching = 33. / (2.*np.pi) * np.array([1.0, 1.47]) 
# target_freqs_plunging = 21. / (2.*np.pi) * np.array([1.0, 1.4142])

target_freqs_pitching = 33. / (2.*np.pi) * np.array([1.0, 2.0]) 
target_freqs_plunging = 21. / (2.*np.pi) * np.array([1.0, 2.0])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.50 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

# target_freqs_pitching = 25. / (2.*np.pi) * np.array([1.0, 2.0]) 
# target_freqs_plunging = 20. / (2.*np.pi) * np.array([1.0, 2.0])

# t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.60 * np.pi/180.)
# t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.12)


zdot_input = - np.gradient(plunge_input,tt_05)/v_inf
thetadot_input = np.gradient(pitch_input, tt_05)/v_inf


# -----
resample_steps = 1 #4 #4 #10
m = 180 #180 #160 #80 #80

n_initial = 10
ntsteps -= n_initial

cl_output = cl_output[n_initial::resample_steps]
cm_output = cm_output[n_initial::resample_steps]

pitch_input = pitch_input[n_initial::resample_steps]
zdot_input = zdot_input[n_initial::resample_steps]
thetadot_input = thetadot_input[n_initial::resample_steps]

tt_05 = tt_05[n_initial::resample_steps]
tau_vector = tau_vector[n_initial::resample_steps]

cl0 = cl_output[0]
cm0 = cm_output[0]

cl_output -= cl0
cm_output -= cm0 

#----


# 
#zdot_input = apply_smooth_start(tt_05, zdot_input, rise_time=0.1)


fig, ax = plt.subplots(3, 1, figsize=(10, 10))

plt.rcParams.update({"font.family": "serif"})

plt.subplot(3,1,1)
plt.plot(tt_05,np.rad2deg(pitch_input),label="PITCH")
plt.plot(tt_05,np.rad2deg(pitch_input - zdot_input),label="PITCH + PLUNGE")
plt.plot(tt_05,np.rad2deg(thetadot_input),label="THETDOT")
plt.plot(tt_05,np.rad2deg(zdot_input),label="ZDOT")
plt.grid(True)
plt.legend() 

plt.subplot(3,1,2)
plt.plot(tt_05,cl_output,label=r"$C_l$")
plt.grid(True)
plt.legend() 

plt.subplot(3,1,3)
plt.plot(tt_05,cm_output,label=r"$C_m$")
plt.grid(True)
plt.legend() 

plt.tight_layout()
plt.show() 




A_pitch = np.zeros((ntsteps,m))
A_thetadot = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn]
    A_thetadot[icolumn:,icolumn] = thetadot_input[:ntsteps-icolumn]
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn]

A = np.hstack((A_pitch,A_thetadot,A_zdot))

print("A size", A.shape)

lambda_reg, _, _, _ = l_curve_analysis(A, cl_output)
single_block = np.diff(np.eye(m), n=2, axis=0)
D2 = block_diag(single_block,single_block,single_block)    
lambda_smooth = lambda_reg * 0.00
lambda_ridge = lambda_reg * 1.0 
decay_rate = 5  # Higher = stronger decay enforcement

single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cl = np.linalg.solve(A.T @ A + reg_matrix, A.T @ cl_output)

lambda_reg, _, _, _ = l_curve_analysis(A, cm_output)
lambda_smooth = lambda_reg * 0.00
lambda_ridge = lambda_reg * 1.0 

single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cm = np.linalg.solve(A.T @ A + reg_matrix, A.T @ cm_output)

cltest_check = A@theta_cl 
cmtest_check = A@theta_cm 

print(theta_cl.shape)

print(cltest_check.shape)

delta_cl = + cl_output - cltest_check 
delta_cm = + cm_output - cmtest_check 

A_pitch2 = A_pitch*A_pitch
A_thetadot2 = A_thetadot*A_thetadot
A_zdot2 = A_zdot*A_zdot
A_pitch_zdot = A_pitch*A_zdot
A_thetadot_zdot = A_thetadot*A_zdot

# A2 = np.hstack( (np.hstack((A_pitch2,A_thetadot2,A_heave2)),(A_pitch_heave,A_thetadot_heave)) )
A2 = np.hstack( (A_pitch2,A_thetadot2,A_zdot2,A_pitch_zdot,A_thetadot_zdot)) 

# uu, s, vh = linalg.svd(A2,full_matrices=False)    
lambda_reg, _, _, _ = l_curve_analysis(A2, cl_output)

single_block = np.diff(np.eye(m), n=2, axis=0)
D2 = block_diag(single_block,single_block,single_block,single_block,single_block)    
lambda_smooth = lambda_reg * 0.1
lambda_ridge = lambda_reg * 1.0 
single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)

L = block_diag(single_block, single_block, single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cl2 = np.linalg.solve(A2.T @ A2 + reg_matrix, A2.T @ delta_cl)


lambda_reg, _, _, _ = l_curve_analysis(A2, cm_output)
lambda_smooth = lambda_reg * 0.1
lambda_ridge = lambda_reg * 1.0 

single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cm2 = np.linalg.solve(A2.T @ A2 + reg_matrix, A2.T @ delta_cm)

cltest_check2 = cltest_check + A2@theta_cl2
cmtest_check2 = cmtest_check + A2@theta_cm2

# one-shot kernels identification: CL 
A_oneshot = np.hstack( (A_pitch, A_thetadot, A_zdot, A_pitch2,A_thetadot2,A_zdot2,A_pitch_zdot,A_thetadot_zdot))

lambda_reg_oneshot, _, _, _ = l_curve_analysis(A_oneshot, cl_output)
single_block = np.diff(np.eye(m), n=2, axis=0)
D2 = block_diag(single_block,single_block,single_block,single_block,single_block,single_block,single_block,single_block)    
lambda_smooth = lambda_reg * 0.0
lambda_ridge = lambda_reg * 1.0 
single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block, single_block, single_block,single_block,single_block,single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
h_cl = np.linalg.solve(A_oneshot.T @ A_oneshot + reg_matrix, A_oneshot.T @ cl_output)

# one-shot kernels identification: CM

lambda_reg_oneshot, _, _, _ = l_curve_analysis(A_oneshot, cm_output)
lambda_smooth = lambda_reg * 0.0
lambda_ridge = lambda_reg * 1.0 
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
h_cm = np.linalg.solve(A_oneshot.T @ A_oneshot + reg_matrix, A_oneshot.T @ cm_output)

cl_check_oneshot = A_oneshot@h_cl
cm_check_oneshot = A_oneshot@h_cm


# ================

fig, ax = plt.subplots(3, 2, figsize=(10, 8))

plt.subplot(3, 2, 1)

# plt.title("LAMBDA = "+str(eigvals[0]))
plt.plot(tau_vector,cl_output,label="Cl CFD")
plt.plot(tau_vector,cltest_check,label="Cl kernel 1")
plt.plot(tau_vector,cltest_check2,label="Cl kernel 2")
plt.plot(tau_vector,cl_check_oneshot,label="Cl one shot")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)

plt.plot(tau_vector,cm_output,label="Cm CFD")
plt.plot(tau_vector,cmtest_check,label="Cm kernel 1")
plt.plot(tau_vector,cmtest_check2,label="Cm kernel 2")
plt.plot(tau_vector,cm_check_oneshot,label="Cm one shot")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)

plt.bar(np.arange(m*3),theta_cl,width=0.45,label="Two-step")
plt.bar(np.arange(m*3) + 0.45,h_cl[:3*m],width=0.45,label="One Shot")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)

plt.bar(np.arange(m*3),theta_cm,width=0.45,label="Two-step")
plt.bar(np.arange(m*3) + 0.45,h_cm[:3*m],width=0.45,label="One Shot")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)

plt.bar(np.arange(m*5),theta_cl2)
plt.bar(np.arange(m*5)+ 0.45,h_cl[3*m:])
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)

plt.bar(np.arange(m*5),theta_cm2)
plt.bar(np.arange(m*5)+ 0.45,h_cm[3*m:])
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

