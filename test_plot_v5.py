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

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def time_integration(theta_cl, theta_cl2, theta_cm, theta_cm2, delta_t, m, nt_int, xi, 
                     v_inf, q, cl_0, cm_0):
    
    n_dof = 2 
    ndof_aero = 3*(m-1) 
    ndof = ndof_aero + 2 * n_dof 

    mass= 87; 
    Iy= 3.765; 
    kz=mass*(3.33*2*np.pi)**2; 
    ktheta=Iy*(5.20*2*np.pi)**2; 

    chord = 0.4032
    area = 0.3303 

    DT = delta_t*resample_steps
    dtau = delta_t*(2*v_inf)/chord * resample_steps

    # aerodynamic forces contribution by (n+1) aoa_eff

    beta = 0.5 

    Q_lift = q * area 
    Q_ma = q * area * chord 

    h_cl_theta = theta_cl[:m]
    h_cl_thetadot = theta_cl[m:2*m]
    h_cl_zdot = theta_cl[2*m:]

    h_cm_theta = theta_cm[:m]
    h_cm_thetadot = theta_cm[m:2*m]
    h_cm_zdot = theta_cm[2*m:]


    K_aa = np.array([[kz, 0],[0, ktheta]])
    M_aa = np.array([[mass, 0],[0,Iy]])
    C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

    AA = np.zeros((ndof,ndof))
    BB = np.zeros((ndof,ndof))

    Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
    Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

    Aphys_D = Bphys + DT*(1.-beta)*Aphys
    Bphys_D = Bphys - DT*beta*Aphys 

    AA[0:2*n_dof,0:2*n_dof] = Aphys_D 
    BB[0:2*n_dof,0:2*n_dof] = Bphys_D 

    BB[4:,4:] = np.eye(3*(m-1))

    # TEMP

    # Volterra coefficients at time (n+1)

    BB[2,1] -= DT * Q_lift * h_cl_theta[0] * beta 
    BB[2,2] -= - DT * Q_lift * h_cl_zdot[0] * beta / v_inf
    BB[2,3] -= DT * Q_lift * h_cl_thetadot[0] * beta / v_inf

    BB[3,1] -= DT * Q_ma * h_cm_theta[0] * beta 
    BB[3,2] -= - DT * Q_ma * h_cm_zdot[0] * beta / v_inf
    BB[3,3] -= DT * Q_ma * h_cm_thetadot[0] * beta / v_inf

    # Volterra coefficients at time (n)

    AA[2,1] += DT * Q_lift * h_cl_theta[1] * beta
    AA[2,2] += - DT * Q_lift * h_cl_zdot[1] * beta / v_inf
    AA[2,3] += DT * Q_lift * h_cl_thetadot[1] * beta / v_inf 

    AA[3,1] += DT * Q_ma * h_cm_theta[1] * beta
    AA[3,2] += - DT * Q_ma * h_cm_zdot[1] * beta / v_inf
    AA[3,3] += DT * Q_ma * h_cm_thetadot[1] * beta / v_inf

    AA[2,1] += DT * Q_lift * h_cl_theta[0] * (1-beta) 
    AA[2,2] += - DT * Q_lift * h_cl_zdot[0] * (1-beta) / v_inf
    AA[2,3] += DT * Q_lift * h_cl_thetadot[0] * (1-beta) / v_inf

    AA[3,1] += DT * Q_ma * h_cm_theta[0] * (1-beta) 
    AA[3,2] += - DT * Q_ma * h_cm_zdot[0] * (1-beta) / v_inf
    AA[3,3] += DT * Q_ma * h_cm_thetadot[0] * (1-beta) / v_inf

    # Volterra coefficients at times (n-1 -> n-m+1)

    icolumn1_theta = 4
    icolumn2_theta = icolumn1_theta + (m-2)

    icolumn1_thetadot = 4 + (m-1)
    icolumn2_thetadot = icolumn1_thetadot + (m-2)

    icolumn1_heavedot = 4 + 2 * (m-1)
    icolumn2_heavedot = icolumn1_heavedot + (m-2)

    AA[2,icolumn1_theta:icolumn2_theta] = DT * Q_lift * h_cl_theta[2:] * beta 
    AA[2,icolumn1_heavedot:icolumn2_heavedot] = DT * Q_lift * h_cl_zdot[2:] * beta 
    AA[2,icolumn1_thetadot:icolumn2_thetadot] = DT * Q_lift * h_cl_thetadot[2:] * beta 

    AA[3,icolumn1_theta:icolumn2_theta] = DT * Q_ma * h_cm_theta[2:] * beta 
    AA[3,icolumn1_heavedot:icolumn2_heavedot] = DT * Q_ma * h_cm_zdot[2:] * beta 
    AA[3,icolumn1_thetadot:icolumn2_thetadot] = DT * Q_ma * h_cm_thetadot[2:] * beta 

    # Volterra coefficients at times (n-1 -> n-m)

    icolumn1_theta = 4
    icolumn2_theta = icolumn1_theta + (m-1)

    icolumn1_thetadot = 4 + (m-1)
    icolumn2_thetadot = icolumn1_thetadot + (m-1)

    icolumn1_heavedot = 4 + 2 * (m-1)
    icolumn2_heavedot = icolumn1_heavedot + (m-1)

    AA[2,icolumn1_theta:icolumn2_theta] += DT * Q_lift * h_cl_theta[1:] * (1-beta) 
    AA[2,icolumn1_heavedot:icolumn2_heavedot] += DT * Q_lift * h_cl_zdot[1:] * (1-beta) 
    AA[2,icolumn1_thetadot:icolumn2_thetadot] += DT * Q_lift * h_cl_thetadot[1:] * (1-beta) 

    AA[3,icolumn1_theta:icolumn2_theta] += DT * Q_ma * h_cm_theta[1:] * (1-beta) 
    AA[3,icolumn1_heavedot:icolumn2_heavedot] += DT * Q_ma * h_cm_zdot[1:] * (1-beta) 
    AA[3,icolumn1_thetadot:icolumn2_thetadot] += DT * Q_ma * h_cm_thetadot[1:] * (1-beta) 

    # update aerodynamic states

    for i_dof in range(3):     
        irow1 = 4+1 + i_dof * (m-1)
        irow2 = irow1 + m -1 - 1 
        icolumn1 = irow1 - 1
        icolumn2 = irow2 - 1 #icolumn1 + m -1 
        print("update coeffs indices ",irow1,irow2,icolumn1,icolumn2)
        AA[irow1:irow2, icolumn1:icolumn2] = np.eye(m-2)

    AA[4,1] = 1.
    AA[4 + (m-1),3] = 1./v_inf #THETA DOT ???
    AA[4 + 2*(m-1),2] = - 1./v_inf #Z DOT??? 

    # print("row 5",AA[4,:])
    # print("row 6",AA[5,:])
    # print("row 7",AA[6,:])

    # time integration 

    #nt_int = 48*1024 # numerical integration steps 

    tau_int = np.linspace(0,nt_int*dtau,nt_int)
    time_int = np.linspace(0,nt_int*DT,nt_int)

    qq = np.zeros((ndof,nt_int))

    qping = 0.1 # heave 
    qq[2,0] = 0*qping # PING
    qq[3,0] = qping # PING

    fNL = np.zeros(ndof)

    BB_1 = linalg.inv(BB)

    f_0 = np.zeros(4 + 3*(m-1))
    f_0[2] = Q_lift * cl_0 
    f_0[3] = Q_ma * cm_0

    theta_lags = np.zeros(m)
    thetadot_lags = np.zeros(m)
    zdot_lags = np.zeros(m)

    n_subit = 3 

    for it in range(1,nt_int):
        for sub_it in range(n_subit):
            theta_lags[0] = (1-beta)*qq[1,it-1] + beta*qq[1,it]
            thetadot_lags[0] = (1-beta)*qq[3,it-1]/v_inf + beta**qq[3,it]/v_inf 
            zdot_lags[0] = -(1-beta)*qq[2,it-1]/v_inf - beta*qq[2,it]/v_inf 

            theta_lags[1:] = (1-beta)*qq[4:4+m-1,it-1] + beta*qq[4:4+m-1,it]
            thetadot_lags[1:] = (1-beta)*qq[4 + m-1: 4+(m-1)*2,it-1] + beta*qq[4 + m-1: 4+(m-1)*2,it]
            zdot_lags[1:] = (1-beta)*qq[4 + 2*(m-1):,it-1] + beta*qq[4 + 2*(m-1):,it]

            all_lags_1 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, theta_lags, thetadot_lags])
            all_lags_2 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, zdot_lags, zdot_lags])

            cl_NL = np.dot(theta_cl2, all_lags_1*all_lags_2)
            cm_NL = np.dot(theta_cm2, all_lags_1*all_lags_2)

            # print(f_NL_lift, f_NL_ma)

            f_0[2] = Q_lift * (cl_0 + cl_NL)
            f_0[3] = Q_ma * (cm_0 + cm_NL)
        
            # if flutter_study == 'NL':       
            #     fNL[2] = q*area*chord*(kernels_cm_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
            #     fNL[3] = q*area*(kernels_cl_NL)@(qq[4:,it-1]*np.abs(qq[4:,it-1])).T
            #     qq[:,it] = AA@(qq[:,it-1] + DT*fNL) 
            # if flutter_study == 'lin':
            qq[:,it] = BB_1@(AA@qq[:,it-1] + DT*f_0) 


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$\theta$")
    plt.xlabel(r'$t\,[s]$')
    plt.ylabel(r'$z\, [m]$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time_int,qq[1,:]*180/np.pi,'k-',linewidth=2,label=r"$\theta$")
    plt.xlabel(r'$t\,[s]$')
    plt.ylabel(r'$\theta\, [DEG]$')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() 
    plt.show()

    return qq


home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA00_R12/" 
#home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA05_1/" #PAOA05/"
filename = home_pitch + "/history_05_00002.dat"    

mach = 0.8
chord = 0.4064 #1. 
v_inf = 123. #mach * 10 / 8 * 277.24 # 123. 

# mach = 0.8
# v_inf = mach * 10 / 8 * 277.24 # 123. 
# chord = 1. 

tref = chord / (2 * v_inf)
subs = 48 

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

# cm_output = -cm_output + 0.25*cl_output

ntsteps = len(cl_output)

target_freqs_pitching = 18.5 / (2.*np.pi) * np.array([1.4048, 2.6437, 5.0113]) 
target_freqs_plunging = 18.5 / (2.*np.pi) * np.array([1.0, 1.9226, 3.6286])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.50 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

factor_pitch = 1.#20. 
factor_tdot = 1.#100. 
factor_zdot = 1.#20. 

zdot_input = np.gradient(plunge_input,tt_05)/v_inf * factor_zdot
thetadot_input = np.gradient(pitch_input, tt_05) /v_inf * factor_tdot
pitch_input *= factor_pitch 

# -----
resample_steps = 1 #4 #4 #10
m = 180 #180 #160 #80 #80

decay_rate = 2.125 #0.25  

n_initial = 0
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
plt.plot(tt_05,np.rad2deg(pitch_input + zdot_input),label="PITCH + PLUNGE")
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

# theta_cl[m:2*m] *= factor

#decay_rate = 5. #0.25  
lambda_reg, _, _, _ = l_curve_analysis(A, cm_output)
lambda_smooth = lambda_reg * 0.00
lambda_ridge = lambda_reg * 1.0 

single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cm = np.linalg.solve(A.T @ A + reg_matrix, A.T @ cm_output) 

# theta_cm[m:2*m] *= factor

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
lambda_smooth = lambda_reg * 0.0
lambda_ridge = lambda_reg * 1.0 
single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)

L = block_diag(single_block, single_block, single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cl2 = np.linalg.solve(A2.T @ A2 + reg_matrix, A2.T @ delta_cl) 

# theta_cl2[m:2*m] *= factor**2
# theta_cl2[4*m:] *= factor



lambda_reg, _, _, _ = l_curve_analysis(A2, cm_output)
lambda_smooth = lambda_reg * 0.0
lambda_ridge = lambda_reg * 1.0 

#decay_rate = 5. #0.25  
single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block, single_block, single_block)
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
theta_cm2 = np.linalg.solve(A2.T @ A2 + reg_matrix, A2.T @ delta_cm) 

# theta_cm2[m:2*m] *= factor**2
# theta_cm2[4*m:] *= factor

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

# h_cl[m:2*m] *= factor 
# h_cl[5*m:5*m] *= factor**2 
# h_cl[7*m:] *= factor 

# one-shot kernels identification: CM

#decay_rate = 5. #0.125  # Higher = stronger decay enforcement
single_block = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
L = block_diag(single_block, single_block, single_block, single_block, single_block,single_block,single_block,single_block)

lambda_reg_oneshot, _, _, _ = l_curve_analysis(A_oneshot, cm_output)
lambda_smooth = lambda_reg * 0.0
lambda_ridge = lambda_reg * 1.0 
reg_matrix = lambda_smooth * D2.T @ D2 + lambda_ridge * L
h_cm = np.linalg.solve(A_oneshot.T @ A_oneshot + reg_matrix, A_oneshot.T @ cm_output) 

# h_cm[m:2*m] *= factor 
# h_cm[5*m:5*m] *= factor**2 
# h_cm[7*m:] *= factor 

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

# --- time integration --- # 

theta_cl = h_cl[:3*m]
theta_cm = h_cm[:3*m]

theta_cl2 = h_cl[3*m:]
theta_cm2 = h_cm[3*m:]

# pitch
theta_cl[:m] *= factor_pitch
theta_cl2[:m] *= factor_pitch**2
theta_cl2[3*m:4*m] *= factor_pitch

theta_cm[:m] *= factor_pitch
theta_cm2[:m] *= factor_pitch**2
theta_cm2[3*m:4*m] *= factor_pitch

# t dot

theta_cl[m:2*m] *= factor_tdot
theta_cl2[m:2*m] *= factor_tdot**2
theta_cl2[4*m:] *= factor_tdot

theta_cm[m:2*m] *= factor_tdot
theta_cm2[m:2*m] *= factor_tdot**2
theta_cm2[4*m:] *= factor_tdot

# z dot

theta_cl[2*m:3*m] *= factor_zdot
theta_cl2[2*m:3*m] *= factor_zdot**2
theta_cl2[3*m:] *= factor_tdot

theta_cm[2*m:3*m] *= factor_zdot
theta_cm2[2*m:3*m] *= factor_zdot**2
theta_cm2[3*m:] *= factor_tdot


# q = 2000. 
nt_int = 12*1024 
xi = 0.000 

results = time_integration(theta_cl, theta_cl2, theta_cm, theta_cm2, dt, m, nt_int, xi, v_inf=123., q = 7000., cl_0 = cl0, cm_0=cm0)

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

h_cl_theta = theta_cl[:m]
h_cl_thetadot = theta_cl[m:2*m]
h_cl_zdot = theta_cl[2*m:]

h_cm_theta = theta_cm[:m]
h_cm_thetadot = theta_cm[m:2*m]
h_cm_zdot = theta_cm[2*m:]


# check

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

    # --- 2. Calculate Responses ---
    # Assuming 'response' is a pre-defined function in your workspace
    cl_pitch    = response(h_cl_theta, aoa, nt_int)
    cl_heave    = response(h_cl_zdot, aoa, nt_int)
    cl_thetadot = response(h_cl_thetadot, aoa, nt_int)

    cm_pitch    = response(h_cm_theta, aoa, nt_int)
    cm_heave    = response(h_cm_zdot, aoa, nt_int)
    cm_thetadot = response(h_cm_thetadot, aoa, nt_int)

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
    ax[1].plot(time_test, cl_heave, label="Cl HEAVE")
    ax[1].plot(time_test, cl_thetadot, label="Cl Thetadot")
    ax[1].set_xlabel(r'$\tau\,[\,]$')
    ax[1].set_ylabel(r'$Cl\, [\,]$')
    ax[1].legend()
    ax[1].grid(True)

    # Subplot 3: Cm Components
    ax[2].plot(time_test, cm_pitch, label="Cm PITCH")
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


time_test, aoa = plot_aero_responses(h_cl_theta, h_cl_zdot, h_cl_thetadot, 
                        h_cm_theta, h_cm_zdot, h_cm_thetadot, 
                        omega = 20., nt_int = 5000, dt = dt)



