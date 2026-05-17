import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import csv 
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag
from tqdm import tqdm


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

def analyze_discrete_instability(A,dt):
    """
    Computes eigenvalues of A, detects unstable modes (|eig| > 1),
    and prints eigenvalues with the first 4 eigenvector components.
    """
    # 1. Compute eigenvalues (w) and right eigenvectors (v)
    # v[:, i] is the eigenvector corresponding to eigenvalue w[i]
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 2. Calculate magnitudes
    magnitudes = np.abs(eigenvalues)
    
    # 3. Find indices where magnitude > 1
    unstable_indices = np.where(magnitudes > 1)[0]
    
    if len(unstable_indices) == 0:
        print("No unstable modes detected (all magnitudes <= 1).")
        return
    
    print(f"Detected {len(unstable_indices)} unstable mode(s):\n")
    print(f"{'Eigenvalue':<41} | {'Frequency (Hz)':<20} | {'Magnitude':<10} | {'First 4 Vector Components'}")
    print("-" * 85)
    
    for idx in unstable_indices:
        eig_val = eigenvalues[idx]
        mag = magnitudes[idx]
        phase = np.angle(eig_val)

        # 2. Convert to frequency in Hz
        frequency_hz = np.abs(phase) / (2 * np.pi * dt)

        # 3. Get the damping ratio (for completeness)
        magnitude = np.abs(eig_val)
        damping_rate = np.log(magnitude) / dt
        
        # Extract the corresponding eigenvector (column idx)
        # and take the first 4 components
        vec_comp = eigenvectors[:4, idx]
        
        # Formatting the complex numbers for readability
        comp_str = ", ".join([f"{c:.4f}" for c in vec_comp])
        
        print(f"{str(eig_val):<30}| {str(frequency_hz):<20} | {mag:<10.4f} | [{comp_str}]")


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

    DT = delta_t * 1.0 # *resample_steps
    dtau = delta_t*(2*v_inf)/chord #* resample_steps

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

    # linear stability analysis 
    analyze_discrete_instability(BB_1@AA,DT)

    for it in tqdm(range(1,nt_int)): #range(1,nt_int):
        # print("it =", it)
        for sub_it in range(n_subit):
            theta_lags[0] = (1-beta)*qq[1,it-1] + beta*qq[1,it]
            thetadot_lags[0] = (1-beta)*qq[3,it-1]/v_inf + beta**qq[3,it]/v_inf 
            zdot_lags[0] = -(1-beta)*qq[2,it-1]/v_inf - beta*qq[2,it]/v_inf 

            theta_lags[1:] = (1-beta)*qq[4:4+m-1,it-1] + beta*qq[4:4+m-1,it]
            thetadot_lags[1:] = (1-beta)*qq[4 + m-1: 4+(m-1)*2,it-1] + beta*qq[4 + m-1: 4+(m-1)*2,it]
            zdot_lags[1:] = (1-beta)*qq[4 + 2*(m-1):,it-1] + beta*qq[4 + 2*(m-1):,it]

            all_lags_1 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, theta_lags, thetadot_lags, thetadot_lags])
            all_lags_2 = np.concatenate([theta_lags, thetadot_lags, zdot_lags, zdot_lags, zdot_lags, theta_lags])

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
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_test/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_test/" % AOA
filename = home_pitch + "/kernels_1_and_2.npz"    

# Load the archive
data = np.load(filename)

# Access the individual arrays
m = data['m']
cl_0 = data['cl_0']
cm_0 = data['cm_0']
h_cl_1 = data['h_cl_1']
h_cl_2 = data['h_cl_2']
h_cm_1 = data['h_cm_1']
h_cm_2 = data['h_cm_2']


# ================

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

plt.subplot(2, 2, 1)

plt.plot(h_cl_1,label="Two-step")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)

plt.plot(h_cm_1,label="Two-step")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)

plt.plot(h_cl_2,label="Two-step")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)

plt.plot(h_cm_2,label="Two-step")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# temp
# m = 240 
# h_cl_2 = h_cl_2[:5*m]
# h_cm_2 = h_cm_2[:5*m]
# norm_factor = 20. 
# h_cl_1 *= norm_factor
# h_cm_1 *= norm_factor
# h_cl_2 *= norm_factor**2
# h_cm_2 *= norm_factor**2

# print(h_cl_1.shape)
# print(h_cl_2.shape)
# print(h_cm_1.shape)
# print(h_cm_2.shape)




time_test, aoa = plot_aero_responses(h_cl_1[:m], h_cl_1[2*m:], h_cl_1[m:2*m], 
                        h_cm_1[:m], h_cm_1[2*m:], h_cm_1[m:2*m], 
                        omega = 20., nt_int = 5000, dt = 2.5e-4)


results = time_integration(h_cl_1, h_cl_2, h_cm_1, h_cm_2, delta_t = 2.5e-4, m = m, nt_int = 15000, xi = 0.000, v_inf=123., q = 5800., cl_0 = cl_0, cm_0=cm_0)

# time_test, aoa = plot_aero_responses(h_cl_theta, h_cl_zdot, h_cl_thetadot, 
#                         h_cm_theta, h_cm_zdot, h_cm_thetadot, 
#                         omega = 20., nt_int = 1000, dt = dt)



