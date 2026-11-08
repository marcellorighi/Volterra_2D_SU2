import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import csv 
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag
from scipy.linalg import lu_factor, lu_solve
from tqdm import tqdm
import os


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

import numpy as np

def build_aeroelastic_matrices(
    Aphys_D, Bphys_D,
    n_dof, m, DT, beta,
    Q_lift, Q_ma, v_inf,
    h_cl_theta, h_cl_zdot, h_cl_thetadot,
    h_cm_theta, h_cm_zdot, h_cm_thetadot,
):
    """
    Build AA, BB for the discrete system  BB @ x_{n+1} = AA @ x_n
    including Volterra (indicial-response) aerodynamic states.

    Parameters
    ----------
    Aphys_D, Bphys_D : (2*n_dof, 2*n_dof) arrays, structural part
    n_dof            : number of structural DOFs (the code below assumes 2)
    m                : number of Volterra kernel samples (each h_* has length m)
    DT, beta         : time step, time-integration weighting (0..1)
    Q_lift, Q_ma     : dynamic-pressure-scaled lift / moment factors
    v_inf            : free-stream velocity
    h_c*_*           : Volterra kernels, 1D arrays of length m

    Returns
    -------
    AA, BB : (N, N) arrays, N = 2*n_dof + 3*(m-1)
    """
    assert n_dof == 2, "indices 2,3 (lift/moment rows) assume n_dof = 2"
    n_phys = 2 * n_dof
    N = n_phys + 3 * (m - 1)

    h_cl_theta    = np.asarray(h_cl_theta)
    h_cl_zdot     = np.asarray(h_cl_zdot)
    h_cl_thetadot = np.asarray(h_cl_thetadot)
    h_cm_theta    = np.asarray(h_cm_theta)
    h_cm_zdot     = np.asarray(h_cm_zdot)
    h_cm_thetadot = np.asarray(h_cm_thetadot)
    for h in (h_cl_theta, h_cl_zdot, h_cl_thetadot,
              h_cm_theta, h_cm_zdot, h_cm_thetadot):
        assert h.shape[0] >= m, "kernels must have at least m samples"

    AA = np.zeros((N, N))
    BB = np.zeros((N, N))

    # structural part
    AA[:n_phys, :n_phys] = Aphys_D
    BB[:n_phys, :n_phys] = Bphys_D
    BB[n_phys:, n_phys:] = np.eye(3 * (m - 1))

    # (row, Q, kernels for theta / zdot / thetadot)
    rows = (
        (2, Q_lift, h_cl_theta, h_cl_zdot, h_cl_thetadot),
        (3, Q_ma,   h_cm_theta, h_cm_zdot, h_cm_thetadot),
    )

    # column blocks of the aerodynamic history states
    c_theta    = n_phys
    c_thetadot = n_phys + (m - 1)
    c_heavedot = n_phys + 2 * (m - 1)

    for r, Q, h_th, h_zd, h_thd in rows:
        # --- time (n+1), goes on BB (note: keeps the sign pattern of the original)
        BB[r, 1] -= DT * Q * h_th[0]  * beta
        BB[r, 2] -= -DT * Q * h_zd[0]  * beta / v_inf
        BB[r, 3] -= DT * Q * h_thd[0] * beta / v_inf

        # --- time (n)
        AA[r, 1] += DT * Q * h_th[1]  * beta
        AA[r, 2] += -DT * Q * h_zd[1]  * beta / v_inf
        AA[r, 3] += DT * Q * h_thd[1] * beta / v_inf

        AA[r, 1] += DT * Q * h_th[0]  * (1 - beta)
        AA[r, 2] += -DT * Q * h_zd[0]  * (1 - beta) / v_inf
        AA[r, 3] += DT * Q * h_thd[0] * (1 - beta) / v_inf

        # --- times (n-1 -> n-m+1): beta part, m-2 samples
        n1 = m - 2
        AA[r, c_theta:c_theta + n1]       = DT * Q * h_th[2:m]  * beta
        AA[r, c_heavedot:c_heavedot + n1] = DT * Q * h_zd[2:m]  * beta
        AA[r, c_thetadot:c_thetadot + n1] = DT * Q * h_thd[2:m] * beta

        # --- times (n-1 -> n-m): (1-beta) part, m-1 samples
        n2 = m - 1
        AA[r, c_theta:c_theta + n2]       += DT * Q * h_th[1:m]  * (1 - beta)
        AA[r, c_heavedot:c_heavedot + n2] += DT * Q * h_zd[1:m]  * (1 - beta)
        AA[r, c_thetadot:c_thetadot + n2] += DT * Q * h_thd[1:m] * (1 - beta)

    # --- shift registers for the aerodynamic states
    for i_dof in range(3):
        irow1 = n_phys + 1 + i_dof * (m - 1)
        irow2 = irow1 + m - 2
        icol1 = irow1 - 1
        icol2 = irow2 - 1
        AA[irow1:irow2, icol1:icol2] = np.eye(m - 2)

    # --- feed new values into the first slot of each register
    AA[n_phys, 1]                 = 1.
    AA[n_phys + (m - 1), 3]       = 1. / v_inf   # theta_dot
    AA[n_phys + 2 * (m - 1), 2]   = -1. / v_inf  # z_dot

    return AA, BB

def time_integration(theta_cl, theta_cl2, h_cl_3, h_cl_4, theta_cm, theta_cm2, h_cm_3, h_cm_4, delta_t, m, nt_int, xi, 
                     v_inf, q, cl_0, cm_0):
    
    n_dof = 2 
    ndof_aero = 3*(m-1) 
    ndof = ndof_aero + 2 * n_dof 

    # mass= 87; 
    # Iy= 3.765; 
    # kz=MASS*(3.33*2*np.pi)**2; 
    # ktheta=IY*(5.20*2*np.pi)**2; 

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


    K_aa = np.array([[KZ, 0],[0, KTHETA]])
    M_aa = np.array([[MASS, 0],[0,IY]])
    C_aa = 2*xi* np.diag([np.sqrt(MASS*KZ),np.sqrt(IY*KTHETA) ])
    C_aa_critical = 2* np.diag([np.sqrt(MASS*KZ),np.sqrt(IY*KTHETA) ])

    AA = np.zeros((ndof,ndof))
    BB = np.zeros((ndof,ndof))

    Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
    Aphys_critical = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa_critical])])
    Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

    Aphys_D = Bphys + DT*(1.-beta)*Aphys
    Bphys_D = Bphys - DT*beta*Aphys 
    Aphys_crit_D = Bphys + DT*(1.-beta)*Aphys_critical
    Bphys_crit_D = Bphys - DT*beta*Aphys_critical 

    AA_crit, BB_crit = build_aeroelastic_matrices(
        Aphys_crit_D, Bphys_crit_D, n_dof=2, m=m, DT=DT, beta=beta,
        Q_lift=Q_lift, Q_ma=Q_ma, v_inf=v_inf,
        h_cl_theta=h_cl_theta, h_cl_zdot=h_cl_zdot, h_cl_thetadot=h_cl_thetadot,
        h_cm_theta=h_cm_theta, h_cm_zdot=h_cm_zdot, h_cm_thetadot=h_cm_thetadot,
    )

    AA, BB = build_aeroelastic_matrices(
        Aphys_D, Bphys_D, n_dof=2, m=m, DT=DT, beta=beta,
        Q_lift=Q_lift, Q_ma=Q_ma, v_inf=v_inf,
        h_cl_theta=h_cl_theta, h_cl_zdot=h_cl_zdot, h_cl_thetadot=h_cl_thetadot,
        h_cm_theta=h_cm_theta, h_cm_zdot=h_cm_zdot, h_cm_thetadot=h_cm_thetadot,
    )

    tau_int = np.arange(nt_int)*dtau #np.linspace(0,nt_int*dtau,nt_int)
    time_int = np.arange(nt_int)*DT #np.linspace(0,nt_int*DT,nt_int)

    qq = np.zeros((ndof,nt_int))

    # qping = 0.1 # heave 
    # qq[2,0] = 0*qping # PING
    # qq[3,0] = qping # PING

    fNL = np.zeros(ndof)

    BB_1 = linalg.inv(BB)

    lu, piv = lu_factor(BB)  
    lu_crit, piv_crit = lu_factor(BB_crit)  

    f_0 = np.zeros(4 + 3*(m-1))
    f_0[2] = Q_lift * cl_0 
    f_0[3] = Q_ma * cm_0

    theta_lags = np.zeros(m)
    thetadot_lags = np.zeros(m)
    zdot_lags = np.zeros(m)

    n_subit = 2

    # linear stability analysis 
    analyze_discrete_instability(BB_1@AA,DT)

    beta = 0.5 

    for it in tqdm(range(1,nt_int)): #range(1,nt_int):
        # linear predictor
        if (it < N_CRIT_DAMPING):
            qq[:,it] = lu_solve((lu_crit, piv_crit), AA_crit@qq[:,it-1]) 
            # qq[0,it] = 0. 
            # qq[2,it] = 0.
        else:     
            qq[:,it] = lu_solve((lu, piv), AA@qq[:,it-1]) 
            # qq[0,it] = 0. 
            # qq[2,it] = 0.

        for sub_it in range(n_subit):

            theta_lags[0] = (1-beta)*qq[1,it-1] + beta*qq[1,it]
            thetadot_lags[0] = (1-beta)*qq[3,it-1]/v_inf + beta*qq[3,it]/v_inf 
            zdot_lags[0] = -(1-beta)*qq[2,it-1]/v_inf - beta*qq[2,it]/v_inf 

            theta_lags[1:] = (1-beta)*qq[4:4+m-1,it-1] + beta*qq[4:4+m-1,it]
            thetadot_lags[1:] = (1-beta)*qq[4 + m-1: 4+(m-1)*2,it-1] + beta*qq[4 + m-1: 4+(m-1)*2,it]
            zdot_lags[1:] = (1-beta)*qq[4 + 2*(m-1):,it-1] + beta*qq[4 + 2*(m-1):,it]

            # 2nd order kernel
            all_lags_1 = np.concatenate([np.abs(theta_lags), np.abs(zdot_lags), theta_lags])
            # all_lags_1 = np.concatenate([theta_lags, zdot_lags, theta_lags])
            all_lags_2 = np.concatenate([theta_lags, zdot_lags, zdot_lags])

            cl_NL = np.dot(theta_cl2, all_lags_1*all_lags_2)
            cm_NL = np.dot(theta_cm2, all_lags_1*all_lags_2)

            # 3rd order kernel
            all_lags_1 = np.concatenate([theta_lags**2, zdot_lags**2, theta_lags, theta_lags**2])
            all_lags_2 = np.concatenate([theta_lags, zdot_lags, zdot_lags**2, zdot_lags])

            cl_NL += np.dot(h_cl_3, all_lags_1*all_lags_2)
            cm_NL += np.dot(h_cm_3, all_lags_1*all_lags_2)

            # 4th order kernel A_pitch4, A_zdot4, A_pitch2_zdot2, A_pitch3_zdot, A_pitch_zdot3
            all_lags_1 = np.concatenate([theta_lags**3, zdot_lags**3, theta_lags**2, theta_lags**3, theta_lags])
            all_lags_2 = np.concatenate([theta_lags,    zdot_lags,    zdot_lags**2,  zdot_lags,     zdot_lags**3])

            cl_NL += np.dot(h_cl_4, all_lags_1*all_lags_2)
            cm_NL += np.dot(h_cm_4, all_lags_1*all_lags_2)

            f_0[2] = Q_lift * (cl_0 + cl_NL)
            f_0[3] = Q_ma * (cm_0 + cm_NL)

            if (it < N_CRIT_DAMPING):
                qq[:,it] = lu_solve((lu_crit, piv_crit), AA_crit@qq[:,it-1] + DT*f_0) 
                # qq[0,it] = 0. 
                # qq[2,it] = 0.
            else:     
                qq[:,it] = lu_solve((lu, piv), AA@qq[:,it-1] + DT*f_0) 
                # qq[0,it] = 0. 
                # qq[2,it] = 0.



    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    plt.subplot(2, 1, 1)

    plt.plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$\theta$")
    plt.xlabel(r'$t\,[s]$')
    plt.ylabel(r'$z\, [m]$')
    # plt.legend()
    # plt.ylim([-0.02,0.12])
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time_int,AOA + qq[1,:]*180/np.pi,'k-',linewidth=2,label=r"$\theta$")
    plt.xlabel(r'$t\,[s]$')
    plt.ylabel(r'$\theta\, [DEG]$')
    # plt.legend()
    # plt.ylim([-0.25, 1.])
    plt.grid(True)

    plt.tight_layout() 
    plt.show()

    return time_int, qq



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

def write_modal_data(time, qq, Team, Dataset, Case, Mach, AoA, Q, Mesh, Dt, output_folder):
    """
    Write modal displacement and velocity data to a formatted text file.

    Parameters
    ----------
    time          : np.ndarray, shape (nt_steps,)
    qq            : np.ndarray, shape (n_dofs, nt_steps)
                    qq[0,:] = mode1 displacement
                    qq[1,:] = mode2 displacement
                    qq[2,:] = mode1 velocity
                    qq[3,:] = mode2 velocity
    Team          : str,   e.g. 'ZHAW'
    Dataset       : int,   dataset index
    Case          : int,   case index
    Mach          : float, Mach number (rounded to 2 decimals in filename)
    AoA           : float, angle of attack (rounded to 1 decimal in filename)
    Q             : float, dynamic pressure (rounded to nearest integer in filename)
    Mesh          : str,   'Coarse', 'Medium', or 'Fine'
    Dt            : float, time step in seconds
    output_folder : str,   path to output directory
    """

    # Build filename
    filename = (
        f'{Team}_'
        f'Dataset{Dataset:d}_'
        f'Case{Case:d}_'
        f'Mach{Mach:.2f}_'
        f'AoA{AoA:.1f}_'
        f'Q{round(Q):d}_'
        f'{Mesh}_'
        f'Dt{Dt}.dat'
    )

    # Full path
    filepath = os.path.join(output_folder, filename)

    # Assemble columns: shape (nt_steps, 5)
    data = np.column_stack([
        time,
        qq[0, :],   # mode1 displacement
        qq[2, :],   # mode1 velocity
        qq[1, :],   # mode2 displacement
        qq[3, :],   # mode2 velocity
    ])

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Write to file
    np.savetxt(
        filepath,
        data,
        header='time          plunge [m]    d/dt(heave)   pitch [RAD]   d/dt(pitch)',
        fmt='%.6e',
        delimiter='  ',
        comments=''
    )

    print(f'Written: {filepath}')
    return filepath

"""
Case 1: 3D wing flutter prediction at Mach 0.80 and angle-of-attack sweep: 0° - 6°
Case 2: 3D wing flutter prediction at Mach 0.74, 0.76, 0.78 and angle-of-attack 3°
Case 3: 2D wing flutter prediction at Mach 0.80 and angle-of-attack sweep: 0° - 6°
"""
AOA = 5
MACH = 80
CASE = 1

PSF = 47.8803
MASS = 87*1.0; 
IY = 3.765*1; 
F_HEAVE = 3.33
F_PITCH = 5.20 # 5.20
KZ =MASS*(F_HEAVE*2*np.pi)**2; 
KTHETA =IY*(F_PITCH*2*np.pi)**2; 

N_CRIT_DAMPING = 500

# first order kernel 
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test_2/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_test/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_test/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_3/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_verylarge/" % AOA
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/M0%s_veryfine/" % MACH
# home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_SAopts_veryfine/" % AOA
# dt = 2.5e-4
# V_freestream = 123.*MACH/80

# Michael 
# home_pitch = "/Users/marcello/Documents/Ael/bscw/Data_from_Michael/AOA0%s" % AOA
home_pitch = "/Users/marcello/Documents/Ael/bscw/Data_from_Michael/AOA0%s_fine" % AOA
# MESH = "Medium"
# DATASET = 1
DATASET = 4
MESH = "Fine"
dt = 2.0e-4
V_freestream = 131.41

filename = home_pitch + "/kernels_1_to_4.npz"    

# Load the archive
data = np.load(filename)

# Access the individual arrays
m = data['m']
cl_0 = data['cl_0']
cm_0 = data['cm_0']
h_cl_1 = data['h_cl_1']
h_cm_1 = data['h_cm_1']
h_cl_2 = data['h_cl_2']
h_cm_2 = data['h_cm_2']
h_cl_3 = data['h_cl_3']
h_cm_3 = data['h_cm_3']
h_cl_4 = data['h_cl_4']
h_cm_4 = data['h_cm_4']

#print(h_cl_1)

print("Cl0 = ",cl_0)
print("Cm0 = ",cm_0)

print("V freestream", V_freestream)
# ================

print(h_cl_1.shape)
print(h_cl_2.shape)
print(h_cl_3.shape)

Q_test = 105*PSF
Q_psf = Q_test / PSF

print("Cl_0 and Cm_0",cl_0,cm_0)
time_int, qq = time_integration(h_cl_1, h_cl_2, h_cl_3, h_cl_4, h_cm_1, h_cm_2, h_cm_3, h_cm_4, delta_t = dt, m = m, nt_int =28000, xi = 0.000, v_inf=V_freestream, q = Q_test, cl_0 = cl_0, cm_0=cm_0)

OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE+20}'


# fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# plt.subplot(2, 1, 1)

# plt.plot(time_int,qq[0,:],'k-',linewidth=2,label=r"$\theta$")
# plt.xlabel(r'$t\,[s]$')
# plt.ylabel(r'$z\, [m]$')
# # plt.legend()
# # plt.ylim([-0.02,0.12])
# plt.grid(True)

# plt.subplot(2, 1, 2)

# plt.plot(time_int,qq[1,:]*180/np.pi,'k-',linewidth=2,label=r"$\theta$")
# plt.xlabel(r'$t\,[s]$')
# plt.ylabel(r'$\theta\, [DEG]$')
# # plt.legend()
# # plt.ylim([-0.25, 1.])
# plt.grid(True)

# plt.tight_layout() 
# plt.show()


# write_modal_data(time=time_int, qq=qq, Team='ZHAW', Dataset=DATASET, Case=CASE,
#                         Mach=MACH/100, AoA=AOA, Q=Q_psf, Mesh=MESH, Dt=dt,
#                         output_folder=OUT_DIR)
