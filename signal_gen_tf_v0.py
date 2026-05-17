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


home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA02/" 
filename = home_pitch + "/history_05_00002.dat"    

# mach = 0.8
# chord = 0.4064 #1. 
# v_inf = 123. #mach * 10 / 8 * 277.24 # 123. 

mach = 0.8
v_inf = mach * 10 / 8 * 277.24 # 123. 
chord = 1. 

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

ntsteps = len(cl_output)

target_freqs_pitching = 12.5 / (2.*np.pi) * np.array([1.4048, 2.6437, 5.0113]) 
target_freqs_plunging = 12.5 / (2.*np.pi) * np.array([1.0, 1.9226, 3.6286])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt, ntsteps*dt, amp = 0.50 * np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt, ntsteps*dt, amp = 0.10)

zdot_input = np.gradient(plunge_input,tt_05)/v_inf # SIGN!!!!!
factor = 1. 
thetadot_input = np.gradient(pitch_input, tt_05) /v_inf * factor

# -----
resample_steps = 1 #4 #4 #10
m = 180 #180 #160 #80 #80

decay_rate = 5.125 #0.25  
decay_rate_2 = 0.25  

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
X = np.zeros((ntsteps - m,m,4),dtype = 'f')

aoa1min = np.min(pitch_input)
aoa1max = np.max(pitch_input)
aoa1span = aoa1max - aoa1min 

aoa12min = np.min(zdot_input)
aoa12max = np.max(zdot_input)
aoa12span = aoa12max - aoa12min 

cl1min = np.min(cl_output)
cl1max = np.max(cl_output)
cl1span = cl1max - cl1min 

cl2min = np.min(cm_output)
cl2max = np.max(cm_output)
cl2span = cl2max - cl2min 


for i in range(ntsteps - m):
    X[i,:m,0] = (pitch_input[i:i+m] - aoa1min)/aoa1span
    X[i,:m,1] = (zdot_input[i:i+m] - aoa12min)/aoa12span
    X[i,:m,2] = (cl_output[i:i+m] - cl1min)/cl1span
    X[i,:m,3] = (cm_output[i:i+m] - cl2min)/cl2span

# X=np.stack([aoa1,aoa2], axis=1).astype(np.float32)
print(X.shape)
print(X.dtype)

y1min = np.min(cl_output)
y1max = np.max(cl_output)
y1span = y1max - y1min 

y2min = np.min(cm_output)
y2max = np.max(cm_output)
y2span = y2max - y2min 

Y=np.stack([ (cl_output[m:] - y1min)/y1span, (cm_output[m:] - y2min)/y2span], axis=1).astype(np.float32)

print(Y.shape)
print(Y.dtype)

# save signals 
filename = "/Users/marcello/Documents/Ael/tf_test/test_signals_su2.npy"

with open(filename, 'wb') as f:
    np.save(f, X)
    np.save(f, Y)

fig = plt.figure()
fig.set_size_inches(14.25, 6.25)

ax = fig.add_subplot(211)
plt.plot(Y[:,0],'b-',markersize = 2, label = "Cl")
plt.plot(Y[:,1],'r-',markersize = 2, label = "Cm")


plt.grid(True)
plt.legend()

ax = fig.add_subplot(212)
plt.plot(X[:,0,0],'b-',markersize = 2, label = "INPUT 1")
plt.plot(X[:,0,1],'r-',markersize = 2, label = "INPUT 2")
plt.plot(X[:,0,2],'g-',markersize = 2, label = "INPUT 3")
plt.plot(X[:,0,3],'k--',markersize = 2, label = "INPUT 4")

plt.grid(True)
plt.legend()
plt.show() 