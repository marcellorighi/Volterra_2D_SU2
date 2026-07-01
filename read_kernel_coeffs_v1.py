import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_aero_responses(h_cl_theta, h_cl_zdot, 
                        h_cm_theta, h_cm_zdot, 
                        h_cl_theta_2, h_cl_zdot_2, h_cl_theta_zdot, 
                        h_cm_theta_2, h_cm_zdot_2, h_cm_theta_zdot, 
                        h_cl_theta_3, h_cl_zdot_3, h_cl_theta_zdot2, h_cl_theta2_zdot, 
                        h_cm_theta_3, h_cm_zdot_3, h_cm_theta_zdot2, h_cm_theta2_zdot, 
                        Aref, omega, nt_int, dt):
    """
    Calculates and plots the aerodynamic lift (Cl) and moment (Cm) responses 
    for pitch, heave, and theta-dot components based on a sinusoidal AoA.
    """
    
    # --- 1. Define Time and Input Signal ---
    time_test = dt * np.arange(nt_int)
    # Sinusoidal Angle of Attack (1 degree amplitude)
    aoa = Aref * np.deg2rad(1.) * np.sin(omega * time_test)
    pitch_input = aoa * 1.0 
    zdot_input = aoa * 1.0 

    # --- 2. Calculate Responses ---
    cl, cl_nl = response_2dofs(h_cl_theta, h_cl_zdot, h_cl_theta_2, h_cl_zdot_2, h_cl_theta_zdot, h_cl_theta_3, h_cl_zdot_3, h_cl_theta_zdot2, h_cl_theta2_zdot, pitch_input,zdot_input, nt_int)
    cm, cm_nl = response_2dofs(h_cm_theta, h_cm_zdot, h_cm_theta_2, h_cm_zdot_2, h_cm_theta_zdot, h_cm_theta_3, h_cm_zdot_3, h_cm_theta_zdot2, h_cm_theta2_zdot, pitch_input,zdot_input, nt_int)

    # --- 3. Visualization ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Subplot 1: Input AoA
    ax[0].plot(time_test, aoa, label="AOA")
    ax[0].set_xlabel(r'$\tau\,[\,]$')
    ax[0].set_ylabel(r'$w\, [\,]$')
    ax[0].legend()
    ax[0].grid(True)

    # Subplot 2: Cl Components
    ax[1].plot(time_test, cl, label="Cl L")
    ax[1].plot(time_test, cl_nl, label="Cl NL")
    ax[1].set_xlabel(r'$\tau\,[\,]$')
    ax[1].set_ylabel(r'$Cl\, [\,]$')
    ax[1].legend()
    ax[1].grid(True)

    # Subplot 3: Cm Components
    ax[2].plot(time_test, cm, label="Cm L")
    ax[2].plot(time_test, cm_nl, label="Cm NL")
    ax[2].set_xlabel(r'$\tau\,[\,]$')
    ax[2].set_ylabel(r'$Cm\, [\,]$')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Optional: Return data for further analysis
    return time_test, aoa


def response_2dofs(h_theta, h_zdot, h_theta2, h_zdot2, h_thetazdot, h_theta3, h_zdot3, h_theta2zdot, h_thetazdot2, pitch_input,zdot_input, nt):
    m = h_theta.shape[0] 
    
    AA = np.diag(np.ones(m-1),-1)

    q = np.zeros((2*m, nt))
    q_pitch = np.zeros((m, nt))
    q_zdot = np.zeros((m, nt))
    # q_thetadot = np.zeros((m, nt))
    BB_m = np.zeros(m)
    BB_m[0] = 1.0

    cn = np.zeros(nt)
    cn_nl = np.zeros(nt)
    
    for it in range(1,nt):
        # q[:,it] = AA_2dofs@q[:,it-1] + BB*aoa[it] 
        # cn[it] = CC@q[:,it]    
        q_pitch[:,it] = AA @ q_pitch[:,it-1] + BB_m*pitch_input[it] 
        q_zdot[:,it] = AA @ q_zdot[:,it-1] + BB_m*zdot_input[it] 

        # 1st order kernels
        cn[it] = h_theta @ q_pitch[:,it] 
        cn[it] += h_zdot @ q_zdot[:,it] 

        # 2nd order kernels
        cn_nl[it] =  cn[it] + h_theta2 @ (np.abs((q_pitch[:,it]))*q_pitch[:,it]) + h_zdot2 @ (np.abs((q_zdot[:,it]))*q_zdot[:,it]) + h_thetazdot @ (q_pitch[:,it] * q_zdot[:,it])   

        # 3rd order kernels
        cn_nl[it] +=  h_theta3 @ q_pitch[:,it]**3 + h_zdot3 @ q_zdot[:,it]**3 + h_theta2zdot @ (q_pitch[:,it]**2 * q_zdot[:,it]) +  h_thetazdot2 @ (q_pitch[:,it] * q_zdot[:,it]**2) 

    return cn, cn_nl 


# suffixes = ["00", "01", "02", "03", "045", "05", "06"]
# suffixes = ["03", "04", "045"]
suffixes = ["00", "01", "02",  "03", "04", "05", "06"]
suffixes = ["00", "01", "02", "05", "06"]
#suffixes = ["74", "76", "78", "80"]
numeric_array = np.array(suffixes, dtype=int)

base_path = "/Users/marcello/Documents/SU2/SC20414/"
#base_path = "/Users/marcello/Documents/SU2/bscw/"

n_aoa = len(suffixes)

# convergence check 
A_ref = np.deg2rad(1.)

m = []
h_cl_1_list = []
h_cl_2_list = []
h_cl_3_list = []
h_cm_1_list = []
h_cm_2_list = []
h_cm_3_list = []

rho_12_cl_list = []
rho_23_cl_list = []

rho_12_cm_list = []
rho_23_cm_list = []

for suffix in suffixes:
    # folder_name = f"AOA{suffix}_R12_test"
    # folder_name = f"AOA{suffix}_R12_veryfine_2"
    folder_name = f"AOA{suffix}_R12_SAopts_veryfine"
    # folder_name = f"AOA{suffix}_R12_veryfine_verylarge"
    # folder_name = f"M0{suffix}"
    folder_path = os.path.join(base_path, folder_name)
    folder_path_obj = Path(folder_path)

    if os.path.exists(folder_path):
        print(f"Processing: {folder_path}")
        # Insert your data operations here
    else:
        print(f"Warning: {folder_path} not found.")

    # filename = folder_path_obj / "kernels_1_and_2.npz"  
    filename = folder_path_obj / "kernels_1_to_3.npz"  

    # Load the archive
    data = np.load(filename)

    # Access the individual arrays
    m.append(data['m'])
    cl_0 = data['cl_0']
    cm_0 = data['cm_0']
    h_cl_1_list.append(data['h_cl_1'])
    h_cl_2_list.append(data['h_cl_2'])
    h_cl_3_list.append(data['h_cl_3'])
    h_cm_1_list.append(data['h_cm_1'])
    h_cm_2_list.append(data['h_cm_2'])
    h_cm_3_list.append(data['h_cm_3'])

    rho_12 = 2/3 * np.linalg.norm(data['h_cl_2']) / np.linalg.norm(data['h_cl_1']) * A_ref
    rho_23 = 4/3 * np.linalg.norm(data['h_cl_3']) / np.linalg.norm(data['h_cl_2']) * A_ref

    rho_12_cl_list.append(rho_12)
    rho_23_cl_list.append(rho_23)

    rho_12 = 2/3 * np.linalg.norm(data['h_cm_2']) / np.linalg.norm(data['h_cm_1']) * A_ref
    rho_23 = 4/3 * np.linalg.norm(data['h_cm_3']) / np.linalg.norm(data['h_cm_2']) * A_ref

    rho_12_cm_list.append(rho_12)
    rho_23_cm_list.append(rho_23)



fig, ax = plt.subplots(2, 2, figsize=(10, 8))

plt.subplot(2, 2, 1)

for i_aoa in range(n_aoa):
    plt.plot(np.array(h_cl_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_l}_1\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)

for i_aoa in range(n_aoa):
    plt.plot(np.array(h_cl_2_list[i_aoa][0*m[i_aoa]:m[i_aoa]]),label = suffixes[i_aoa])
    # plt.plot(np.array(h_cl_2_list[i_aoa][m[i_aoa]:2*m[i_aoa]]),label = suffixes[i_aoa])

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_l}_2\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)

for i_aoa in range(n_aoa):
    plt.plot(np.array(h_cl_1_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]]),label = suffixes[i_aoa])

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_m}_1\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)

for i_aoa in range(n_aoa):
    plt.plot(np.array(h_cm_2_list[i_aoa][0*m[i_aoa]:m[i_aoa]]),label = suffixes[i_aoa])
    # plt.plot(np.array(h_cm_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_m}_2\, [\,]$')
plt.legend()
plt.grid(True)



# plt.subplot(2, 3, 5)

# for i_aoa in range(n_aoa):
#     plt.plot(np.array(h_cm_1_list[i_aoa][m[i_aoa]:2*m[i_aoa]]),label = suffixes[i_aoa])

# plt.xlabel(r'$\tau\,[\,]$')
# plt.ylabel(r'$w\, [\,]$')
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 3, 6)

# for i_aoa in range(n_aoa):
#     plt.plot(np.array(h_cm_1_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]]),label = suffixes[i_aoa])

# plt.xlabel(r'$\tau\,[\,]$')
# plt.ylabel(r'$w\, [\,]$')
# plt.legend()
# plt.grid(True)

plt.tight_layout()
plt.show()  
    

Aref = 12.
omega = 2*np.pi * 25. 
nt_int = 5000 
dt = 2.5e-4

i_aoa = 0

h_cl_theta = np.array(h_cl_1_list[i_aoa][:m[i_aoa]])
h_cl_zdot = np.array(h_cl_1_list[i_aoa][2*m[i_aoa]:])

h_cm_theta = np.array(h_cm_1_list[i_aoa][:m[i_aoa]]) 
h_cm_zdot = np.array(h_cm_1_list[i_aoa][2*m[i_aoa]:]) 

h_cl_theta_2 = np.array(h_cl_2_list[i_aoa][0*m[i_aoa]:m[i_aoa]]) 
h_cl_zdot_2 = np.array(h_cl_2_list[i_aoa][1*m[i_aoa]:2*m[i_aoa]]) 
h_cl_theta_zdot = np.array(h_cl_2_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]]) 

h_cm_theta_2 = np.array(h_cm_2_list[i_aoa][0*m[i_aoa]:m[i_aoa]]) 
h_cm_zdot_2  = np.array(h_cm_2_list[i_aoa][1*m[i_aoa]:2*m[i_aoa]]) 
h_cm_theta_zdot  = np.array(h_cm_2_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]])  

h_cl_theta_3 = np.array(h_cl_3_list[i_aoa][0*m[i_aoa]:m[i_aoa]])  
h_cl_zdot_3 = np.array(h_cl_3_list[i_aoa][1*m[i_aoa]:2*m[i_aoa]])   
h_cl_theta_zdot2 = np.array(h_cl_3_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]])  
h_cl_theta2_zdot = np.array(h_cl_3_list[i_aoa][3*m[i_aoa]:4*m[i_aoa]])   

h_cm_theta_3 = np.array(h_cm_3_list[i_aoa][0*m[i_aoa]:m[i_aoa]])   
h_cm_zdot_3  = np.array(h_cm_3_list[i_aoa][1*m[i_aoa]:2*m[i_aoa]])  
h_cm_theta_zdot2 = np.array(h_cm_3_list[i_aoa][2*m[i_aoa]:3*m[i_aoa]])   
h_cm_theta2_zdot = np.array(h_cm_3_list[i_aoa][3*m[i_aoa]:4*m[i_aoa]])   

time_test, aoa = plot_aero_responses(h_cl_theta, h_cl_zdot, 
                        h_cm_theta, h_cm_zdot, 
                        h_cl_theta_2, h_cl_zdot_2, 0*h_cl_theta_zdot, 
                        h_cm_theta_2, h_cm_zdot_2, 0*h_cm_theta_zdot, 
                        h_cl_theta_3, h_cl_zdot_3, h_cl_theta_zdot2, h_cl_theta2_zdot, 
                        h_cm_theta_3, h_cm_zdot_3, h_cm_theta_zdot2, h_cm_theta2_zdot, 
                        Aref, omega, nt_int, dt)




for i_aoa in range(n_aoa):
    print("Cl",i_aoa, rho_12_cl_list[i_aoa])
    print("Cl",i_aoa, rho_23_cl_list[i_aoa])
    print("Cm",i_aoa, rho_12_cm_list[i_aoa])
    print("Cm",i_aoa, rho_23_cm_list[i_aoa])



fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plt.subplot(2, 1, 1)

for i_aoa in range(n_aoa):
    plt.plot(numeric_array[i_aoa],rho_12_cl_list[i_aoa],'ko')
    plt.plot(numeric_array[i_aoa],rho_12_cm_list[i_aoa],'k*')

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_l}_1\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

for i_aoa in range(n_aoa):
    plt.plot(numeric_array[i_aoa],rho_23_cl_list[i_aoa],'ko')
    plt.plot(numeric_array[i_aoa],rho_23_cm_list[i_aoa],'k*')

plt.xlabel(r'$\,[\,]$')
plt.ylabel(r'$h^{C_l}_1\, [\,]$')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()  
    
# second series 

# m = []
# h_cl_1_SA_list = []
# h_cl_2_SA_list = []
# h_cm_1_SA_list = []
# h_cm_2_SA_list = []

# for suffix in suffixes:
#     folder_name = f"AOA{suffix}_R12_SAopts_test"
#     folder_path = os.path.join(base_path, folder_name)
#     folder_path_obj = Path(folder_path)

#     if os.path.exists(folder_path):
#         print(f"Processing: {folder_path}")
#         # Insert your data operations here
#     else:
#         print(f"Warning: {folder_path} not found.")

#     filename = folder_path_obj / "kernels_1_and_2.npz"  

#     # Load the archive
#     data = np.load(filename)

#     # Access the individual arrays
#     m.append(data['m'])
#     cl_0 = data['cl_0']
#     cm_0 = data['cm_0']
#     h_cl_1_SA_list.append(data['h_cl_1'])
#     h_cl_2_SA_list.append(data['h_cl_2'])
#     h_cm_1_SA_list.append(data['h_cm_1'])
#     h_cm_2_SA_list.append(data['h_cm_2'])
    

# for i_aoa in range(n_aoa):
#     fig, ax = plt.subplots(2, 1, figsize=(10, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(np.array(h_cl_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
#     plt.plot(np.array(h_cl_1_SA_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
#     plt.subplot(2, 1, 2)
#     plt.plot(np.array(h_cm_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
#     plt.plot(np.array(h_cm_1_SA_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
#     plt.show()
