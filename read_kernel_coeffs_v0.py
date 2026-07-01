import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

suffixes = ["00", "01", "02", "03", "045", "05", "06"]
base_path = "/Users/marcello/Documents/SU2/SC20414/"

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

rho_12_list = []
rho_23_list = []

for suffix in suffixes:
    folder_name = f"AOA{suffix}_R12_test"
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

    rho_12 = np.linalg.norm(data['h_cl_2']) / np.linalg.norm(data['h_cl_1']) * A_ref
    rho_23 = np.linalg.norm(data['h_cl_3']) / np.linalg.norm(data['h_cl_2']) * A_ref

    rho_12_list.append(rho_12)
    rho_23_list.append(rho_23)




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
    
# second series 

m = []
h_cl_1_SA_list = []
h_cl_2_SA_list = []
h_cm_1_SA_list = []
h_cm_2_SA_list = []

for suffix in suffixes:
    folder_name = f"AOA{suffix}_R12_SAopts_test"
    folder_path = os.path.join(base_path, folder_name)
    folder_path_obj = Path(folder_path)

    if os.path.exists(folder_path):
        print(f"Processing: {folder_path}")
        # Insert your data operations here
    else:
        print(f"Warning: {folder_path} not found.")

    filename = folder_path_obj / "kernels_1_and_2.npz"  

    # Load the archive
    data = np.load(filename)

    # Access the individual arrays
    m.append(data['m'])
    cl_0 = data['cl_0']
    cm_0 = data['cm_0']
    h_cl_1_SA_list.append(data['h_cl_1'])
    h_cl_2_SA_list.append(data['h_cl_2'])
    h_cm_1_SA_list.append(data['h_cm_1'])
    h_cm_2_SA_list.append(data['h_cm_2'])
    

for i_aoa in range(n_aoa):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(np.array(h_cl_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
    plt.plot(np.array(h_cl_1_SA_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
    plt.subplot(2, 1, 2)
    plt.plot(np.array(h_cm_1_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
    plt.plot(np.array(h_cm_1_SA_list[i_aoa][:m[i_aoa]]),label = suffixes[i_aoa])
    plt.show()
