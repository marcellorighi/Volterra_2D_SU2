import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from volterra_nn_v0 import (
    VolterraNN, solve_full_regularized, plot_time_series_comparison,
    plot_parity, plot_nmse_bar, read_csv, generate_schroeder_multisine,
    build_lagged_windows, per_output_nmse, predict,
)
import numpy as np
import matplotlib.pyplot as plt

# device 
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

AOA = 5
home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine" % AOA
save_path = home_pitch +"/volterra_nn_checkpoint.pt"
checkpoint = torch.load(save_path, weights_only=False)

model = VolterraNN(
    n_inputs=checkpoint['n_inputs'],
    output_names=checkpoint['output_names'],
    channels=checkpoint['channels'],
    kernel_size=checkpoint['kernel_size'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device) 
model.eval()   # important: turns off dropout for evaluation

m = checkpoint['m']   # use this m when calling build_lagged_windows on the new signal


# ------------------------------------------------------------------
# Data: signals + targets
# ------------------------------------------------------------------
    
AOA = 5
MACH = 80
dt = 2.5e-4

chord = 0.4064 #1. 
v_inf = 123. * MACH / 80 # * 277.24 # 123. 

home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2" % AOA

filename = home_pitch + "/history_05_00002.dat"    

chord = 0.4064 #1. 
v_inf = 123. * MACH / 80 # * 277.24 # 123. 
print("V freestream", v_inf)

tref = chord / (2 * v_inf)
subs = 48 

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

ntsteps = len(cl_output)

target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.8849]) 
target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226])

# target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
# target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt,
                                                ntsteps*dt, amp=0.95*np.pi/180.)
t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt,
                                                ntsteps*dt, amp=0.065)
zdot_input = np.gradient(plunge_input, t) / v_inf

cl = cl_output - np.mean(cl_output)
cm = cm_output - np.mean(cm_output)

m = 72                              # SAME memory depth as your Volterra fit

A_pitch = np.zeros((ntsteps,m))
A_zdot = np.zeros((ntsteps,m))

for icolumn in range(m):
    A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
    A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

A_123 = np.hstack( (A_pitch, A_zdot))

X = build_lagged_windows([pitch_input, zdot_input], m)   # (n_time, m, 2)
y = np.stack([cl, cm], axis=1)                            # (n_time, 2)

# ------------------------------------------------------------------
# Optional: residual-on-Volterra hybrid
# ------------------------------------------------------------------
# from regularized_mode_solve import solve_full_regularized
h_cl = solve_full_regularized(A_123, cl, m, lambda_ridge=1e-3,
                                lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
h_cm = solve_full_regularized(A_123, cm, m, lambda_ridge=1e-3,
                                lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
baseline = np.stack([A_123 @ h_cl, A_123 @ h_cm], axis=1)   # (n_time, 2)

plt.figure(figsize=(12, 9))

plt.subplot(2,1,1)

plt.plot(A_123@ h_cl, 'b-', linewidth=1,
            label='Volterra')

plt.plot(cl, 'k-', linewidth=1,
            label='Reference')

plt.xlabel('t')
# plt.ylabel(f'q_{i+1}')
plt.grid(True)

plt.subplot(2,1,2)

plt.plot(A_123@ h_cm, 'b-', linewidth=1,
            label='Volterra')

plt.plot(cm, 'k-', linewidth=1,
            label='Reference')

plt.xlabel('t')
# plt.ylabel(f'q_{i+1}')
plt.grid(True)

plt.show()


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
pred = predict(model, X, device=device)
# if using hybrid: 
pred = pred + baseline
print(per_output_nmse(pred, y))

output_names = ['cl', 'cm']
# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
plot_time_series_comparison(pred, y, t, output_names, baseline=baseline)
plot_parity(pred, y, output_names)

nn_nmse = dict(zip(output_names, per_output_nmse(pred, y)))
linear_pred = baseline  # A @ h, no NN correction
linear_nmse = dict(zip(output_names, per_output_nmse(linear_pred, y)))

plot_nmse_bar({'linear': linear_nmse, 'nn_hybrid': nn_nmse})

plt.show()