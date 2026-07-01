import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import signal
import csv 

def read_csv(filename):
    with open(filename, newline='') as f_input:
        next(f_input)
        next(f_input)
        return [list(map(float, row)) for row in csv.reader(f_input)]


mach = 0.8
chord = 1. 

vref = mach * 10 / 8 * 272
tref = chord / (2 * vref)
subs = 48 
cl0 = 0.620363
cm0 = 0.09120
cd0 = 0.10125 


theta0 = 5.0
t0z = 1./25. 
daoa_plunge = theta0*np.pi/180. #np.arctan(10/272)

dt = 2.5e-4 

iCd = 12
iCl = 13
iCm = 17

AOA = 3
MACH = 76
home_pitch = "/Users/marcello/Documents/SU2/bscw/M0%s" % MACH
filename = home_pitch + "/history_05_00002.dat"    

# cl0 = 0.331
# cm0 = 0.0949

print(filename)
data = np.array(read_csv(filename)) 
cd_plunge = (data[0*subs::subs,iCd] - cd0) #/ daoa_plunge
cl_plunge = (data[0*subs::subs,iCl] - cl0) #/ daoa_plunge
cm_plunge = (data[0::subs,iCm] - cm0) #/ daoa_plunge # positive nose up
# it_05 = data[0*subs::subs,0]
# tau_plunge = (it_05-2)*dt/tref
# tt_05 = (it_05-2)*dt

# print(cl_plunge.shape)

# omega = 11. 
# aoa = daoa_plunge * np.sin(omega * tt_05)
# aoa_deg = aoa*180/np.pi
# aoa_deg_m = 5.0 
# aoa_deg_r = 4.0 
# aoa_deg = aoa_deg_m + aoa_deg_r * np.sin(omega * tt_05)

# theta = theta0 * (1. - np.exp(-tt_05/t0z))
# thetadot = theta0 / t0z * np.exp(-tt_05/t0z)

fig, ax = plt.subplots(3,1, figsize=(15, 8.5))

plt.subplot(3, 1, 1)

plt.plot(cl_plunge,'k-',linewidth=1,label='CL')
plt.xlabel(r'$\tau\,[\,]$')        
plt.ylabel(r'$C_l\,[\,]$')     
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)

plt.plot(cm_plunge, 'k-',linewidth=1,label='CM')
plt.xlabel(r'$\tau\,[\,]$')        
plt.ylabel(r'$C_m\,[\,]$')     
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)

plt.plot(cd_plunge, 'k-',linewidth=1,label='CD')
plt.xlabel(r'$\tau\,[\,]$')        
plt.ylabel(r'$C_d\,[\,]$')     
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
