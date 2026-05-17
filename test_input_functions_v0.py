import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.,5.,1000)

k1 = 8.0
k2 = 0.2
omega_1 = 2*np.pi * 3.3
omega_2 = 2*np.pi * 5.7

b1 = 10.30 
b2 = 10.30 
b3 = 0.15 

x1 = 2.0 
x2 = 8.0 

y1 = np.arctan(k1 * np.sin(omega_1 * x + 0*np.pi))
y2 = np.arctan(k2 * np.sin(omega_2 * x + np.pi))
y3 = np.gradient(y2,x) / 10
# y1 = np.exp( -b1 * x**2) #np.tanh(k1 * np.sin(omega_1 * x + 0.5*np.pi))
# y2 = np.exp( -b2 * x**2)
# y3 = np.exp( -b1 * (x - x1)**2) - np.exp( -b2 * (x - x2)**2)#np.tanh(k2 * np.sin(omega_3 * x))
# y1 = 1. - np.exp(-omega_1 * x)
# y3 = 1. - np.exp(-omega_3 * x)
# y3 = y1 + y2int
#y2 = np.gradient(y3,x) / 10
# y2_analytic = omega_3 * np.exp(-omega_3 * x)

y4 = y1 + y3

fft_freqs = np.fft.fftfreq(1000, d = 25./1000)
positive_mask = fft_freqs >= 0
freqs = fft_freqs[positive_mask]
fft_values = np.fft.fft(y4)
Y4 = 2.0/1000 * np.abs(fft_values[positive_mask])


fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plt.subplot(2, 1, 1)

plt.plot(x,y1,label="PITCH")
plt.plot(x,y3,label=r"$dh/dt$")
#plt.plot(x,y2_analytic,label=r"$dy1/dt$")
plt.plot(x,y4,'k-',linewidth = 1, label=r"$\alpha$")
plt.xlabel(r'$\tau\,[\,]$')
plt.ylabel(r'$w\, [\,]$')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(freqs,Y4,'k.-',label="Two-step")

plt.show() 



