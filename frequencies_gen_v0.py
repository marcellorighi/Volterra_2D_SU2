import numpy as np
import matplotlib.pyplot as plt

def make_incommensurate(freqs, perturbation=0.05):
    """
    Add small perturbation to break harmonic relationships
    
    perturbation: fraction of frequency spacing
    """
    df = np.diff(freqs).mean()
    
    # Add random perturbation (but deterministic for repeatability)
    np.random.seed(42)
    perturb = perturbation * df * (np.random.rand(len(freqs)) - 0.5)
    
    freqs_incomm = freqs + perturb
    
    # Verify no exact harmonic relationships
    for i, f1 in enumerate(freqs_incomm):
        for j, f2 in enumerate(freqs_incomm[i+1:], i+1):
            ratio = f2 / f1
            if abs(ratio - round(ratio)) < 0.01:
                print(f"Warning: {f1:.2f} and {f2:.2f} are nearly harmonic!")
    
    return freqs_incomm

def design_frequency_grid(f_min, f_max, n_freqs, spacing='logarithmic'):
    """
    Design frequency grid for Volterra identification
    
    Parameters:
    -----------
    f_min, f_max : float
        Frequency band of interest (Hz)
    n_freqs : int
        Number of frequency components
    spacing : str
        'logarithmic' (better for wide bands) or 'linear'
    """
    if spacing == 'logarithmic':
        # Logarithmic spacing: better resolution at low frequencies
        # Common for wide-band systems
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    else:
        # Linear spacing: uniform resolution
        freqs = np.linspace(f_min, f_max, n_freqs)
    
    return freqs

# For your case: 3-10 Hz
f_min, f_max = 2.0, 10.0

# Number of frequencies (discussed below)
n_freqs = 6  # Good starting point

#freqs = design_frequency_grid(f_min, f_max, n_freqs, spacing='linear')
freqs = design_frequency_grid(f_min, f_max, n_freqs, spacing='logarithmic')

freqs_2 = make_incommensurate(freqs, perturbation=0.05)

print(f"Training frequencies: {freqs}")
# Example output: [3.0, 4.17, 5.33, 6.5, 7.67, 8.83, 10.0]

print(f"Training frequencies: {freqs_2}")

print(f"Training frequencies: {freqs_2 / freqs_2[0]}")

