import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CASE = 1
OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE+10}'
# OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE}'

# --- choose the file: either an explicit path, or a pattern you refine ---
# FILE = OUT_DIR + 'your_file_name.csv'
candidates = sorted(glob.glob(OUT_DIR + '**/*', recursive=True))
candidates = [f for f in candidates if f.endswith(('.csv', '.txt', '.dat', '.npz'))]
for i, f in enumerate(candidates):
    print(i, f)
FILE = candidates[71]          # <-- set the index (or FILE = path) of the case to check
print("\nReading:", FILE)

# --- load ---
if FILE.endswith('.npz'):
    z = np.load(FILE)
    print("keys:", list(z.keys()))
    t = z['time']                          # adapt the key names to what is printed
    Y = z['qq'] if 'qq' in z else None
    names = [f"q{i}" for i in range(Y.shape[0])]
    Y = Y.T                                # -> (nt, ndof)
else:
    df = pd.read_csv(FILE, sep=None, engine='python', comment='#')
    print(df.head())
    print("columns:", list(df.columns))
    t = df.iloc[:, 0].to_numpy()           # assume first column is time
    Y = df.iloc[:, 1:].to_numpy()
    names = [str(c) for c in df.columns[1:]]

# --- basic checks ---
print(f"\nshape: {Y.shape},  t: {t[0]:.4g} -> {t[-1]:.4g} s,  dt = {np.median(np.diff(t)):.3e} s")
print("NaN/inf present:", not np.all(np.isfinite(Y)))
print("max |value| per column:", np.nanmax(np.abs(Y), axis=0))

# --- plot the first few columns (z, theta, zdot, thetadot if they come first) ---
ncol = min(4, Y.shape[1])
fig, ax = plt.subplots(ncol, 1, figsize=(9, 2.2*ncol), sharex=True)
ax = np.atleast_1d(ax)
for j in range(ncol):
    ax[j].plot(t, Y[:, j], lw=0.8)
    ax[j].set_ylabel(names[j])
    ax[j].grid(True)
ax[-1].set_xlabel('t [s]')
fig.suptitle(FILE.split('/')[-1], fontsize=9)
plt.tight_layout()
plt.show()