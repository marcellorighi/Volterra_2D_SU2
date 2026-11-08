import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CASE = 1
OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE+20}'
# OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE}'

# --- choose the file: either an explicit path, or a pattern you refine ---
# FILE = OUT_DIR + 'your_file_name.csv'
candidates = sorted(glob.glob(OUT_DIR + '**/*', recursive=True))
candidates = [f for f in candidates if f.endswith(('.csv', '.txt', '.dat', '.npz'))]
for i, f in enumerate(candidates):
    print(i, f)
import os

# --- choose what to compare: a list of indices into `candidates`, or paths ---
# FILES = [candidates[1], candidates[2], candidates[3]]
# or a range:  
FILES = candidates[19:24]
COLS = [1, 3]            # columns to plot: z, theta, zdot (thetadot = 3 is left out)
T_RANGE = None              # e.g. (0.0, 2.0) to restrict the time axis, None = all

def load_case(path):
    if path.endswith('.npz'):
        z = np.load(path)
        Y = z['qq'].T                      # -> (nt, ndof)
        t = z['time']
        names = [f"q{i}" for i in range(Y.shape[1])]
    else:
        df = pd.read_csv(path, sep=None, engine='python', comment='#')
        t = df.iloc[:, 0].to_numpy()
        Y = df.iloc[:, 1:].to_numpy()
        names = [str(c) for c in df.columns[1:]]
    return t, Y, names

data = []
for f in FILES:
    t, Y, names = load_case(f)
    print(f"{os.path.basename(f)}: shape {Y.shape}, t {t[0]:.4g} -> {t[-1]:.4g} s, "
          f"dt = {np.median(np.diff(t)):.3e} s, finite: {np.all(np.isfinite(Y))}")
    data.append((os.path.basename(f), t, Y, names))

cols = [c for c in COLS if all(c < d[2].shape[1] for d in data)]
fig, ax = plt.subplots(len(cols), 1, figsize=(10, 2.4 * len(cols)), sharex=True, squeeze=False)
ax = ax[:, 0]
for label, t, Y, names in data:
    sel = slice(None) if T_RANGE is None else (t >= T_RANGE[0]) & (t <= T_RANGE[1])
    for a, j in zip(ax, cols):
        a.plot(t[sel], Y[sel, j], lw=0.8, label=label)
for a, j in zip(ax, cols):
    a.set_ylabel(data[0][3][j])
    a.grid(True)
ax[0].legend(fontsize=7, ncol=2)
ax[-1].set_xlabel('t [s]')
plt.tight_layout()
plt.show()