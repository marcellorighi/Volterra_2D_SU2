import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

CASE = 1
OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE+20}'
# OUT_DIR = f'/Users/marcello/Documents/SU2/bscw/AePW4_Data/Case_{CASE}'

# --- choose the file: either an explicit path, or a pattern you refine ---
# FILE = OUT_DIR + 'your_file_name.csv'
candidates = sorted(glob.glob(OUT_DIR + '**/*', recursive=True))
candidates = [f for f in candidates if f.endswith(('.csv', '.txt', '.dat', '.npz'))]
for i, f in enumerate(candidates):
    print(i, f)

ap = argparse.ArgumentParser(description="Plot the response of selected cases.")
ap.add_argument("--idx", type=int, nargs="+",
                help="explicit indices into candidates, e.g. --idx 71 75 80")
ap.add_argument("--range", type=int, nargs=2, metavar=("START", "STOP"),
                help="python-style slice of candidates (STOP excluded), e.g. --range 71 75")
ap.add_argument("--files", nargs="+", help="explicit paths (instead of indices)")
ap.add_argument("--cols", type=int, nargs="+", default=[1, 3],
                help="column indices of Y to plot (default: 0 2)")
ap.add_argument("--trange", type=float, nargs=2, metavar=("T0", "T1"),
                help="restrict the time axis")
ap.add_argument("--save", action="store_true", help="save one PNG per case")
ap.add_argument("--overlay", action="store_true",
                help="draw all cases on the same axes instead of one figure per case")
ap.add_argument("--list", action="store_true", help="print the indices and file names, then exit")
args = ap.parse_args()

if args.list:
    for i, c in enumerate(candidates):
        print(i, os.path.basename(c))
    raise SystemExit

if args.files:
    FILES = args.files
elif args.idx:
    bad = [i for i in args.idx if not -len(candidates) <= i < len(candidates)]
    if bad:
        raise SystemExit(f"indices out of range {bad}; there are {len(candidates)} candidates")
    FILES = [candidates[i] for i in args.idx]
elif args.range:
    FILES = candidates[args.range[0]:args.range[1]]
else:
    raise SystemExit("give --idx, --range or --files (use --list to see the indices)")

COLS = args.cols
T_RANGE = tuple(args.trange) if args.trange else None
SAVE_FIGS = args.save

def load_case(path):
    if path.endswith(".npz"):
        with np.load(path) as z:
            Y = z["qq"].T
            t = z["time"]
        names = [f"q{i}" for i in range(Y.shape[1])]
        return t, Y, names

    names = ["time", "plunge", "zdot", "pitch", "pitchdot"]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        skiprows=1,
        header=None
    )

    if df.shape[1] != 5:
        # Print the first data row to diagnose any remaining mismatch
        with open(path) as f:
            first_lines = [next(f, "").rstrip() for _ in range(3)]
        raise ValueError(
            f"{path}: expected 5 columns, found {df.shape[1]}.\n"
            f"First lines of file:\n" + "\n".join(first_lines)
        )

    df.columns = names
    df = df.apply(pd.to_numeric, errors="raise")

    t = df["time"].to_numpy(dtype=float)
    Y = df[names[1:]].to_numpy(dtype=float)

    return t, Y, names[1:]

data = []
for f in FILES:
    t, Y, names = load_case(f)
    print(f"{os.path.basename(f)}: shape {Y.shape}, t {t[0]:.4g} -> {t[-1]:.4g} s, "
          f"dt = {np.median(np.diff(t)):.3e} s, finite: {np.all(np.isfinite(Y))}")
    data.append((os.path.basename(f), t, Y, names))
SAVE_FIGS = False          # True: also save one PNG per case
OUT_FIG_DIR = "case_plots"

if SAVE_FIGS:
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

cols = [c for c in COLS if all(c < d[2].shape[1] for d in data)]

for label, t, Y, names in data:
    sel = slice(None) if T_RANGE is None else (t >= T_RANGE[0]) & (t <= T_RANGE[1])
    fig, ax = plt.subplots(len(cols), 1, figsize=(10, 2.6 * len(cols)),
                           sharex=True, squeeze=False)
    ax = ax[:, 0]
    for a, j in zip(ax, cols):
        a.plot(t[sel], Y[sel, j], lw=0.8, label=label)
        a.set_ylabel(names[j])
        a.grid(True)
        a.legend(fontsize=7)
    ax[-1].set_xlabel('t [s]')
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_FIG_DIR, os.path.splitext(label)[0] + ".png"), dpi=130)

plt.show()