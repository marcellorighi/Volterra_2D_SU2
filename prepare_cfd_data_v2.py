#!/usr/bin/env python3
"""Prepare training and test datasets from unsteady CFD histories."""

import argparse
import csv
import os
import warnings

import numpy as np

# ------------------------------------------------------------ Configuration

# AOAS = [0, 1, 2, 3, 4, 5, 6]
AOAS = [0, 1, 2]
TURB_MODELS = ["B"]
# TURB_MODELS = ["A", "B", "C", "D"]

CASES_DIR = "cases"
STEADY_SUBDIR = "steady"
STEADY_FILE = "history.csv"

TRAIN_SUBDIR = "unsteady"
TRAIN_FILE = "history_05_00002.dat"

TEST_SUBDIR = "test_unsteady"
TEST_FILE = "history_05_00002.dat"

OUTPUT_DIR = os.path.join("identification", "data")
PLOT_DIR = os.path.join("identification", "plots")

DT = 1.0e-4
N_INNER = 48
CM_SIGN = -1.0

# Excitation signals are written by multisine_input_ramp_v3.py
# (set WRITE_FILE = True and OUT_FILE to one of these names).
TRAIN_INPUT_FILE = "multisine_train.dat"
TEST_INPUT_FILE = "multisine_test.dat"

# Column names in the input file; plunge axis must match the CFD motion.
PITCH_COLUMN = "theta"       # degrees in the file
PLUNGE_AXIS = None           # "y" -> column "ydot", "z" -> "zdot";
                             # None = use whichever single axis the file has

# CFD output columns
COLUMN_NAMES = {
    "it": "Time_Iter",
    "inner": "Inner_Iter",
    "cl": "CL",
    "cm": "CMz",
}
SKIP_HEADER = 2
DELIMITER = ","


# ------------------------------------------------------------ Paths

def case_name(aoa):
    s = f"{float(aoa):g}"
    if s.startswith("-"):
        return f"AOA_m{s[1:].replace('.', 'p')}"
    return f"AOA_{s.replace('.', 'p')}"


def case_dir(aoa, model):
    return os.path.join(
        CASES_DIR, case_name(aoa), "results", model
    )


# ------------------------------------------------------------ Reading

def read_history(filename):
    """Read a history file, discarding incomplete/non-numeric rows."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open(filename, newline="") as f:
        lines = f.read().splitlines()

    # A final line without a newline may be only partially written.
    if lines and not open(filename, "rb").read().endswith(b"\n"):
        lines = lines[:-1]

    if len(lines) <= SKIP_HEADER:
        raise ValueError(f"{filename}: no history data found")

    header = next(csv.reader([lines[SKIP_HEADER - 1]], delimiter=DELIMITER))
    header = [h.strip().strip('"') for h in header]
    index = {name: i for i, name in enumerate(header)}

    missing = [
        name for name in COLUMN_NAMES.values() if name not in index
    ]
    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}; header is {header}"
        )

    rows = []
    for line in lines[SKIP_HEADER:]:
        try:
            row = next(csv.reader([line], delimiter=DELIMITER))
            values = [float(v) for v in row]
            if len(values) < len(header) or not np.all(np.isfinite(values)):
                continue
            rows.append(values)
        except (ValueError, StopIteration):
            continue

    if not rows:
        raise ValueError(f"{filename}: no complete numeric rows")

    data = np.asarray(rows)
    return {
        key: data[:, index[name]]
        for key, name in COLUMN_NAMES.items()
    }

def read_steady(aoa, model):
    """Read the final steady CL and CMz values from history.csv."""
    filename = os.path.join(
        case_dir(aoa, model), STEADY_SUBDIR, STEADY_FILE
    )

    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open(filename, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"{filename}: empty file")

        header = [h.strip().strip('"') for h in header]

        try:
            icl = header.index("CL")
            icm = header.index("CMz")
        except ValueError as err:
            raise ValueError(
                f"{filename}: could not find CL and CMz in header {header}"
            ) from err

        last = None
        for row in reader:
            if len(row) <= max(icl, icm):
                continue
            try:
                cl = float(row[icl])
                cmz = float(row[icm])
            except ValueError:
                continue

            if np.isfinite(cl) and np.isfinite(cmz):
                last = (cl, CM_SIGN * cmz)

    if last is None:
        raise ValueError(f"{filename}: no valid numeric data rows")

    return last

def final_inner_iteration(history):
    """Keep the last inner-iteration sample for each physical time step."""
    it = history["it"]
    inner = history["inner"]

    # Prefer the nominal final inner iteration where it is present.
    keep = inner == (N_INNER - 1)

    # If the file does not contain those rows, use the last row for each
    # Time_Iter instead.
    if not np.any(keep):
        _, reverse_indices = np.unique(it[::-1], return_index=True)
        keep_indices = len(it) - 1 - reverse_indices
        keep = np.zeros(len(it), dtype=bool)
        keep[keep_indices] = True

    selected = np.flatnonzero(keep)
    selected = selected[np.argsort(it[selected] * 0 + it[selected])]
    # Ensure chronological order and one sample per Time_Iter.
    selected = selected[np.argsort(it[selected], kind="stable")]
    _, reverse_indices = np.unique(it[selected][::-1], return_index=True)
    selected = selected[len(selected) - 1 - reverse_indices]
    selected = selected[np.argsort(it[selected], kind="stable")]

    return {key: values[selected] for key, values in history.items()}


# ------------------------------------------------------------ Inputs

def read_input_file(filename):
    """Read a signal file written by multisine_input_ramp_v3.py.

    Returns (columns, meta): columns maps column name -> array, meta maps
    the keys of the '# META key=value ...' header line to floats.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    comments = []
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                break
            comments.append(line[1:].strip())

    # Last two comment lines are the column names and the units.
    names = comments[-2].split()
    data = np.loadtxt(filename, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(
            f"{filename}: {data.shape[1]} columns but header names {names}"
        )

    meta = {}
    for line in comments:
        if line.startswith("META"):
            for item in line.split()[1:]:
                key, value = item.split("=")
                meta[key] = float(value)

    return dict(zip(names, data.T)), meta


def make_inputs(time, v_inf, input_file):
    """Pitch [rad] and plunge-induced angle atan(zdot/V) [rad] at `time`."""
    cols, meta = read_input_file(input_file)

    dt_file = meta.get("DT", DT)
    if not np.isclose(dt_file, DT):
        raise ValueError(f"{input_file}: DT={dt_file} differs from DT={DT}")

    if PLUNGE_AXIS is None:
        found = [c for c in ("ydot", "zdot") if c in cols]
        if len(found) != 1:
            raise ValueError(
                f"{input_file}: set PLUNGE_AXIS; plunge columns found: {found}"
            )
        plunge_col = found[0]
    else:
        plunge_col = f"{PLUNGE_AXIS}dot"
    for name in ("time", PITCH_COLUMN, plunge_col):
        if name not in cols:
            raise ValueError(
                f"{input_file}: column '{name}' not found; "
                f"available: {list(cols)}"
            )

    t_in = cols["time"]
    if time[0] < t_in[0] - 1e-12 or time[-1] > t_in[-1] + 1e-12:
        raise ValueError(
            f"{input_file}: covers t = {t_in[0]:g} ... {t_in[-1]:g} s but "
            f"the CFD history needs {time[0]:g} ... {time[-1]:g} s"
        )

    pitch = np.radians(np.interp(time, t_in, cols[PITCH_COLUMN]))
    zdot = np.interp(time, t_in, cols[plunge_col])
    zdotoV = np.arctan(zdot / v_inf)

    return pitch, zdotoV, meta

# ------------------------------------------------------------ Processing

def prepare_dataset(aoa, model, subdir, filename, output_name,
                    v_inf, signal_file):
    input_path = os.path.join(
        case_dir(aoa, model), subdir, filename
    )
    history = final_inner_iteration(read_history(input_path))

    # A signal file with the same name placed next to the CFD history
    # overrides the shared one for this case only.
    local = os.path.join(
        case_dir(aoa, model), subdir, os.path.basename(signal_file)
    )
    if os.path.isfile(local):
        signal_file = local
        print(f"{case_name(aoa)}/{model}: using case-specific {local}")

    n = len(history["cl"])
    # Time_Iter is the physical time step index (t_i = i * DT in the solver).
    time = history["it"] * DT
    pitch, zdotoV, meta = make_inputs(time, v_inf, signal_file)

    # Preserve the absolute CFD coefficients. CM is converted to the
    # sign convention used in the original identification script.
    cl = history["cl"]
    cm = CM_SIGN * history["cm"]

    # The steady reference is shared by training and test.
    cl0, cm0 = read_steady(aoa, model)

    output_path = os.path.join(
        OUTPUT_DIR, case_name(aoa), model, output_name
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = np.column_stack((time, pitch, zdotoV, cl, cm))
    np.savetxt(
        output_path, data, delimiter=",",
        header="TIME,PITCH,ZDOTOV,CL,CM", comments="",
        fmt="%.10e"
    )

    print(
        f"{case_name(aoa)}/{model}: wrote {output_path} "
        f"({n} samples; steady CL0={cl0:.6g}, CM0={cm0:.6g})"
    )

    # Ramp/period information for the identification (discard the ramp,
    # average whole periods).
    meta_path = os.path.splitext(output_path)[0] + "_meta.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["KEY", "VALUE"])
        writer.writerow(["SIGNAL_FILE", signal_file])
        for key, value in meta.items():
            writer.writerow([key, value])

    steady_path = os.path.join(
        OUTPUT_DIR, case_name(aoa), model, "steady.csv"
    )
    os.makedirs(os.path.dirname(steady_path), exist_ok=True)

    with open(steady_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AOA_DEG", "MODEL", "CL0", "CM0"])
        writer.writerow([aoa, model, cl0, cm0])


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training and test CFD datasets."
    )
    parser.add_argument(
        "--mach", type=float, default=0.8,
        help="freestream Mach number"
    )
    parser.add_argument(
        "--vref", type=float, default=123.0,
        help="reference speed: v_inf = vref * Mach / 0.80"
    )
    parser.add_argument(
        "--aoa", type=float, nargs="+", default=None
    )
    parser.add_argument(
        "--model", nargs="+", default=None
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="plot the prepared training and test data"
    )
    parser.add_argument(
        "--which", choices=("train", "test", "both"), default="train",
        help="dataset(s) to prepare and plot (default: train)"
    )
    args = parser.parse_args()

    if args.mach <= 0 or args.vref <= 0:
        parser.error("--mach and --vref must be positive")

    v_inf = args.vref * args.mach / 0.80
    aoas = [
        a for a in AOAS
        if args.aoa is None or any(np.isclose(a, x) for x in args.aoa)
    ]
    models = [
        m for m in TURB_MODELS
        if args.model is None or m in args.model
    ]
     
    do_train = args.which in ("train", "both")
    do_test  = args.which in ("test", "both")

    for aoa in aoas:
        for model in models:
            if do_train:
                prepare_dataset(
                    aoa, model, TRAIN_SUBDIR, TRAIN_FILE, "training.csv",
                    v_inf, TRAIN_INPUT_FILE
                )
            if do_test:
                prepare_dataset(
                    aoa, model, TEST_SUBDIR, TEST_FILE, "test.csv",
                    v_inf, TEST_INPUT_FILE
                )

    if args.plot:
        plot_datasets(aoas, models, which=args.which)

def plot_datasets(aoas, models, which="both"):
    import matplotlib.pyplot as plt

    files = []
    if which in ("train", "both"):
        files.append(("training.csv", "Training"))
    if which in ("test", "both"):
        files.append(("test.csv", "Test"))

    os.makedirs(PLOT_DIR, exist_ok=True)

    for aoa in aoas:
        for model in models:
            folder = os.path.join(
                OUTPUT_DIR, case_name(aoa), model
            )
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for filename, label in files:
                path = os.path.join(folder, filename)
                data = np.genfromtxt(
                    path, delimiter=",", names=True
                )
                axes[0, 0].plot(data["TIME"], data["PITCH"], label=label)
                axes[0, 1].plot(data["TIME"], data["ZDOTOV"], label=label)
                axes[1, 0].plot(data["TIME"], data["CL"], label=label)
                axes[1, 1].plot(data["TIME"], data["CM"], label=label)

            for ax, title in zip(
                axes.flat, ["Pitch", "ZDOT", "CL", "CM"]
            ):
                ax.set_title(title)
                ax.set_xlabel("Time [s]")
                ax.grid(True)
                ax.legend()

            fig.suptitle(f"{case_name(aoa)} — model {model}")
            fig.tight_layout()
            out = os.path.join(
                PLOT_DIR, f"{case_name(aoa)}_{model}.png"
            )
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()