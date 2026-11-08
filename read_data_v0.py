"""
load_data.py: read the training, test and steady-state data of all cases.

Folder layout:  <DATA_DIR>/AOA_<aoa>/<model>/{training.csv, test.csv, steady.csv}

Usage:
    from load_data import load_all
    data = load_all()
    train = data[(2, "A")]["training"]    # pandas DataFrame
    print(train["PITCH"].to_numpy())

or run it directly to print a summary:  python load_data.py
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("/Users/marcello/Documents/SU2/AePW4_TC3/identification/data")  # <-- change
AOAS = [0, 1, 2, 3, 4, 5, 6]
MODELS = ["A", "B", "C", "D"]


def load_all(data_dir=DATA_DIR, aoas=AOAS, models=MODELS):
    """Return {(aoa, model): {"training": df, "test": df, "steady": dict}}."""
    data = {}
    for aoa in aoas:
        for model in models:
            d = Path(data_dir) / f"AOA_{aoa}" / model
            paths = {k: d / f"{k}.csv" for k in ("training", "test", "steady")}
            missing = [p.name for p in paths.values() if not p.exists()]
            if missing:
                print(f"AOA {aoa}, model {model}: skipped, missing {missing}")
                continue
            steady = pd.read_csv(paths["steady"])
            if steady.empty:
                print(f"AOA {aoa}, model {model}: skipped, empty steady.csv")
                continue
            data[(aoa, model)] = {
                "training": pd.read_csv(paths["training"]),
                "test": pd.read_csv(paths["test"]),
                "steady": steady.iloc[0].to_dict(),     # e.g. CL0, CM0 (and the model name)
            }
    return data


if __name__ == "__main__":
    data = load_all()
    print(f"Loaded {len(data)} cases.\n")
    for (aoa, model), c in data.items():
        print(f"AOA {aoa}, model {model}: "
              f"{len(c['training'])} training rows, {len(c['test'])} test rows, "
              f"CL0 = {c['steady']['CL0']:.5f}, CM0 = {c['steady']['CM0']:.5f}")
    first = next(iter(data.values()))
    print("\nColumns:", list(first["training"].columns))