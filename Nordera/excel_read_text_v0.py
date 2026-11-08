import pandas as pd
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print(SCRIPT_DIR)

df = pd.read_excel(
    "all_results-2.xlsx",
    sheet_name="aoa_0"
)

print(df.columns)

time = df["time"].to_numpy()

pitch_cl = df["pitch_cl"].to_numpy()
pitch_cm = df["pitch_cm"].to_numpy()

plunge_cl = df["plunge_cl"].to_numpy()
plunge_cm = df["plunge_cm"].to_numpy()

pitch_cl_smooth = df["pitch_cl_smooth"].to_numpy()
pitch_cm_smooth = df["pitch_cm_smooth"].to_numpy()

plunge_cl_smooth = df["plunge_cl_smooth"].to_numpy()
plunge_cm_smooth = df["plunge_cm_smooth"].to_numpy()

fig, ax = plt.subplots(
    figsize=(7, 3.5)
)

ax.plot(
    time,
    pitch_cl,
    label="Raw",
    linewidth=0.8
)

ax.plot(
    time,
    pitch_cl_smooth,
    label="Smoothed",
    linewidth=1.5
)

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$C_L$")

ax.grid(True, linestyle=":")
ax.legend()

fig.tight_layout()

plt.show()