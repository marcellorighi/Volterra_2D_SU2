import numpy as np
import matplotlib.pyplot as plt

# ── Data ─────────────────────────────────────────────────────────────
labels  = [r'$\rho_2$', r'$\rho_3$', r'$\rho_4$']
values1 = [0.45, 0.05, 0.02]   # first bar in each group
values2 = [0.40, 0.20, 0.02]   # second bar in each group

# ── Layout ───────────────────────────────────────────────────────────
x     = np.arange(len(labels))  # group centre positions
width = 0.35                     # bar width

fig, ax = plt.subplots(figsize=(7, 4))

bars1 = ax.bar(x - width/2, values1, width, label=r'$C_l$',
               color='#222222', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, values2, width, label=r'$C_m$',
               color='#888888', edgecolor='white', linewidth=0.5)

# ── Value labels on top of each bar ──────────────────────────────────
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# ── Axes formatting ───────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
# ax.set_ylabel('Value')
ax.set_ylim(0, 0.55)
ax.legend(framealpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/bar_chart.png', dpi=150)
plt.show()
#print("Saved to bar_chart.png")