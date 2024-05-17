import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

input = "ER-ER-N500-D50"

xticks = [5,150,325,490]
yticks = [5,150,325,490]

# xticks = [5,25,50,75,100]
# yticks = [5,25,50,75,100]

bin_size = 2

k1_min = xticks[0]
k1_max = xticks[-1] + bin_size
k2_min = yticks[0]
k2_max = yticks[-1] + bin_size
k1s = np.arange(k1_min, k1_max, bin_size)
k2s = np.arange(k2_min, k2_max, bin_size)
#--------------------------------------------
df = pd.read_csv(f'data/{input}-S0-R0.csv')
data1a = np.zeros((len(k1s), len(k2s)))
data1b = np.zeros((len(k1s), len(k2s)))
data1c = np.zeros((len(k1s), len(k2s)))
num_rep = np.zeros((len(k1s), len(k2s)))
for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        print("1: ", i, j, "\r", end="")
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        data1a[i, j] = df.loc[mask, 'prob_avg'].sum()
        data1b[i, j] = df.loc[mask, 'correct'].sum()
        data1c[i, j] = df.loc[mask, 'η_avg'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data1a /= num_rep
data1b /= num_rep
data1c /= num_rep
#--------------------------------------------
df = pd.read_csv(f'data/{input}-S1-R0.csv')
data2a = np.zeros((len(k1s), len(k2s)))
data2b = np.zeros((len(k1s), len(k2s)))
data2c = np.zeros((len(k1s), len(k2s)))
num_rep = np.zeros((len(k1s), len(k2s)))
for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        print("2: ", i, j, "\r", end="")
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        data2a[i, j] = df.loc[mask, 'prob_avg'].sum()
        data2b[i, j] = df.loc[mask, 'correct'].sum()
        data2c[i, j] = df.loc[mask, 'η_avg'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data2a /= num_rep
data2b /= num_rep
data2c /= num_rep
#--------------------------------------------
df = pd.read_csv(f'data/{input}-S1-R1.csv')
data3a = np.zeros((len(k1s), len(k2s)))
data3b = np.zeros((len(k1s), len(k2s)))
data3c = np.zeros((len(k1s), len(k2s)))
num_rep = np.zeros((len(k1s), len(k2s)))
for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        print("3: ", i, j, "\r", end="")
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        data3a[i, j] = df.loc[mask, 'prob_avg'].sum()
        data3b[i, j] = df.loc[mask, 'correct'].sum()
        data3c[i, j] = df.loc[mask, 'η_avg'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data3a /= num_rep
data3b /= num_rep
data3c /= num_rep
#--------------------------------------------

fig, ax = plt.subplots(3, 4, figsize=(9.5, 8.8))

ax1a, ax2a, ax3a, cmapa, ax1b, ax2b, ax3b, cmapb, ax1c, ax2c, ax3c, cmapc = ax.flatten()

plt.subplots_adjust(wspace=0.2, hspace=0.05, left=0.1, bottom=0.08, right=1.14, top=0.95)

ax1a.tick_params(direction='out', top=True, right=True)
ax2a.tick_params(direction='out', top=True, right=True)
ax3a.tick_params(direction='out', top=True, right=True)
ax1b.tick_params(direction='out', top=True, right=True)
ax2b.tick_params(direction='out', top=True, right=True)
ax3b.tick_params(direction='out', top=True, right=True)
ax1c.tick_params(direction='out', top=True, right=True)
ax2c.tick_params(direction='out', top=True, right=True)
ax3c.tick_params(direction='out', top=True, right=True)

# --- plot

ax1a.text(0.5, 1.1, "RA", fontsize=20, ha='center', va='center', transform=ax1a.transAxes)
ax2a.text(0.5, 1.1, "PC", fontsize=20, ha='center', va='center', transform=ax2a.transAxes)
ax3a.text(0.5, 1.1, "NC", fontsize=20, ha='center', va='center', transform=ax3a.transAxes)


cmapa.axis('off')
cmapb.axis('off')
cmapc.axis('off')

# remove x space from cmapa axis
cmapa.set_position([0.05, 0.05, 0.95, 0.05])
cmapb.set_position([0.05, 0.05, 0.95, 0.05])
cmapc.set_position([0.05, 0.05, 0.95, 0.05])

c1a = ax1a.pcolormesh(k1s, k2s, data1a, cmap='viridis', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c2a = ax2a.pcolormesh(k1s, k2s, data2a, cmap='viridis', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c3a = ax3a.pcolormesh(k1s, k2s, data3a, cmap='viridis', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
divider = make_axes_locatable(cmapa)
cbara = plt.colorbar(c3a, fraction=0.046, pad=0.04, ax=cmapa, ticks=[0, 0.3, 0.7, 1], cax=cmapc.inset_axes((0.89, 12.9, 0.012, 4.8)))
cbara.ax.set_yticklabels(['0', '0.3', '0.7', '1'], fontsize=16)
cbara.ax.set_ylabel('$p$', rotation=0, fontsize=20, labelpad=9, y=0.55)

c1b = ax1b.pcolormesh(k1s, k2s, data1b, cmap='Reds', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c2b = ax2b.pcolormesh(k1s, k2s, data2b, cmap='Reds', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c3b = ax3b.pcolormesh(k1s, k2s, data3b, cmap='Reds', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
divider = make_axes_locatable(cmapb)
cbarb = plt.colorbar(c3b, fraction=0.046, pad=0.04, ax=cmapb, ticks=[0, 0.3, 0.7, 1], cax=cmapc.inset_axes((0.89, 7.1, 0.012, 4.8)))
cbarb.ax.set_yticklabels(['0', '0.3', '0.7', '1'], fontsize=16)
cbarb.ax.set_ylabel('$\\Delta$', rotation=0, fontsize=20, labelpad=9, y=0.55)

lim = 0.25
c1c = ax1c.pcolormesh(k1s, k2s, data1c, cmap='bwr', linewidth=0, rasterized=True, vmin=-lim, vmax=lim)
c2c = ax2c.pcolormesh(k1s, k2s, data2c, cmap='bwr', linewidth=0, rasterized=True, vmin=-lim, vmax=lim)
c3c = ax3c.pcolormesh(k1s, k2s, data3c, cmap='bwr', linewidth=0, rasterized=True, vmin=-lim, vmax=lim)
divider = make_axes_locatable(cmapc)
cbarc = plt.colorbar(c3c, fraction=0.046, pad=0.04, ax=cmapc, ticks=[-lim, 0, lim], cax=cmapc.inset_axes((0.89, 1.35, 0.012, 4.8)))
cbarc.ax.set_yticklabels([f'< -{lim}', '0', f'> {lim}'], fontsize=16)
cbarc.ax.set_ylabel('$\\langle \\eta \\rangle$', rotation=0, fontsize=20, labelpad=-9, y=0.55)

ax1a.set_xticklabels([])
ax2a.set_xticklabels([])
ax2a.set_yticklabels([])
ax3a.set_xticklabels([])
ax3a.set_yticklabels([])
ax1b.set_xticklabels([])
ax2b.set_xticklabels([])
ax2b.set_yticklabels([])
ax3b.set_xticklabels([])
ax3b.set_yticklabels([])
ax2c.set_yticklabels([])
ax3c.set_yticklabels([])

ax1c.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax1a.set_ylabel('$\\langle k^{[1]} \\rangle$')
ax1b.set_ylabel('$\\langle k^{[1]} \\rangle$')
ax1c.set_ylabel('$\\langle k^{[1]} \\rangle$')
ax1a.set_xticks(xticks)
ax1b.set_xticks(xticks)
ax1c.set_xticks(xticks)
ax1a.set_yticks(yticks)
ax1b.set_yticks(yticks)
ax1c.set_yticks(yticks)
ax1a.set_xlim(xticks[0], xticks[-1])
ax1b.set_xlim(xticks[0], xticks[-1])
ax1c.set_xlim(xticks[0], xticks[-1])
ax1a.set_ylim(yticks[0], yticks[-1])
ax1b.set_ylim(yticks[0], yticks[-1])
ax1c.set_ylim(yticks[0], yticks[-1])
ax1a.set_aspect('equal')
ax1b.set_aspect('equal')
ax1c.set_aspect('equal')

ax2c.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax2a.set_xticks(xticks)
ax2b.set_xticks(xticks)
ax2c.set_xticks(xticks)
ax2a.set_yticks(yticks)
ax2b.set_yticks(yticks)
ax2c.set_yticks(yticks)
ax2a.set_xlim(xticks[0], xticks[-1])
ax2b.set_xlim(xticks[0], xticks[-1])
ax2c.set_xlim(xticks[0], xticks[-1])
ax2a.set_ylim(yticks[0], yticks[-1])
ax2b.set_ylim(yticks[0], yticks[-1])
ax2c.set_ylim(yticks[0], yticks[-1])
ax2a.set_aspect('equal')
ax2b.set_aspect('equal')
ax2c.set_aspect('equal')

ax3c.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax3a.set_xticks(xticks)
ax3b.set_xticks(xticks)
ax3c.set_xticks(xticks)
ax3a.set_yticks(yticks)
ax3b.set_yticks(yticks)
ax3c.set_yticks(yticks)
ax3a.set_xlim(xticks[0], xticks[-1])
ax3b.set_xlim(xticks[0], xticks[-1])
ax3c.set_xlim(xticks[0], xticks[-1])
ax3a.set_ylim(yticks[0], yticks[-1])
ax3b.set_ylim(yticks[0], yticks[-1])
ax3c.set_ylim(yticks[0], yticks[-1])
ax3a.set_aspect('equal')
ax3b.set_aspect('equal')
ax3c.set_aspect('equal')

plt.savefig(f'plots/fig2-{input}-plot.pdf', dpi=300)