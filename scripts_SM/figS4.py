import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

xticks = [5,25,50,75,100]
yticks = [5,25,50,75,100]

bin_size = 2

k1_min = xticks[0]
k1_max = xticks[-1] + bin_size
k2_min = yticks[0]
k2_max = yticks[-1] + bin_size
k1s = np.arange(k1_min, k1_max, bin_size)
k2s = np.arange(k2_min, k2_max, bin_size)
#--------------------------------------------
df = pd.read_csv(f'data_SM/event-ER-ER-N500-D50-S1-R1.csv')
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
        data1b[i, j] = df.loc[mask, 'δ'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data1a /= num_rep
data1b /= num_rep
#--------------------------------------------
df = pd.read_csv(f'data_SM/SF-SF-N500-D50-S1-R1.csv')
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
        data2b[i, j] = df.loc[mask, 'δ'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data2a /= num_rep
data2b /= num_rep
#--------------------------------------------
df = pd.read_csv(f'data_SM/ER-SF-N500-D50-S1-R1.csv')
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
        data3b[i, j] = df.loc[mask, 'δ'].sum()
        num_rep[i, j] = mask.sum()
num_rep[num_rep == 0] = 1
data3a /= num_rep
data3b /= num_rep
#--------------------------------------------

fig, ax = plt.subplots(2, 4, figsize=(9.5, 6.2))

ax1a, ax2a, ax3a, cmapa, ax1b, ax2b, ax3b, cmapb = ax.flatten()

plt.subplots_adjust(wspace=0.2, hspace=0.05, left=0.1, bottom=0.08, right=1.14, top=0.95)

ax1a.tick_params(direction='out', top=True, right=True)
ax2a.tick_params(direction='out', top=True, right=True)
ax3a.tick_params(direction='out', top=True, right=True)
ax1b.tick_params(direction='out', top=True, right=True)
ax2b.tick_params(direction='out', top=True, right=True)
ax3b.tick_params(direction='out', top=True, right=True)

# --- plot

ax1a.text(0.5, 1.1, "ER-ER (NC)", fontsize=20, ha='center', va='center', transform=ax1a.transAxes)
ax2a.text(0.5, 1.1, "SF-SF (NC)", fontsize=20, ha='center', va='center', transform=ax2a.transAxes)
ax3a.text(0.5, 1.1, "ER-SF (NC)", fontsize=20, ha='center', va='center', transform=ax3a.transAxes)


cmapa.axis('off')
cmapb.axis('off')

# remove x space from cmapa axis
cmapa.set_position([0.05, 0.05, 0.95, 0.05])
cmapb.set_position([0.05, 0.05, 0.95, 0.05])

import seaborn as sns
from matplotlib.colors import ListedColormap
cmapp = sns.color_palette("blend:#352b87,#007fdf,#3bc181,#ffbb00,#ffd700", as_cmap=True)
def pastel_cmap(cmap):
    n = cmap.N
    pastel_colors = np.clip(cmap(np.linspace(0, 1, n)) + 0.1, 0, 1)
    return sns.blend_palette(pastel_colors, n_colors=n, as_cmap=True)
cmapp = pastel_cmap(cmapp)

c1a = ax1a.pcolormesh(k1s, k2s, data1a, cmap=cmapp, linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c2a = ax2a.pcolormesh(k1s, k2s, data2a, cmap=cmapp, linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c3a = ax3a.pcolormesh(k1s, k2s, data3a, cmap=cmapp, linewidth=0, rasterized=True, vmin=0, vmax=1.0)
divider = make_axes_locatable(cmapa)
cbara = plt.colorbar(c3a, fraction=0.046, pad=0.04, ax=cmapa, ticks=[0, 0.3, 0.7, 1], cax=cmapa.inset_axes((0.89, 10.3, 0.012, 6.8)))
cbara.ax.set_yticklabels(['0', '0.3', '0.7', '1'], fontsize=16)
cbara.ax.set_ylabel('$p$', rotation=0, fontsize=20, labelpad=9, y=0.55)

c1b = ax1b.pcolormesh(k1s, k2s, data1b, cmap='Greens', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c2b = ax2b.pcolormesh(k1s, k2s, data2b, cmap='Greens', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
c3b = ax3b.pcolormesh(k1s, k2s, data3b, cmap='Greens', linewidth=0, rasterized=True, vmin=0, vmax=1.0)
divider = make_axes_locatable(cmapb)
cbarb = plt.colorbar(c3b, fraction=0.046, pad=0.04, ax=cmapb, ticks=[0, 0.3, 0.7, 1], cax=cmapb.inset_axes((0.89, 1.5, 0.012, 6.8)))
cbarb.ax.set_yticklabels(['0', '0.3', '0.7', '1'], fontsize=16)
cbarb.ax.set_ylabel('$\\delta$', rotation=0, fontsize=20, labelpad=9, y=0.55)

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

ax1b.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax1a.set_ylabel('$\\langle k^{[1]} \\rangle$')
ax1b.set_ylabel('$\\langle k^{[1]} \\rangle$')
ax1a.set_xticks(xticks)
ax1b.set_xticks(xticks)
ax1a.set_yticks(yticks)
ax1b.set_yticks(yticks)
ax1a.set_xlim(xticks[0], xticks[-1])
ax1b.set_xlim(xticks[0], xticks[-1])
ax1a.set_ylim(yticks[0], yticks[-1])
ax1b.set_ylim(yticks[0], yticks[-1])
ax1a.set_aspect('equal')
ax1b.set_aspect('equal')

ax2b.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax2a.set_xticks(xticks)
ax2b.set_xticks(xticks)
ax2a.set_yticks(yticks)
ax2b.set_yticks(yticks)
ax2a.set_xlim(xticks[0], xticks[-1])
ax2b.set_xlim(xticks[0], xticks[-1])
ax2a.set_ylim(yticks[0], yticks[-1])
ax2b.set_ylim(yticks[0], yticks[-1])
ax2a.set_aspect('equal')
ax2b.set_aspect('equal')

ax3b.set_xlabel('$\\langle k^{[2]} \\rangle$')
ax3a.set_xticks(xticks)
ax3b.set_xticks(xticks)
ax3a.set_yticks(yticks)
ax3b.set_yticks(yticks)
ax3a.set_xlim(xticks[0], xticks[-1])
ax3b.set_xlim(xticks[0], xticks[-1])
ax3a.set_ylim(yticks[0], yticks[-1])
ax3b.set_ylim(yticks[0], yticks[-1])
ax3a.set_aspect('equal')
ax3b.set_aspect('equal')

plt.savefig(f'plots_SM/figS4-delta.pdf', dpi=300)
