import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

input = "data/ER-ER-N1000-D50-S1-R1"
df = pd.read_csv(f'{input}.csv')
N = 1000

k1s = df['k1'].to_numpy()
k2s = df['k2'].to_numpy()

#--------------------------------------------
bin_size = 2

k1_min = 5
k1_max = 110
k2_min = 5
k2_max = 110
k1s = np.arange(k1_min, k1_max, bin_size)
k2s = np.arange(k2_min, k2_max, bin_size)

data = np.zeros((len(k1s), len(k2s)))
num_rep = np.zeros((len(k1s), len(k2s)))

for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        data[i, j] = df.loc[mask, 'prob_avg'].sum()
        num_rep[i, j] = mask.sum()

num_rep[num_rep == 0] = 1

data /= num_rep
#--------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

plt.subplots_adjust(wspace=0.5, hspace=2, left=.15, bottom=0.1, right=.88, top=1)
ax.tick_params(direction='out', top=True, right=True)

# --- plot

import seaborn as sns
from matplotlib.colors import ListedColormap
cmap = sns.color_palette("blend:#352b87,#007fdf,#3bc181,#ffbb00,#ffd700", as_cmap=True)
def pastel_cmap(cmap):
    n = cmap.N
    pastel_colors = np.clip(cmap(np.linspace(0, 1, n)) + 0.1, 0, 1)
    return sns.blend_palette(pastel_colors, n_colors=n, as_cmap=True)
cmap = pastel_cmap(cmap)

# --- heatmap
c = ax.pcolormesh(k1s, k2s, data, cmap=cmap, linewidth=0, rasterized=True, vmin=0, vmax=1.0)
cbar = plt.colorbar(c, fraction=0.046, pad=0.04, ticks=[0, 0.3, 0.7, 1])
cbar.ax.set_ylabel('$q$', rotation=0, fontsize=20, labelpad=9, y=0.55)

# --- model
xlims = [5, 100]
ylims = [5, 100]
def lambda2_ER(N, k, l):
    p = l * k / (N - 1)
    gamma = 0.57721566490153286060651209008240243104215933593992
    lambda2s = p * (N - 1) - np.sqrt(2 * p * (1 - p) * (N - 1) * np.log(N)) + np.sqrt((N - 1) * p * (1 - p) / (2 * np.log(N))) * np.log(np.sqrt(2 * np.pi * np.log(N**2 / (2 * np.pi)))) - np.sqrt((N - 1) * p * (1 - p) / (2 * np.log(N))) * gamma
    return lambda2s / l

def lambda2_RR(N, k, l):
    k = k * l
    l_t = k - 2 * np.sqrt(k - 1)
    return l_t / l

def f(x, k1, N):
    return lambda2_RR(N, (k1+x)/2, 2) - lambda2_ER(N, k1, 1)

from scipy.optimize import newton

k1s_theory = list(range(15,100))
k2s_theory = np.zeros(len(k1s_theory))

for i in range(len(k1s_theory)):
    k1_theory = k1s_theory[i]
    k2s_theory[i] = newton(f, k1_theory, args=(k1_theory,N,))

#  --- theoretical bounds
ax.plot(k1s_theory, k2s_theory, color='#000000', linewidth=2.5, linestyle='--')
ax.plot(k2s_theory, k1s_theory, color='#000000', linewidth=2.5, linestyle='--')

# --- lable + ticks + legend

ax.set_xlabel('$\\langle k^{[2]} \\rangle$', fontfamily='Times')
ax.set_ylabel('$\\langle k^{[1]} \\rangle$', fontfamily='Times')

xticks = [5,25,50,75,100]
yticks = [5,25,50,75,100]

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(yticks[0], yticks[-1])

ax.set_aspect('equal')

ax.text(-0.15, 0.85, "(a)", fontsize=20, ha='center', va='center', transform=ax.transAxes)

plt.savefig(f'plots/fig4a-ER-NC-bounds.pdf')
