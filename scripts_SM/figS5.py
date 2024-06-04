import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

input = f'ER-ER-N500-D50-S1-R1'

bin_size = 2
k1_min = 4
k1_max = 109 + bin_size
k2_min = 4
k2_max = 109 + bin_size
k1s = np.arange(k1_min, k1_max, bin_size)
k2s = np.arange(k2_min, k2_max, bin_size)
data = np.zeros((len(k1s), len(k2s)))
data2 = np.zeros((len(k1s), len(k2s)))
num_rep = np.zeros((len(k1s), len(k2s)))

#--------------------------------------------
for k in range(1, 6):
    print(f'Processing {k}...')
    
    if input == f'ER-ER-N500-D50-S1-R1':
        df = pd.read_csv(f'data_SM/ER-ER-N500-D10-S1-R1-opt-{k}.csv')
    else:
        df = pd.read_csv(f'data_SM/{input}-opt.csv')

    # Vectorized computations
    for i in range(len(k1s)):
        k1_min = k1s[i]
        k1_max = k1_min + bin_size
        for j in range(len(k2s)):
            k2_min = k2s[j]
            k2_max = k2_min + bin_size
            mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
            data[i, j] = df.loc[mask, 'prob_avg'].sum() # ζ_avg prob_avg
            num_rep[i, j] = mask.sum()

    # Avoid division by zero
    num_rep[num_rep == 0] = 1

    # Perform division
    data /= num_rep
#--------------------------------------------
num_rep = np.zeros((len(k1s), len(k2s)))

df = pd.read_csv(f'data_SM/{input}.csv')

# Vectorized computations
for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        data2[i, j] = df.loc[mask, 'prob_avg'].sum() # ζ_avg prob_avg
        num_rep[i, j] = mask.sum()

# Avoid division by zero
num_rep[num_rep == 0] = 1

# Perform division
data2 /= num_rep
#--------------------------------------------

data = data - data2

# compute the ratio of correct predictions
Delta = data.sum() / (data.shape[0] * data.shape[1])
print("increment = ", round(Delta, 4))

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

plt.subplots_adjust(wspace=0.5, hspace=2, left=.15, bottom=0.05, right=.85, top=1)
ax.tick_params(direction='out', top=True, right=True)


# --- plot

cmap = 'Blues'

# heatmap
c = ax.pcolormesh(k1s, k2s, data, cmap=cmap, linewidth=0, rasterized=True, vmin=0, vmax=1)
cbar = plt.colorbar(c, fraction=0.046, pad=0.04, ticks=[0, 0.3, 0.7, 1])
cbar.ax.set_ylabel('$\\Delta q$', rotation=0, fontsize=20, labelpad=9, y=0.55)

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

plt.savefig(f'plots_SM/figS5-opt-ER.pdf')
