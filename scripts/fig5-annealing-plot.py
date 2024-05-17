import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

df = pd.read_csv('data/annealing2.csv')

df_info = pd.read_csv('data/annealing-info.csv')
free_best = df_info['free_best'].to_numpy()
restricted_best = df_info['restricted_best'].to_numpy()

num_cols = (len(df.columns) - 1)//2

t = df['t'].to_numpy()

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot()

plt.subplots_adjust(wspace=0, hspace=0, left=0.13,
                    top=0.9, right=0.95, bottom=0.16)
ax.tick_params(direction='in', top=False, right=False)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.2)
ax.tick_params(width=1.2)

# --- plot

colors = ["red", "blue"]

for i in range(1, num_cols+1):
    
    l = df[f'λ{i}'].to_numpy()

    if i <= num_cols//2:
        j = 0
    else:
        j = 1
    if i == free_best:
        a = 1
        lambda_max_free = l[-1]
    elif i == restricted_best:
        a = 1
        lambda_max_restricted = l[-1]
    else:
        a = 0.2
    
    ax.step(t, l, '-', linewidth=1.5, color=colors[j], alpha=a)

# --- lable + ticks + legend

ax.set_xlabel('step', fontfamily='Times', fontsize=18)
ax.set_ylabel('$\\lambda^{[s]}_2$', fontfamily='Times', fontsize=18)

ax.set_xlim(10, )
ax.set_ylim(29.1, 30)

ax.set_yticks([29.5,30, 30.5])
ax.set_yticklabels(['$29.5$', '$30.0$', '$30.5$'])

ax.set_xscale('log')
ax.tick_params(axis='x', which='minor', direction='in')
ax.xaxis.set_tick_params(which='both', pad=8)

# only four ticks in the y-axis
ax.locator_params(axis='y', nbins=4)

import matplotlib.lines as mlines
red_line = mlines.Line2D([], [], color='red', label="free SA")
blue_line = mlines.Line2D([], [], color='blue', label="restricted SA")
plt.legend(handles=[red_line, blue_line], loc='upper left', frameon=False, fontsize=16)


plt.savefig(f'plots/fig5-annealing.pdf', bbox_inches='tight')
