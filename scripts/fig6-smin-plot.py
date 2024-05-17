import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

df = pd.read_csv('data/smin.csv')

num_cols = (len(df.columns) - 1)//2

t = df['t'].to_numpy()

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot()

plt.subplots_adjust(wspace=0, hspace=0, left=0.1,
                    top=0.9, right=.98, bottom=0.15)
ax.tick_params(direction='in', top=False, right=False)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.2)
ax.tick_params(width=1.2)

# --- plot

colors = ["#029E02", "#4B0092"]

for i in range(1, num_cols+1):
    l = df[f'δ{i}'].to_numpy()
    if i <= num_cols//2:
        j = 0
    else:
        j = 1
    ax.step(t, l, '--', linewidth=1.5, color=colors[j], alpha=0.5)

for i in range(1, num_cols+1):
    l = df[f'λ{i}'].to_numpy()
    if i <= num_cols//2:
        j = 0
    else:
        j = 1
    ax.step(t, l, '-', linewidth=1.5, color=colors[j], alpha=1)

# --- lable + ticks + legend

ax.set_xlabel('step', fontfamily='Times', fontsize=18)
ax.set_ylabel('$\\lambda^{[s]}_{2}, \\, s^{[s]}_{\\min}$', fontfamily='Times', fontsize=18)
# ax.xaxis.set_label_coords(0.5, -0.1)
# ax.yaxis.set_label_coords(-0.12, 0.5)

ax.set_xlim(0, )
# ax.set_ylim(25,30)

ax.locator_params(axis='y', nbins=4)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

a = mlines.Line2D([], [], color='grey', linestyle="--", label="$s^{[s]}_{\\min}$")
b = mlines.Line2D([], [], color='grey', linestyle="-", label="$\\lambda^{[s]}_{2}$")

c = mpatches.Patch(color='green', linestyle="-", label="ER")
d = mpatches.Patch(color='purple', linestyle="-", label="SF")

plt.legend(handles=[a, b, c, d], loc='best', frameon=False, fontsize=18, ncol=2)#, bbox_to_anchor=(0, 0))

plt.savefig(f'plots/fig6-smin.pdf')
