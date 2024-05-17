import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rc('font', family='Times', size=15)
plt.rc('mathtext', fontset='cm')

df = pd.read_csv('tuning.csv')

df_info = pd.read_csv('tuning-info.csv')

i_best = df_info['i_best'].to_numpy()
T_best = df_info['T_best'].to_numpy()[0]

num_cols = (len(df.columns) - 1)//2

t = df['t'].to_numpy()

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot()

plt.subplots_adjust(wspace=0, hspace=0, left=0.15,
                    top=0.9, right=0.95, bottom=0.15)
ax.tick_params(direction='in', top=False, right=False)
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(1.2)
ax.tick_params(width=1.2)

# --- plot

for i in range(1, num_cols+1):
    
    if i == i_best:
        a = 1
    else:
        a = 0.2
    
    l = df[f'λ{i}'].to_numpy()
    ax.step(t, l, '-', linewidth=1.5, color="purple", alpha=a)

# --- lable + ticks

ax.set_xlabel('$t$', fontfamily='Times', fontsize=18)
ax.set_ylabel('$\\lambda_2$', fontfamily='Times', fontsize=18)

ax.locator_params(axis='y', nbins=4)

# ax.set_xscale('log')
# ax.tick_params(axis='x', which='minor', direction='in')
# ax.xaxis.set_tick_params(which='both', pad=8)

formatted_string = "{:.0e}".format(T_best)
formatted_string = formatted_string.replace('e', ' \\times 10^{').replace('-0', '-') + '}'
text = '$T^{\\mathrm{initial}}_{\\mathrm{best}} ' + f'= {formatted_string}$'
ax.text(0.2, 0.2, text, fontsize=15, ha='center', va='center', transform=ax.transAxes)

ax.set_xlim(0, )

plt.savefig(f'finetuning.pdf')
