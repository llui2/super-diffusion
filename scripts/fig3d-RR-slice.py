import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=18)
plt.rc('mathtext', fontset='cm')

df = pd.read_csv('data/RR-RR-N1000-D50-S0-R0.csv')

df = df[df['k1'] == 40]

k1 = df['k1'].to_numpy()
k2 = df['k2'].to_numpy()
ka = df['ka_avg'].to_numpy()

l1 = df['λ1_avg'].to_numpy()
l2 = df['λ2_avg'].to_numpy()
la = df['λa_avg'].to_numpy()

e_l1 = df['σ_λ1'].to_numpy()
e_l2 = df['σ_λ2'].to_numpy()
e_la = df['σ_λa'].to_numpy()

c = ['green', 'blue', 'red']

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot()

plt.subplots_adjust(wspace=0, hspace=0, left=0.15,
                    top=0.95, right=0.95, bottom=0.15)
ax.tick_params(direction='in', top=True, right=True)

# --- plot

ax.plot(k2, l1, color=c[0], marker='o', markersize=1,
        linewidth=1)
ax.fill_between(k2, l1 + e_l1, l1 - e_l1, color=c[0], alpha=0.2)

ax.plot(k2, l2, color=c[1], marker='o', markersize=1,
        linewidth=1)
ax.fill_between(k2, l2 + e_l2, l2 - e_l2, color=c[1], alpha=0.2)

ax.plot(k2, la, color=c[2], marker='o', markersize=1,
        linewidth=1)
ax.fill_between(k2, la + e_la, la - e_la, color=c[2], alpha=0.2)

# --- theoretical prediction

def li_RR(N, k, l):
    k = k * l
    l_t = k - 2 * np.sqrt(k - 1)
    return l_t / l

N = 1000

l1_t = li_RR(N, k1, 1)
ax.plot(k2, l1_t, color=c[0], marker=' ', markersize=1, linestyle='--', linewidth=1, label='$\\lambda_2^{\\mathrm{RR}}(k^{[1]})$')

l1_t = li_RR(N, k2, 1)
ax.plot(k2, l1_t, color=c[1], marker=' ', markersize=1, linestyle='--', linewidth=1, label='$\\lambda_2^{\\mathrm{RR}}(k^{[2]})$')

la_t = li_RR(N, ka, 2)
ax.plot(k2, la_t, color=c[2], marker=' ', markersize=1, linestyle='--', linewidth=1, label='$\\frac{1}{2} \\lambda_2^{\\mathrm{RR}}(2s^{[s]})$')

# --- lable + ticks + legend

ax.set_xlabel('$k^{[2]}$', fontfamily='Times')
ax.set_ylabel('$\\langle \\lambda^{[\\alpha]}_2 \\rangle$', fontfamily='Times')

ax.set_xlim(20, 60)
ax.set_ylim(10, 45)

ax.set_xticks(np.arange(25, 60, step=5))
ax.set_yticks(np.arange(15, 45, step=5))

leg = plt.legend(loc='upper left', fancybox=False, shadow=False, frameon=False, ncol=1, fontsize="15", labelspacing=.6)
plt.setp(leg.get_lines(), linewidth=1.2)

ax.text(-0.15, 0.85, "(d)", fontsize=20, ha='center', va='center', transform=ax.transAxes)

plt.savefig(f'plots/fig3d-RR-slice.pdf')