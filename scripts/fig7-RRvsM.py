import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times', size=10)
plt.rc('mathtext', fontset='cm')

fig = plt.figure(figsize=(3, 3))
ax = plt.subplot()

plt.subplots_adjust(wspace=0, hspace=0, left=0.17,
                    top=0.95, right=0.95, bottom=0.15)
ax.tick_params(direction='in', top=True, right=True)

# --- plot

k = np.arange(25, 60, 1)

# --- theoretical

def li_FC(N, k):
       return N * k / (N - 1)

def li_RR(N, k, l):
       return k - 2 * np.sqrt(k - 1) / np.sqrt(l)

# ---

c = ['blue', 'green', 'orange', 'red']
c = ['r' for i in range(4)]

N = 1000

# FC
l0_t = li_FC(N, k)
ax.plot(k, l0_t, color='black', marker=' ', markersize=1, linestyle='--', linewidth=1, label='$\\lambda_2^{\\mathrm{FC}}$')

# w = 1
l1_t = li_RR(N, k, 1)
ax.plot(k, l1_t, color=c[0], marker=' ', markersize=1, linestyle=':', linewidth=1, label='$M = 1$')

# w = 0.5
l1_t = li_RR(N, k, 2)
ax.plot(k, l1_t, color=c[1], marker=' ', markersize=1, linestyle='-.', linewidth=1, label='$M = 2$')

# w = 0.5
l1_t = li_RR(N, k, 10)
ax.plot(k, l1_t, color=c[2], marker=' ', markersize=1, linestyle='--', linewidth=1, label='$M = 10$')

# w = 0.1
l1_t = li_RR(N, k, 100)
ax.plot(k, l1_t, color=c[3], marker=' ', markersize=1, linestyle='-', linewidth=1, label='$M = 100$')

# --- lable + ticks + legend

ax.set_xlabel('$s^{[s]}$', fontfamily='Times')
ax.set_ylabel('$\\frac{1}{M} \\lambda^{\\mathrm{RR}}_2(M s^{[s]})$', fontfamily='Times')

ax.set_xlim(25, 55)
ax.set_ylim(10, 60)

ax.set_xticks(np.arange(25, 60, step=10))
ax.set_yticks(np.arange(15, 65, step=20))

leg = plt.legend(loc='best', fancybox=False, shadow=False, frameon=False, ncol=1, fontsize="8", labelspacing=0.6)
plt.setp(leg.get_lines(), linewidth=.8)

plt.savefig(f'plots/fig7-RRvsM.pdf')
