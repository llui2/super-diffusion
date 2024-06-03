import numpy as np
import pandas as pd

input = "data/ER-ER-N500-D50-S1-R1"

df = pd.read_csv(f'{input}.csv')

k1s = df['k1'].to_numpy()
k2s = df['k2'].to_numpy()

xticks = [5,25,50,75,100]
yticks = [5,25,50,75,100]
#--------------------------------------------
bin_size = 2

k1_min = xticks[0]
k1_max = xticks[-1] + bin_size
k2_min = yticks[0]
k2_max = yticks[-1] + bin_size
k1s = np.arange(k1_min, k1_max, bin_size)
k2s = np.arange(k2_min, k2_max, bin_size)

data = np.zeros((len(k1s), len(k2s)))
data2 = np.zeros((len(k1s), len(k2s)))

num_rep = np.zeros((len(k1s), len(k2s)))

for i in range(len(k1s)):
    k1_min = k1s[i]
    k1_max = k1_min + bin_size
    for j in range(len(k2s)):
        k2_min = k2s[j]
        k2_max = k2_min + bin_size
        mask = (df['k1_avg'] >= k1_min) & (df['k1_avg'] < k1_max) & (df['k2_avg'] >= k2_min) & (df['k2_avg'] < k2_max)
        
        value = 'prob_avg' # prob_avg correct
        data[i, j] = df.loc[mask, value].sum() 

        num_rep[i, j] = mask.sum()

num_rep[num_rep == 0] = 1

data /= num_rep
#--------------------------------------------
# compute the ratio of value
print("input = ", input)
ratio = data.sum() / (data.shape[0] * data.shape[1])
print(f"ratio {value} = ", round(ratio, 3))