import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fontTools.misc.cython import returns
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

MAX = -9
MAX_i = -2
df = df.sort_values('D(mean_wv(F,R),mean_wv(F,S))', ascending=True)
df = df.reset_index(drop=True)
pccT, _ = pearsonr(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
print(pccT)
for i in np.arange(-0.5,1,0.05):
    a = pd.DataFrame()
    b = pd.DataFrame()
    for index, row in df.iterrows():
        if row['D(mean_wv(F,R),mean_wv(F,S))'] < i:
            a = pd.concat([a, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))': row['D(mean_wv(F,R),mean_wv(F,S))'],
                                             'D(mean_wv(Vu,S),mean_wv(Vc,S))': row[
                                                 'D(mean_wv(Vu,S),mean_wv(Vc,S))']}])], ignore_index=True)
        else:
            b = pd.concat([a, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))': row['D(mean_wv(F,R),mean_wv(F,S))'],
                                             'D(mean_wv(Vu,S),mean_wv(Vc,S))': row[
                                                 'D(mean_wv(Vu,S),mean_wv(Vc,S))']}])], ignore_index=True)
    if b.shape == (0,0):
        continue

    pccA, _ = pearsonr(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    pccB, _ = pearsonr(b['D(mean_wv(F,R),mean_wv(F,S))'], b['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    DIFF = pccB-pccA
    if DIFF > MAX:
        MAX = DIFF
        MAX_i = i
    print(f'A(-1,{i}): {pccA} | B({i},1): {pccB} | DIFF = {(pccB-pccA)}')


print(MAX, MAX_i)

