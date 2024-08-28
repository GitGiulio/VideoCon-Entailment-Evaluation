import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

df = df.sort_values('D(mean_wv(F,R),mean_wv(F,S))', ascending=True)
df = df.reset_index(drop=True)
pccT, _ = pearsonr(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
print(pccT)
a = pd.DataFrame()
b = pd.DataFrame()
for i in np.arange(0,1,0.05):
    for index, row in df.iterrows():
        if row['D(mean_wv(F,R),mean_wv(F,S))'] < i:
            a = pd.concat([a, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))': row['D(mean_wv(F,R),mean_wv(F,S))'],
                                             'D(mean_wv(Vu,S),mean_wv(Vc,S))': row[
                                                 'D(mean_wv(Vu,S),mean_wv(Vc,S))']}])], ignore_index=True)
        else:
            b = pd.concat([b, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))': row['D(mean_wv(F,R),mean_wv(F,S))'],
                                             'D(mean_wv(Vu,S),mean_wv(Vc,S))': row[
                                                 'D(mean_wv(Vu,S),mean_wv(Vc,S))']}])], ignore_index=True)
    pccA, _ = pearsonr(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    pccB, _ = pearsonr(b['D(mean_wv(F,R),mean_wv(F,S))'], b['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    print(f'A(-1,{i}): {pccA} | B({i},1): {pccB} | DIFF = {(pccB-pccA)}')

    sccA, _ = spearmanr(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    sccB, _ = spearmanr(b['D(mean_wv(F,R),mean_wv(F,S))'], b['D(mean_wv(Vu,S),mean_wv(Vc,S))'])
    print(f'A(-1,{i}): {sccA} | B({i},1): {sccB} | DIFF = {(sccB-sccA)}')

c = 0
d = 0
print(a.shape)
print(b.shape)
print(df.shape)
for index, row in df.iterrows():
    if row['D(mean_wv(Vc,R),mean_wv(Vc,S))'] < 0.0:
        c += 1
    else:
        d += 1
print(c,d)