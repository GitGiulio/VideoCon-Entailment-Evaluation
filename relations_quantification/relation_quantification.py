import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import re

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

df['D(mean_wv(F,R),mean_wv(F,S))'] = df['ff_cap_entail_mean']-df['ff_neg_cap_entail_mean']
df['D(mean_wv(Vc,R),mean_wv(Vc,S))'] = df['video conditioned real entail mean_wv']-df['video conditioned synth entail mean_wv']

cov = cov(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'])
print(f'cov(D(mean_wv(F,R),mean_wv(F,S)),D(mean_wv(Vc,R),mean_wv(Vc,S)) = {cov}')

pcc, _ = pearsonr(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'])
print('Pearsons correlation: %.13f' % pcc)

scc, _ = spearmanr(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'])
print('Spearmans correlation: %.13f' % scc)

for x,y in zip(a,b):

    cov = cov(df[x],df[y])
    print(f'cov({x},{y}) = {cov}')

    pcc, _ = pearsonr(df[x],df[y])
    print('Pearsons correlation: %.13f' % pcc)

    scc, _ = spearmanr(df[x],df[y])
    print('Spearmans correlation: %.13f' % scc)
