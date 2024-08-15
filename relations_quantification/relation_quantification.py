import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import spearmanr

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')
"""
df = df.rename(columns={"ff_cap_clip_flant": "clip_flant(F,R)",
                   "ff_cap_instructblip_flant": "instructblip(F,R)",
                   "ff_cap_llava": "llava(F,R)",
                   "ff_cap_entail_mean": "mean_wv(F,R)",
                   "ff_neg_cap_clip_flant": "clip_flant(F,S)",
                   "ff_neg_cap_instructblip_flant": "instructblip(F,S)",
                   "ff_neg_cap_llava": "llava(F,S)",
                   "ff_neg_cap_entail_mean": "mean_wv(F,S)",
                   "ff_entail_diff_clip_flant": "D(clip_flant(F,R),clip_flant(F,S))",
                   "ff_entail_diff_instructblip_flant": "D(instructblip(F,R),instructblip(F,S))",
                   "ff_entail_diff_llava": "D(llava(F,R),llava(F,S))",
                   "ff_entail_diff_mean": "D(mean_wv(F,R),mean_wv(F,S))",
                   })

for type in ['conditioned','unconditioned']:
    for r in [1,2,3,4,5]:
        for caption in ['synth','real']:
            new_type = 'a'
            if type == 'conditioned':
                new_type = 'c'
            elif type == 'unconditioned':
                new_type = 'u'
            else:
                print('AAAAA')
                exit(1)
            new_caption = 'a'
            if caption == 'synth':
                new_caption = 'S'
            elif caption == 'real':
                new_caption = 'R'
            else:
                print('AAAAA')
                exit(1)

            if r == 1:
                df = df.rename(columns={f"video {type} {caption} entail mean_wv": f"mean_wv(V{new_type},{new_caption})",
                                   f"video {type} {caption} entail mean": f"mean(V{new_type},{new_caption})",
                                   f"videocon_{caption} {type} r{r} ent": f"videocon(V{new_type}_{r},{new_caption})",
                                   f"clip_flant {caption} {type} r{r} ent": f"clip_flant(V{new_type}_{r},{new_caption})",
                                   f"instructblip_flant {caption} {type} r{r} ent": f"instructblip(V{new_type}_{r},{new_caption})",
                                   f"llava {caption} {type} r{r} ent": f"llava(V{new_type}_{r},{new_caption})",
                                   })
            else:
                df = df.rename(columns={f"videocon_{caption} {type} r{r} ent": f"videocon(V{new_type}_{r},{new_caption})",
                                   f"clip_flant {caption} {type} r{r} ent": f"clip_flant(V{new_type}_{r},{new_caption})",
                                   f"instructblip_flant {caption} {type} r{r} ent": f"instructblip(V{new_type}_{r},{new_caption})",
                                   f"llava {caption} {type} r{r} ent": f"llava(V{new_type}_{r},{new_caption})",
                                   })

df['D(mean_wv(F,R),mean_wv(F,S))'] = df['mean_wv(F,R)'] - df['mean_wv(F,S)']
df['D(mean_wv(Vc,R),mean_wv(Vc,S))'] = df['mean_wv(Vc,R)'] - df['mean_wv(Vc,S)']
df['D(mean_wv(Vu,R),mean_wv(Vu,S))'] = df['mean_wv(Vu,R)'] - df['mean_wv(Vu,S)']
df['D(mean_wv(Vu,S),mean_wv(Vc,S))'] = df['mean_wv(Vu,S)'] - df['mean_wv(Vc,S)']
df['D(mean_wv(Vu,R),mean_wv(Vc,R))'] = df['mean_wv(Vu,R)'] - df['mean_wv(Vc,R)']
"""


Covariance = []
Correlation = []
Pearsonr = []
Spearmanr = []

ouf = open('relations_quantification.txt', 'w')
X = ['mean_wv(F,S)','mean_wv(F,R)','D(mean_wv(F,R),mean_wv(F,S))']
Y = ['mean_wv(Vc,S)','mean_wv(Vc,R)','mean_wv(Vu,R)','mean_wv(Vu,S)',
     'D(mean_wv(Vc,R),mean_wv(Vc,S))','D(mean_wv(Vu,R),mean_wv(Vu,S))','D(mean_wv(Vu,R),mean_wv(Vc,R))','D(mean_wv(Vu,S),mean_wv(Vc,S))']
for x in X:
    for y in Y:
        print(f'{x}  |  {y}')

        ouf.write(f'{x}  |  {y}\n')

        covariance = cov(df[x], df[y])
        Covariance.append(covariance[0][1])
        print(f'covariance = {covariance[0][1]}')
        ouf.write(f'covariance = {covariance[0][1]} | ')
        """
            correlation = np.corrcoef(df[x], df[y])
            Correlation.append(correlation[0][1])
            print(f'correlation = {correlation[0][1]}')
            ouf.write(f'correlation = {correlation[0][1]} | ')
        """
        pcc, p = pearsonr(df[x], df[y])
        Pearsonr.append(pcc)
        print('Pearsons correlation: %.13f' % pcc)
        ouf.write('Pearsons correlation: %.13f | ' % pcc)
        print('p-value: %.13f' % p)
        ouf.write('p-value: %.13f | ' % p)

        scc, p = spearmanr(df[x], df[y])
        Spearmanr.append(scc)
        print('Spearmans correlation: %.13f ' % scc)
        ouf.write('Spearmans correlation: %.13f | ' % scc)
        print('p-value: %.13f\n' % p)
        ouf.write('p-value: %.13f\n' % p)

ouf.close()
# df.to_csv('C:\\Users\giuli\PycharmProjects\VideoCon-Entailment-Evaluation\data\complete_df.csv')
