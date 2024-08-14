import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'scatter'])

df = pd.read_csv('../../data/complete_df.csv')

for round in [1,2,4,5]:
    df[f'D(clip_flant(Vc_{round},R),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vc_{round},R)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vu_{round},S))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vu_{round},S)']
    df[f'D(clip_flant(Vu_{round},S),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vu_{round},S)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vc_{round},R))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vc_{round},R)']
    df[f'D(llava(Vc_{round},R),llava(Vc_{round},S))'] = df[f'llava(Vc_{round},R)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vu_{round},S))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vu_{round},S)']
    df[f'D(llava(Vu_{round},S),llava(Vc_{round},S))'] = df[f'llava(Vu_{round},S)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vc_{round},R))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vc_{round},R)']
    df[f'D(instructblip(Vc_{round},R),instructblip(Vc_{round},S))'] = df[f'instructblip(Vc_{round},R)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vu_{round},S))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vu_{round},S)']
    df[f'D(instructblip(Vu_{round},S),instructblip(Vc_{round},S))'] = df[f'instructblip(Vu_{round},S)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vc_{round},R))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vc_{round},R)']

df[f'D(mean(Vc,R),mean(Vc,S))'] = df[f'mean(Vc,R)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vu,S))'] = df[f'mean(Vu,R)'] - df[f'mean(Vu,S)']
df[f'D(mean(Vu,S),mean(Vc,S))'] = df[f'mean(Vu,S)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vc,R))'] = df[f'mean(Vu,R)'] - df[f'mean(Vc,R)']

size = 0.6
transparency = 1

plt.rcParams.update({'font.size': 9})
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

fig.suptitle('Unconditional and conditional comparison',fontsize=11)

ax[0].scatter(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#f79410',marker='.', s=size, alpha=transparency)
ax[0].set_xlabel('models_average(F,$T_R$) - models_average(F,$T_S$)')
ax[0].set_ylabel('models_average($V_S^C$,$T_R$) - models_average($V_S^C$,$T_S$)')
ax[0].set_title('Conditioned')
#ax[0].set_xlim(xlim)
#ax[0].set_ylim(ylim)

ax[1].scatter(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vu,R),mean_wv(Vu,S))'], c='#145d9e',marker='.', s=size, alpha=transparency)
ax[1].set_xlabel('models_average(F,$T_R$) - models_average(F,$T_S$)')
ax[1].set_ylabel('models_average($V_S^U$,$T_R$) - models_average($V_S^U$,$T_S$)')
ax[1].set_title('Unconditioned')
#ax[1].set_xlim(xlim)
#ax[1].set_ylim(ylim)
fig.set_tight_layout(tight=True)


plt.savefig(f'plots/P6_cond_uncon_comparison.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()