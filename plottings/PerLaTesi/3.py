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

size = 0.5
transparency = 1  # df[y]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 7})

fig.suptitle('Simple conditional generation trends', fontsize=9)

ax[0].scatter(df['mean_wv(F,R)'], df['mean_wv(Vc,R)'], c='#7d5f8d',marker='.', s=size, alpha=transparency)
ax[0].set_xlabel('models_average(F,R)')
ax[0].set_ylabel('models_average(Vc,R)')
#ax[0].set_xlim(xlim)
#ax[0].set_ylim(ylim)
z = np.polyfit(df['mean_wv(F,R)'], df['mean_wv(Vc,R)'], 1)
p = np.poly1d(z)
ax[0].plot(df['mean_wv(F,R)'], p(df['mean_wv(F,R)']), "r--",linewidth=0.7, label='Trend line')

ax[1].scatter(df['mean_wv(F,S)'], df['mean_wv(Vc,S)'], c='#7d5f8d',marker='.', s=size, alpha=transparency)
ax[1].set_xlabel('models_average(F,S)')
ax[1].set_ylabel('models_average(Vc,S)')
#ax[1].set_xlim(xlim)
#ax[1].set_ylim(ylim)
z = np.polyfit(df['mean_wv(F,S)'], df['mean_wv(Vc,S)'], 1)
p = np.poly1d(z)
ax[1].plot(df['mean_wv(F,S)'], p(df['mean_wv(F,S)']), "r--",linewidth=0.7, label='Trend line')
fig.set_tight_layout(tight=True)


plt.legend(['models mean','Trend line'], markerscale=15, ncol=1, bbox_to_anchor=(1, 1))

plt.savefig(f'plots/P3.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()