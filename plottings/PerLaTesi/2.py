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

# if model == 'clip_flant':
#     color = '#f13511'
# elif model == 'llava':
#     color = '#f79410'
# elif model == 'instructblip_flant':
#     color = '#0cb14d'
# elif model == 'videocon':
#     color = '#145d9e'
# elif model == 'models_mean':
#     color = '#4a4a4a'
# elif model == 'mean_without_videocon':
#     color = '#7d5f8d'
size = 0.15
transparency = 1  # df[y]
# for i in range(len(transparency)):
#    transparency[i] = min(abs(transparency[i])+0.1, 1)

plt.rcParams.update({'font.size': 5})
plt.title('DIFF(mean_wv(F,R),mean_wv(F,S)) vs DIFF(mean_wv(Vc,R),mean_wv(Vc,S))',fontsize=7)
plt.scatter(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], s=size,c='#00D7D7',marker='.', alpha=transparency)

z = np.polyfit(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], 1)  # TODO polinomial fit for trends HOOOOOOOO
p = np.poly1d(z)
plt.plot(df['D(mean_wv(F,R),mean_wv(F,S))'], p(df['D(mean_wv(F,R),mean_wv(F,S))']), "r--",linewidth=0.5, label='Trend line')

plt.legend(['models mean','Trend line'], markerscale=10, ncol=1, bbox_to_anchor=(1, 1))

plt.savefig(f'plots/P2.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()