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

size = 0.3
transparency = 1  # df[y]

plt.rcParams.update({'font.size': 5})

plt.title('Relative degradation of conditioned videos', fontsize=7)

plt.scatter(df['mean_wv(F,R)'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency)
plt.xlabel('models_average($F,T_R$)')
plt.ylabel('models_average($V_S^C,T_R$) - models_average($V_S^C,T_S$)')
plt.xlim([0,1])
plt.ylim([0,1])
a = pd.DataFrame([{'temp x':0,'temp y':0}])
for index,row in df.iterrows():
    if row['D(mean_wv(Vc,R),mean_wv(Vc,S))'] > 0:
        a = pd.concat([a, pd.DataFrame([{'temp x':row['mean_wv(F,R)'],'temp y':row['D(mean_wv(Vc,R),mean_wv(Vc,S))']}])], ignore_index=True)

z = np.polyfit(a['temp x'], a['temp y'], 1)
p = np.poly1d(z)
plt.plot(a['temp x'], p(a['temp x']), "r-",linewidth=0.7, label='Trend line')

plt.legend(['models average','Trend line'], markerscale=10, ncol=1, bbox_to_anchor=(1, 1))

plt.savefig(f'plots/P4_cond_video_degrading.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()