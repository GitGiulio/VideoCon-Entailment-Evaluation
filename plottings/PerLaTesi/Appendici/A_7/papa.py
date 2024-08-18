import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'scatter'])

df = pd.read_csv('../../with_mean.csv')

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

a = pd.DataFrame()
for index,row in df.iterrows():
    if row['D(mean_wv(F,R),mean_wv(F,S))'] > 0:
        a = pd.concat([a, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))':row['D(mean_wv(F,R),mean_wv(F,S))'],
                                         'D(mean_wv(Vu,S),mean_wv(Vc,S))':row['D(mean_wv(Vu,S),mean_wv(Vc,S))']}])], ignore_index=True)

size = 0.6
transparency = 1

plt.rcParams.update({'font.size': 5})

plt.title('Unconditioned - conditioned trend', fontsize=7)

plt.scatter(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vu,S),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency)
plt.xlabel('models_average(F,$T_R$) - models_average(F,$T_S$)')
plt.ylabel('models_average($V_S^U$,$T_S$) - models_average($V_S^C$,$T_S$)')
plt.xlim([0,1])
plt.ylim([-1,1])
z = np.polyfit(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vu,S),mean_wv(Vc,S))'], 2)
p = np.poly1d(z)
xs = np.linspace(0, 1, 10000)

plt.plot(xs, p(xs), 'r-',linewidth=0.6, label='Trend line')

z = np.polyfit(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vu,S),mean_wv(Vc,S))'], 1)
p = np.poly1d(z)
plt.plot(xs, p(xs), 'r-',linewidth=0.6, label='linear regression')


plt.legend(['models mean','Trend line'], markerscale=5, ncol=1,loc=3)

plt.savefig(f'P7_papa.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()