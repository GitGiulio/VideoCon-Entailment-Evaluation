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

b = pd.DataFrame()
for index,row in df.iterrows():
    if row['D(mean_wv(Vc,R),mean_wv(Vc,S))'] > 0:
        b = pd.concat([b, pd.DataFrame([{'mean_wv(F,R)':row['mean_wv(F,R)'],'D(mean_wv(Vc,R),mean_wv(Vc,S))':row['D(mean_wv(Vc,R),mean_wv(Vc,S))']}])], ignore_index=True)


a = pd.DataFrame()
c = 0
b = b.sort_values('mean_wv(F,R)',ascending=True)
b = b.reset_index(drop=True)
for index,row in b.iterrows():
    c += 1
    if c >= 23:
        b1 = (b['mean_wv(F,R)'].at[index] + b['mean_wv(F,R)'].at[index - 1] +
              b['mean_wv(F,R)'].at[index - 2] +
              b['mean_wv(F,R)'].at[index - 3] +
              b['mean_wv(F,R)'].at[index - 4] +
              b['mean_wv(F,R)'].at[index - 5] +
              b['mean_wv(F,R)'].at[index - 6] +
              b['mean_wv(F,R)'].at[index - 7] +
              b['mean_wv(F,R)'].at[index - 8] +
              b['mean_wv(F,R)'].at[index - 9] +
              b['mean_wv(F,R)'].at[index - 10] +
              b['mean_wv(F,R)'].at[index - 11] +
              b['mean_wv(F,R)'].at[index - 12] +
              b['mean_wv(F,R)'].at[index - 13] +
              b['mean_wv(F,R)'].at[index - 14] +
              b['mean_wv(F,R)'].at[index - 15] +
              b['mean_wv(F,R)'].at[index - 16] +
              b['mean_wv(F,R)'].at[index - 17] +
              b['mean_wv(F,R)'].at[index - 18] +
              b['mean_wv(F,R)'].at[index - 19] +
              b['mean_wv(F,R)'].at[index - 20] +
              b['mean_wv(F,R)'].at[index - 21] +
              b['mean_wv(F,R)'].at[index - 22]
              )
        b2 = (b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index] + b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 1] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 2] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 3] + b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 4] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 5] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 6] + b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 7] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 8] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 9] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 10] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 11] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 12] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 13] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 14] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 15] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 16] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 17] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 18] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 19] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 20] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 21] +
              b['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 22]
              )
        a = pd.concat([a, pd.DataFrame([{'mean_wv(F,R)':  b1/23,'D(mean_wv(Vc,R),mean_wv(Vc,S))': b2/23}])], ignore_index=True)
size = 0.3
transparency = 1  # df[y]

plt.rcParams.update({'font.size': 5})

plt.title('Relative degradation of conditioned videos', fontsize=7)

plt.scatter(df['mean_wv(F,R)'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models average')
plt.scatter(a['mean_wv(F,R)'], a['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered')
plt.xlabel('models_average($F,T_R$)')
plt.ylabel('models_average($V_S^C,T_R$) - models_average($V_S^C,T_S$)')
plt.xlim([0,1])
plt.ylim([0,1])

z = np.polyfit(b['mean_wv(F,R)'], b['D(mean_wv(Vc,R),mean_wv(Vc,S))'], 1)
p = np.poly1d(z)
xs = np.linspace(0, 1, 10000)

plt.plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')

plt.legend(markerscale=10, ncol=1, loc=1)

plt.savefig(f'plots/P4_cond_video_degrading.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()