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

a = pd.DataFrame()
c = 0
df = df.sort_values('D(mean_wv(F,R),mean_wv(F,S))',ascending=True)
df = df.reset_index(drop=True)
for index,row in df.iterrows():
    c += 1
    if c >= 23:
        b1 = (df['D(mean_wv(F,R),mean_wv(F,S))'].at[index] + df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 1] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 2] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 3] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 4] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 5] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 6] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 7] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 8] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 9] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 10] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 11] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 12] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 13] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 14] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 15] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 16] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 17] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 18] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 19] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 20] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 21] +
              df['D(mean_wv(F,R),mean_wv(F,S))'].at[index - 22]
              )
        b2 = (df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index] + df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 1] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 2] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 3] + df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 4] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 5] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 6] + df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 7] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 8] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 9] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 10] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 11] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 12] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 13] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 14] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 15] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 16] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 17] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 18] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 19] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 20] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 21] +
              df['D(mean_wv(Vc,R),mean_wv(Vc,S))'].at[index - 22]
              )
        a = pd.concat([a, pd.DataFrame([{'D(mean_wv(F,R),mean_wv(F,S))':  b1/23,'D(mean_wv(Vc,R),mean_wv(Vc,S))': b2/23}])], ignore_index=True)

size = 0.3
transparency = 1

plt.rcParams.update({'font.size': 5})

plt.title('Difference induces difference', fontsize=7)

plt.scatter(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models average raw')
plt.scatter(a['D(mean_wv(F,R),mean_wv(F,S))'], a['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered')
plt.xlabel('models_average($F,T_R$) - models_average($F,T_S$)')
plt.ylabel('models_average($V_S^C,T_R$) - models_average($V_S^C,T_S$)')
plt.xlim([-1,1])
plt.ylim([-1,1])
z = np.polyfit(df['D(mean_wv(F,R),mean_wv(F,S))'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], 1)
p = np.poly1d(z)
xs = np.linspace(-1, 1, 10000)

plt.plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')

plt.legend( markerscale=10, ncol=1, bbox_to_anchor=(1, 1))

plt.savefig(f'plots/P5_effect_of_difference.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()