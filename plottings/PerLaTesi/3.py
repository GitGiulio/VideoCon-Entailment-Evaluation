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
df = df.sort_values('mean_wv(F,R)',ascending=True)
df = df.reset_index(drop=True)
for index,row in df.iterrows():
    c += 1
    if c >= 23:
        b1 = (df['mean_wv(F,R)'].at[index] + df['mean_wv(F,R)'].at[index - 1] +
              df['mean_wv(F,R)'].at[index - 2] +
              df['mean_wv(F,R)'].at[index - 3] +
              df['mean_wv(F,R)'].at[index - 4] +
              df['mean_wv(F,R)'].at[index - 5] +
              df['mean_wv(F,R)'].at[index - 6] +
              df['mean_wv(F,R)'].at[index - 7] +
              df['mean_wv(F,R)'].at[index - 8] +
              df['mean_wv(F,R)'].at[index - 9] +
              df['mean_wv(F,R)'].at[index - 10] +
              df['mean_wv(F,R)'].at[index - 11] +
              df['mean_wv(F,R)'].at[index - 12] +
              df['mean_wv(F,R)'].at[index - 13] +
              df['mean_wv(F,R)'].at[index - 14] +
              df['mean_wv(F,R)'].at[index - 15] +
              df['mean_wv(F,R)'].at[index - 16] +
              df['mean_wv(F,R)'].at[index - 17] +
              df['mean_wv(F,R)'].at[index - 18] +
              df['mean_wv(F,R)'].at[index - 19] +
              df['mean_wv(F,R)'].at[index - 20] +
              df['mean_wv(F,R)'].at[index - 21] +
              df['mean_wv(F,R)'].at[index - 22]
              )
        b2 = (df['mean_wv(Vc,R)'].at[index] + df['mean_wv(Vc,R)'].at[index - 1] +
              df['mean_wv(Vc,R)'].at[index - 2] +
              df['mean_wv(Vc,R)'].at[index - 3] + df['mean_wv(Vc,R)'].at[index - 4] +
              df['mean_wv(Vc,R)'].at[index - 5] +
              df['mean_wv(Vc,R)'].at[index - 6] + df['mean_wv(Vc,R)'].at[index - 7] +
              df['mean_wv(Vc,R)'].at[index - 8] +
              df['mean_wv(Vc,R)'].at[index - 9] +
              df['mean_wv(Vc,R)'].at[index - 10] +
              df['mean_wv(Vc,R)'].at[index - 11] +
              df['mean_wv(Vc,R)'].at[index - 12] +
              df['mean_wv(Vc,R)'].at[index - 13] +
              df['mean_wv(Vc,R)'].at[index - 14] +
              df['mean_wv(Vc,R)'].at[index - 15] +
              df['mean_wv(Vc,R)'].at[index - 16] +
              df['mean_wv(Vc,R)'].at[index - 17] +
              df['mean_wv(Vc,R)'].at[index - 18] +
              df['mean_wv(Vc,R)'].at[index - 19] +
              df['mean_wv(Vc,R)'].at[index - 20] +
              df['mean_wv(Vc,R)'].at[index - 21] +
              df['mean_wv(Vc,R)'].at[index - 22]
              )
        a = pd.concat([a, pd.DataFrame([{'mean_wv(F,R)':  b1/23,'mean_wv(Vc,R)': b2/23}])], ignore_index=True)

b = pd.DataFrame()
c = 0
df = df.sort_values('mean_wv(F,S)',ascending=True)
df = df.reset_index(drop=True)
for index,row in df.iterrows():
    c += 1
    if c >= 23:
        b1 = (df['mean_wv(F,S)'].at[index] + df['mean_wv(F,S)'].at[index - 1] +
              df['mean_wv(F,S)'].at[index - 2] +
              df['mean_wv(F,S)'].at[index - 3] +
              df['mean_wv(F,S)'].at[index - 4] +
              df['mean_wv(F,S)'].at[index - 5] +
              df['mean_wv(F,S)'].at[index - 6] +
              df['mean_wv(F,S)'].at[index - 7] +
              df['mean_wv(F,S)'].at[index - 8] +
              df['mean_wv(F,S)'].at[index - 9] +
              df['mean_wv(F,S)'].at[index - 10] +
              df['mean_wv(F,S)'].at[index - 11] +
              df['mean_wv(F,S)'].at[index - 12] +
              df['mean_wv(F,S)'].at[index - 13] +
              df['mean_wv(F,S)'].at[index - 14] +
              df['mean_wv(F,S)'].at[index - 15] +
              df['mean_wv(F,S)'].at[index - 16] +
              df['mean_wv(F,S)'].at[index - 17] +
              df['mean_wv(F,S)'].at[index - 18] +
              df['mean_wv(F,S)'].at[index - 19] +
              df['mean_wv(F,S)'].at[index - 20] +
              df['mean_wv(F,S)'].at[index - 21] +
              df['mean_wv(F,S)'].at[index - 22]
              )
        b2 = (df['mean_wv(Vc,S)'].at[index] + df['mean_wv(Vc,S)'].at[index - 1] +
              df['mean_wv(Vc,S)'].at[index - 2] +
              df['mean_wv(Vc,S)'].at[index - 3] + df['mean_wv(Vc,S)'].at[index - 4] +
              df['mean_wv(Vc,S)'].at[index - 5] +
              df['mean_wv(Vc,S)'].at[index - 6] + df['mean_wv(Vc,S)'].at[index - 7] +
              df['mean_wv(Vc,S)'].at[index - 8] +
              df['mean_wv(Vc,S)'].at[index - 9] +
              df['mean_wv(Vc,S)'].at[index - 10] +
              df['mean_wv(Vc,S)'].at[index - 11] +
              df['mean_wv(Vc,S)'].at[index - 12] +
              df['mean_wv(Vc,S)'].at[index - 13] +
              df['mean_wv(Vc,S)'].at[index - 14] +
              df['mean_wv(Vc,S)'].at[index - 15] +
              df['mean_wv(Vc,S)'].at[index - 16] +
              df['mean_wv(Vc,S)'].at[index - 17] +
              df['mean_wv(Vc,S)'].at[index - 18] +
              df['mean_wv(Vc,S)'].at[index - 19] +
              df['mean_wv(Vc,S)'].at[index - 20] +
              df['mean_wv(Vc,S)'].at[index - 21] +
              df['mean_wv(Vc,S)'].at[index - 22]
              )
        b = pd.concat([b, pd.DataFrame([{'mean_wv(F,S)':  b1/23,'mean_wv(Vc,S)': b2/23}])], ignore_index=True)
size = 1
transparency = 1  # df[y]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 12})

fig.suptitle('Simple conditional generation trends', fontsize=15)

ax[0].scatter(df['mean_wv(F,R)'], df['mean_wv(Vc,R)'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[0].scatter(a['mean_wv(F,R)'], a['mean_wv(Vc,R)'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered data')
ax[0].set_xlabel('models_average($F,T_R$)')
ax[0].set_ylabel('models_average($V_S^C,T_R$)')
ax[0].set_title('Real caption')
#ax[0].set_xlim(xlim)
#ax[0].set_ylim(ylim)
z = np.polyfit(df['mean_wv(F,R)'], df['mean_wv(Vc,R)'], 1)
p = np.poly1d(z)
ax[0].plot(df['mean_wv(F,R)'], p(df['mean_wv(F,R)']), "r-",linewidth=0.7, label='Trend line')

ax[1].scatter(df['mean_wv(F,S)'], df['mean_wv(Vc,S)'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[1].scatter(b['mean_wv(F,S)'], b['mean_wv(Vc,S)'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered data')
ax[1].set_xlabel('models_average($F,T_S$)')
ax[1].set_ylabel('models_average($V_S^C,T_S$)')
ax[1].set_title('Synthetic caption')
#ax[1].set_xlim(xlim)
#ax[1].set_ylim(ylim)
z = np.polyfit(df['mean_wv(F,S)'], df['mean_wv(Vc,S)'], 1)
p = np.poly1d(z)
xs = np.linspace(0, 1, 10000)

ax[1].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')
fig.set_tight_layout(tight=True)

plt.legend(markerscale=8, ncol=1, loc=4)

plt.savefig(f'plots/P3_basic_relations.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()