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

b = pd.DataFrame()
for index,row in df.iterrows():
    if row['D(mean_wv(Vc,R),mean_wv(Vc,S))'] > 0:
        b = pd.concat([b, pd.DataFrame([{'mean_wv(F,S)':row['mean_wv(F,S)'],'mean_wv(F,R)':row['mean_wv(F,R)'],'D(mean_wv(Vc,R),mean_wv(Vc,S))':row['D(mean_wv(Vc,R),mean_wv(Vc,S))']}])], ignore_index=True)

def filter(df,val,x,y):
    a = pd.DataFrame()
    c = 0
    df = df.sort_values(x,ascending=True)
    df = df.reset_index(drop=True)
    for index,row in df.iterrows():
        c += 1
        if c >= val:
            b1 = 0
            b2 = 0
            for i in range(val):
                b1 += df[x].at[index-i]
                b2 += df[y].at[index-i]
            a = pd.concat([a, pd.DataFrame([{x:  b1/val,y: b2/val}])],ignore_index=True)
    return a

a = filter(b,23,'mean_wv(F,R)','D(mean_wv(Vc,R),mean_wv(Vc,S))')
c = filter(b,23,'mean_wv(F,S)','D(mean_wv(Vc,R),mean_wv(Vc,S))')
size = 1
transparency = 1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 12})

fig.suptitle('Mirrored trend when ordered by $T_S$', fontsize=15)

ax[0].scatter(df['mean_wv(F,R)'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[0].scatter(a['mean_wv(F,R)'], a['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered data')
ax[0].set_xlabel('models_average($F,T_R$)')
ax[0].set_ylabel('models_average($V_S^C,T_R$) - models_average($V_S^C,T_S$)')
ax[0].set_title('ordered by alignment with $T_R$')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,1])
z = np.polyfit(b['mean_wv(F,R)'], b['D(mean_wv(Vc,R),mean_wv(Vc,S))'], 1)
p = np.poly1d(z)
xs = np.linspace(0, 1, 10000)
ax[0].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')


ax[1].scatter(df['mean_wv(F,S)'], df['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#7d5f8d',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[1].scatter(c['mean_wv(F,S)'], c['D(mean_wv(Vc,R),mean_wv(Vc,S))'], c='#00D7D7',marker='.', s=size, alpha=transparency, label='filtered data')
ax[1].set_xlabel('models_average($F,T_S$)')
ax[1].set_ylabel('models_average($V_S^C,T_R$) - models_average($V_S^C,T_S$)')
ax[1].set_title('ordered by alignment with $T_S$')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,1])
z = np.polyfit(b['mean_wv(F,S)'], b['D(mean_wv(Vc,R),mean_wv(Vc,S))'], 1)
p = np.poly1d(z)
ax[1].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')


fig.set_tight_layout(tight=True)

#plt.legend(markerscale=8, ncol=1, loc=4)

plt.savefig(f'D3_mirrored_trend.png', dpi=300)
plt.clf()
