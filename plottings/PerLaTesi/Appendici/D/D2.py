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

df[f'D(clip_flant(Vc_mean,R),clip_flant(Vc_mean,S))'] = df[f'clip_flant(Vc_mean,R)'] - df[f'clip_flant(Vc_mean,S)']
df[f'D(llava(Vc_mean,R),llava(Vc_mean,S))'] = df[f'llava(Vc_mean,R)'] - df[f'llava(Vc_mean,S)']
df[f'D(instructblip(Vc_mean,R),instructblip(Vc_mean,S))'] = df[f'instructblip(Vc_mean,R)'] - df[f'instructblip(Vc_mean,S)']

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

a = filter(df,23,'clip_flant(F,R)','D(clip_flant(Vc_mean,R),clip_flant(Vc_mean,S))')
b = filter(df,23,'llava(F,R)','D(llava(Vc_mean,R),llava(Vc_mean,S))')
c = filter(df,23,'instructblip(F,R)','D(instructblip(Vc_mean,R),instructblip(Vc_mean,S))')
size = 1
transparency = 1  # df[y]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size': 12})

fig.suptitle('Models comparison', fontsize=15)

ax[0].scatter(df['clip_flant(F,R)'], df['D(clip_flant(Vc_mean,R),clip_flant(Vc_mean,S))'], c='#f79410',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[0].scatter(a['clip_flant(F,R)'], a['D(clip_flant(Vc_mean,R),clip_flant(Vc_mean,S))'], c='#e77410',marker='.', s=size, alpha=transparency, label='filtered data')
ax[0].set_xlabel('clip_flant($F,T_R$)')
ax[0].set_ylabel('clip_flant($V_S^C,T_R$) - clip_flant($V_S^C,T_S$)')
ax[0].set_title('Clip flant')
ax[0].set_xlim([0,1])
ax[0].set_ylim([-1,1])
z = np.polyfit(df['clip_flant(F,R)'], df['D(clip_flant(Vc_mean,R),clip_flant(Vc_mean,S))'], 1)
p = np.poly1d(z)
xs = np.linspace(0, 1, 10000)
ax[0].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')


ax[1].scatter(df['llava(F,R)'], df['D(llava(Vc_mean,R),llava(Vc_mean,S))'], c='#1a6fc4',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[1].scatter(b['llava(F,R)'], b['D(llava(Vc_mean,R),llava(Vc_mean,S))'], c='#0f3fba',marker='.', s=size, alpha=transparency, label='filtered data')
ax[1].set_xlabel('llava($F,T_R$)')
ax[1].set_ylabel('llava($V_S^C,T_R$) - llava($V_S^C,T_S$)')
ax[1].set_title('Llava')
ax[1].set_xlim([0,1])
ax[1].set_ylim([-1,1])
z = np.polyfit(df['llava(F,R)'], df['D(llava(Vc_mean,R),llava(Vc_mean,S))'], 1)
p = np.poly1d(z)
ax[1].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')

ax[2].scatter(df['instructblip(F,R)'], df['D(instructblip(Vc_mean,R),instructblip(Vc_mean,S))'], c='#0cb14d',marker='.', s=size, alpha=transparency, label='models avereage raw')
ax[2].scatter(c['instructblip(F,R)'], c['D(instructblip(Vc_mean,R),instructblip(Vc_mean,S))'], c='#0c811d',marker='.', s=size, alpha=transparency, label='filtered data')
ax[2].set_xlabel('instructblip($F,T_R$)')
ax[2].set_ylabel('instructblip($V_S^C,T_R$) - instructblip($V_S^C,T_S$)')
ax[2].set_title('Instructblip flant')
ax[2].set_xlim([0,1])
ax[2].set_ylim([-1,1])
z = np.polyfit(df['instructblip(F,R)'], df['D(instructblip(Vc_mean,R),instructblip(Vc_mean,S))'], 1)
p = np.poly1d(z)
ax[2].plot(xs, p(xs), "r-",linewidth=0.7, label='linear regression')

fig.set_tight_layout(tight=True)

#plt.legend(markerscale=8, ncol=1, loc=4)

plt.savefig(f'D2_models_comparison.png', dpi=300)
plt.clf()
