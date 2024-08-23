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

df[f'D(llava(Vc,R),llava(Vc,S))'] = df[f'llava(Vc_mean,R)'] - df[f'llava(Vc_mean,S)']
df[f'D(llava(Vu,R),llava(Vu,S))'] = df[f'llava(Vu_mean,R)'] - df[f'llava(Vu_mean,S)']
df[f'D(llava(Vu,S),llava(Vc,S))'] = df[f'llava(Vu_mean,S)'] - df[f'llava(Vc_mean,S)']
df[f'D(llava(Vu,R),llava(Vc,R))'] = df[f'llava(Vu_mean,R)'] - df[f'llava(Vc_mean,R)']

df[f'D(instructblip(Vc,R),instructblip(Vc,S))'] = df[f'instructblip(Vc_mean,R)'] - df[f'instructblip(Vc_mean,S)']
df[f'D(instructblip(Vu,R),instructblip(Vu,S))'] = df[f'instructblip(Vu_mean,R)'] - df[f'instructblip(Vu_mean,S)']
df[f'D(instructblip(Vu,S),instructblip(Vc,S))'] = df[f'instructblip(Vu_mean,S)'] - df[f'instructblip(Vc_mean,S)']
df[f'D(instructblip(Vu,R),instructblip(Vc,R))'] = df[f'instructblip(Vu_mean,R)'] - df[f'instructblip(Vc_mean,R)']

df[f'D(clip_flant(Vc,R),clip_flant(Vc,S))'] = df[f'clip_flant(Vc_mean,R)'] - df[f'clip_flant(Vc_mean,S)']
df[f'D(clip_flant(Vu,R),clip_flant(Vu,S))'] = df[f'clip_flant(Vu_mean,R)'] - df[f'clip_flant(Vu_mean,S)']
df[f'D(clip_flant(Vu,S),clip_flant(Vc,S))'] = df[f'clip_flant(Vu_mean,S)'] - df[f'clip_flant(Vc_mean,S)']
df[f'D(clip_flant(Vu,R),clip_flant(Vc,R))'] = df[f'clip_flant(Vu_mean,R)'] - df[f'clip_flant(Vc_mean,R)']

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

a = filter(df,23,'D(llava(F,R),llava(F,S))','D(llava(Vu,S),llava(Vc,S))')
b = filter(df,23,'D(clip_flant(F,R),clip_flant(F,S))','D(clip_flant(Vu,S),clip_flant(Vc,S))')
c = filter(df,23,'D(instructblip(F,R),instructblip(F,S))','D(instructblip(Vu,S),instructblip(Vc,S))')

size = 0.6
transparency = 1

plt.rcParams.update({'font.size': 9})

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

fig.suptitle('Unconditional - conditional trend',fontsize=11)

ax[0].scatter(df['D(llava(F,R),llava(F,S))'], df['D(llava(Vu,S),llava(Vc,S))'], c='#1a6fc4',marker='.', s=size, alpha=transparency)
ax[0].scatter(a['D(llava(F,R),llava(F,S))'], a['D(llava(Vu,S),llava(Vc,S))'], c='#0f3fba',marker='.', s=size, alpha=transparency)
ax[0].set_xlabel('llava(F,$T_R$) - llava(F,$T_S$)')
ax[0].set_ylabel('llava($V_S^U$,$T_S$) - llava($V_S^C$,$T_S$)')
ax[0].set_title('LLAVA')
ax[0].set_xlim([-1,1])
ax[0].set_ylim([-1,1])
xs = np.linspace(-1, 1, 10000)

z = np.polyfit(df['D(llava(F,R),llava(F,S))'], df['D(llava(Vu,S),llava(Vc,S))'], 3)
p = np.poly1d(z)
ax[0].plot(xs, p(xs), "r-", label='Trend line')

ax[1].scatter(df['D(clip_flant(F,R),clip_flant(F,S))'], df['D(clip_flant(Vu,S),clip_flant(Vc,S))'], c='#f79410',marker='.', s=size, alpha=transparency)
ax[1].scatter(b['D(clip_flant(F,R),clip_flant(F,S))'], b['D(clip_flant(Vu,S),clip_flant(Vc,S))'], c='#e77410',marker='.', s=size, alpha=transparency)
ax[1].set_xlabel('clip_flant(F,$T_R$) - clip_flant(F,$T_S$)')
ax[1].set_ylabel('clip_flant($V_S^U$,$T_S$) - clip_flant($V_S^C$,$T_S$)')
ax[1].set_title('clip_flant')
ax[1].set_xlim([-1,1])
ax[1].set_ylim([-1,1])
z = np.polyfit(df['D(clip_flant(F,R),clip_flant(F,S))'], df['D(clip_flant(Vu,S),clip_flant(Vc,S))'], 3)
p = np.poly1d(z)
ax[1].plot(xs, p(xs), "r-", label='Trend line')

ax[2].scatter(df['D(instructblip(F,R),instructblip(F,S))'], df['D(instructblip(Vu,S),instructblip(Vc,S))'], c='#0cb14d',marker='.', s=size, alpha=transparency)
ax[2].scatter(c['D(instructblip(F,R),instructblip(F,S))'], c['D(instructblip(Vu,S),instructblip(Vc,S))'], c='#0c811d',marker='.', s=size, alpha=transparency)
ax[2].set_xlabel('instructblip(F,$T_R$) - instructblip(F,$T_S$)')
ax[2].set_ylabel('instructblip($V_S^U$,$T_S$) - instructblip($V_S^C$,$T_S$)')
ax[2].set_title('instructblip')
ax[2].set_xlim([-1,1])
ax[2].set_ylim([-1,1])
z = np.polyfit(df['D(instructblip(F,R),instructblip(F,S))'], df['D(instructblip(Vu,S),instructblip(Vc,S))'], 3)
p = np.poly1d(z)
ax[2].plot(xs, p(xs), "r-", label='Trend line')

fig.set_tight_layout(tight=True)
plt.savefig(f'P7_models_comparison.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()