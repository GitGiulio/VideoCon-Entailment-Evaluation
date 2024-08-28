import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

from plottings.plots_instruction_handling import color

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
def centered_subplots(rows, figsize=None):
    grid_dim = max(rows)
    grid_shape = (len(rows), 2 * grid_dim)

    if figsize:
        fig = plt.figure(figsize=(figsize))
    else:
        fig = plt.figure(figsize=(2 * grid_dim, 3 * len(rows)))

    allaxes = []

    jrow = 0
    for row in rows:
        offset = 0
        for i in range(row):
            if row < grid_dim:
                offset = grid_dim - row

            ax_position = (jrow, 2 * i + offset)
            ax = plt.subplot2grid(grid_shape, ax_position, fig=fig, colspan=2)
            allaxes.append(ax)

        jrow += 1

    return allaxes

a1 = pd.DataFrame()
b1 = pd.DataFrame()
c1 = pd.DataFrame()

for index,row in df.iterrows():
    if row['D(llava(F,R),llava(F,S))'] > 0:
        a1 = pd.concat([a1, pd.DataFrame([{'D(llava(F,R),llava(F,S))':row['D(llava(F,R),llava(F,S))'],'D(llava(Vu,S),llava(Vc,S))':row['D(llava(Vu,S),llava(Vc,S))']}])], ignore_index=True)
    if row['D(clip_flant(F,R),clip_flant(F,S))'] > 0:
        b1 = pd.concat([b1, pd.DataFrame([{'D(clip_flant(F,R),clip_flant(F,S))':row['D(clip_flant(F,R),clip_flant(F,S))'],'D(clip_flant(Vu,S),clip_flant(Vc,S))':row['D(clip_flant(Vu,S),clip_flant(Vc,S))']}])], ignore_index=True)
    if row['D(instructblip(F,R),instructblip(F,S))'] > 0:
        c1 = pd.concat([c1, pd.DataFrame([{'D(instructblip(F,R),instructblip(F,S))':row['D(instructblip(F,R),instructblip(F,S))'],'D(instructblip(Vu,S),instructblip(Vc,S))':row['D(instructblip(Vu,S),instructblip(Vc,S))']}])], ignore_index=True)

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

plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax = centered_subplots([2,1],figsize=(10, 10))

plt.suptitle('Model comparison',fontsize=13)

ax[0].scatter(df['D(llava(F,R),llava(F,S))'], df['D(llava(Vu,S),llava(Vc,S))'], c='#1a6fc4',marker='.', s=size, alpha=transparency)
ax[0].scatter(a['D(llava(F,R),llava(F,S))'], a['D(llava(Vu,S),llava(Vc,S))'], c='#0f3fba',marker='.', s=size, alpha=transparency)
ax[0].set_xlabel('llava(F,$T_R$) - llava(F,$T_S$)')
ax[0].set_ylabel('llava($V_S^U$,$T_S$) - llava($V_S^C$,$T_S$)')
ax[0].set_title('LLAVA')
ax[0].set_xlim([-1,1])
ax[0].set_ylim([-1,1])
xs = np.linspace(-1, 1, 10000)
xs2 = np.linspace(0, 1, 10000)

z = np.polyfit(df['D(llava(F,R),llava(F,S))'], df['D(llava(Vu,S),llava(Vc,S))'], 1)
p = np.poly1d(z)
ax[0].plot(xs, p(xs), "r-")
z = np.polyfit(a1['D(llava(F,R),llava(F,S))'], a1['D(llava(Vu,S),llava(Vc,S))'], 2)
p = np.poly1d(z)
ax[0].plot(xs2, p(xs2), "-",color='#e42af5')
ys = np.linspace(0, 0, 10000)
ax[0].plot(xs, ys,'--',c='#505050',linewidth=0.6)
ax[0].plot(ys, xs,'--',c='#505050',linewidth=0.6)

ax[1].scatter(df['D(clip_flant(F,R),clip_flant(F,S))'], df['D(clip_flant(Vu,S),clip_flant(Vc,S))'], c='#f79410',marker='.', s=size, alpha=transparency)
ax[1].scatter(b['D(clip_flant(F,R),clip_flant(F,S))'], b['D(clip_flant(Vu,S),clip_flant(Vc,S))'], c='#e77410',marker='.', s=size, alpha=transparency)
ax[1].set_xlabel('clip_flant(F,$T_R$) - clip_flant(F,$T_S$)')
ax[1].set_ylabel('clip_flant($V_S^U$,$T_S$) - clip_flant($V_S^C$,$T_S$)')
ax[1].set_title('clip_flant')
ax[1].set_xlim([-1,1])
ax[1].set_ylim([-1,1])

z = np.polyfit(df['D(clip_flant(F,R),clip_flant(F,S))'], df['D(clip_flant(Vu,S),clip_flant(Vc,S))'], 1)
p = np.poly1d(z)
ax[1].plot(xs, p(xs), "r-")
z = np.polyfit(b1['D(clip_flant(F,R),clip_flant(F,S))'], b1['D(clip_flant(Vu,S),clip_flant(Vc,S))'], 2)
p = np.poly1d(z)
ax[1].plot(xs2, p(xs2), "-",color='#e42af5')
ax[1].plot(xs, ys,'--',c='#505050',linewidth=0.6)
ax[1].plot(ys, xs,'--',c='#505050',linewidth=0.6)

ax[2].scatter(df['D(instructblip(F,R),instructblip(F,S))'], df['D(instructblip(Vu,S),instructblip(Vc,S))'], c='#0cb14d',marker='.', s=size, alpha=transparency)
ax[2].scatter(c['D(instructblip(F,R),instructblip(F,S))'], c['D(instructblip(Vu,S),instructblip(Vc,S))'], c='#0c811d',marker='.', s=size, alpha=transparency)
ax[2].set_xlabel('instructblip(F,$T_R$) - instructblip(F,$T_S$)')
ax[2].set_ylabel('instructblip($V_S^U$,$T_S$) - instructblip($V_S^C$,$T_S$)')
ax[2].set_title('instructblip')
ax[2].set_xlim([-1,1])
ax[2].set_ylim([-1,1])
z = np.polyfit(df['D(instructblip(F,R),instructblip(F,S))'], df['D(instructblip(Vu,S),instructblip(Vc,S))'], 1)
p = np.poly1d(z)
ax[2].plot(xs, p(xs), "r-")
z = np.polyfit(c1['D(instructblip(F,R),instructblip(F,S))'], c1['D(instructblip(Vu,S),instructblip(Vc,S))'], 2)
p = np.poly1d(z)
ax[2].plot(xs2, p(xs2), "-",color='#e42af5')
ax[2].plot(xs, ys,'--',c='#505050',linewidth=0.6)
ax[2].plot(ys, xs,'--',c='#505050',linewidth=0.6)


plt.tight_layout()
plt.savefig(f'F1.png', dpi=300)
plt.clf()

matplotlib.pyplot.close()