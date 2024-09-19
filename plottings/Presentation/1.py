import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from scipy.ndimage import label

plt.style.use(['science','scatter'])

df = pd.read_csv('with_mean.csv')

"""
df['videocon(Vc_mean,R)'] = (df[f'videocon(Vc_1,R)'] + df[f'videocon(Vc_2,R)'] + df[f'videocon(Vc_4,R)'] + df[f'videocon(Vc_5,R)'] ) / 4
df['instructblip(Vc_mean,R)'] = (df[f'instructblip(Vc_1,R)'] + df[f'instructblip(Vc_2,R)'] + df[f'instructblip(Vc_4,R)'] + df[f'instructblip(Vc_5,R)'] ) / 4
df['llava(Vc_mean,R)'] = (df[f'llava(Vc_1,R)'] + df[f'llava(Vc_2,R)'] + df[f'llava(Vc_4,R)'] + df[f'llava(Vc_5,R)'] ) / 4
df['clip_flant(Vc_mean,R)'] = (df[f'clip_flant(Vc_1,R)'] + df[f'clip_flant(Vc_2,R)'] + df[f'clip_flant(Vc_4,R)'] + df[f'clip_flant(Vc_5,R)'] ) / 4
df.to_csv('with_mean.csv', index=False)
exit(2)
"""

def filter(df,val):
    a = pd.DataFrame()
    c = 0
    df = df.sort_values('clip_flant(Vc_mean,S)',ascending=True)
    df = df.reset_index(drop=True)
    for index,row in df.iterrows():
        c += 1
        if c >= val:
            b1 = 0
            b2 = 0
            b3 = 0
            b4 = 0
            for i in range(val):
                b1 += df['clip_flant(Vc_mean,S)'].at[index-i]
                b2 += df['videocon(Vc_mean,S)'].at[index-i]
                b3 += df['instructblip(Vc_mean,S)'].at[index-i]
                b4 += df['llava(Vc_mean,S)'].at[index-i]
            a = pd.concat([a, pd.DataFrame([{'clip_flant(Vc_mean,S)':  b1/val,'videocon(Vc_mean,S)': b2/val
                                               ,'instructblip(Vc_mean,S)': b3/val,'llava(Vc_mean,S)': b4/val}])],ignore_index=True)
    return a

a = filter(df,17)
df = df.sort_values('clip_flant(Vc_mean,S)',ascending=True)
df = df.reset_index(drop=True)
size = 0.05
transparency = 0.5  # df[y]

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 11})

plt.title('Models correlation for conditional videos', fontsize=13)

plt.scatter(df.index.values, df['videocon(Vc_mean,S)'],c='#f79410',s=size*2,alpha=transparency,marker='+',label="videocon")
plt.scatter(df.index.values, df['instructblip(Vc_mean,S)'],c='#0cb14d',s=size,alpha=transparency,marker='x',label="instructblip")
plt.scatter(df.index.values, df['llava(Vc_mean,S)'],c='#145d9e',s=size*2,alpha=transparency,marker='+',label= "llava")
plt.scatter(df.index.values, df['clip_flant(Vc_mean,S)'],c='#f13511',s=size,alpha=transparency,marker='x',label="clip_flant")
plt.xlabel('video-caption pair')
plt.ylabel('$E(V_S^C,T_S)$')
#ax[0].set_xlim(xlim)
#ax[0].set_ylim(ylim)

plt.tight_layout()
legend = plt.legend(ncol=2,loc=(0.0,-0.55),markerscale=20)
for lh in legend.legend_handles:
    lh.set_alpha(1)
plt.savefig(f'models_correlation.png', dpi=300)
plt.clf()
