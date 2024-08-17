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

size = 0.15
transparency = 0.7
plt.rcParams.update({'font.size': 5})

ds = df.sort_values('clip_flant(Vc_mean,S)',ascending=True)

ds.reset_index(drop=True, inplace=True)

plt.title("Models correlation for conditional videos",fontsize=7)
plt.scatter(ds.index.values, ds['videocon(Vc_mean,S)'],c='#f79410',s=size,alpha=transparency,marker='+',label="videocon")
plt.scatter(ds.index.values, ds['instructblip(Vc_mean,S)'],c='#0cb14d',s=size,alpha=transparency,marker='x',label="instructblip")
plt.scatter(ds.index.values, ds['llava(Vc_mean,S)'],c='#145d9e',s=size,alpha=transparency,marker='+',label= "llava")
plt.scatter(ds.index.values, ds['clip_flant(Vc_mean,S)'],c='#f13511',s=size,alpha=transparency,marker='x',label="clip_flant")

plt.xlabel('video caption pair')
plt.ylabel('$E(V_S^C,T_S)$')

#plt.text(18000, 0.63, 'Here could\nbe a short\n description\n', fontsize=5, color='black')

legend = plt.legend(facecolor="pink", loc=2,markerscale=10)
plt.savefig(f'plots/P1_models_correlation_videos.png', dpi=300)
plt.clf()


