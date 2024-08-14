import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

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

ds = df.sort_values('clip_flant(Vc_mean,R)',ascending=True)

ds.reset_index(drop=True, inplace=True)

plt.title("Models correlation for conditional videos",fontsize=7)
plt.scatter(ds.index.values, ds['videocon(Vc_mean,R)'],c='#f79410',s=size,alpha=transparency,marker='+')
plt.scatter(ds.index.values, ds['instructblip(Vc_mean,R)'],c='#0cb14d',s=size,alpha=transparency,marker='x')
plt.scatter(ds.index.values, ds['llava(Vc_mean,R)'],c='#145d9e',s=size,alpha=transparency,marker='+')
plt.scatter(ds.index.values, ds['clip_flant(Vc_mean,R)'],c='#f13511',s=size,alpha=transparency,marker='x')

plt.text(18000, 0.63, 'Here could\nbe a short\n description\n', fontsize=5, color='black')

plt.legend(["videocon", "instructblip", "llava", "clip_flant"], markerscale=10, ncol=1, bbox_to_anchor=(1, 1))

plt.savefig(f'plots/P1_models_correlation_videos.png', dpi=300)
plt.clf()


