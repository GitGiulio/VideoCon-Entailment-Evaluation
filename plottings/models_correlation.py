
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

df['ff_entail_diff_llava'] = df['ff_cap_llava'] - df['ff_neg_cap_llava']
df['ff_entail_diff_clip_flant'] = df['ff_cap_clip_flant'] - df['ff_neg_cap_clip_flant']
df['ff_entail_diff_instructblip_flant'] = df['ff_cap_instructblip_flant'] - df['ff_neg_cap_instructblip_flant']

ff_sets = ['ff_entail_diff_llava', 'ff_entail_diff_clip_flant', 'ff_entail_diff_instructblip_flant',
           'ff_cap_llava', 'ff_neg_cap_llava', 'ff_cap_clip_flant', 'ff_neg_cap_clip_flant',
           'ff_cap_instructblip_flant', 'ff_neg_cap_instructblip_flant']
video_sets = ['videocon_synth', 'videocon_real',
              'clip_flant synth', 'clip_flant real',
              'instructblip_flant synth','instructblip_flant real',
              'llava synth', 'llava real']

#inf.write(f'{x}|{y1}|{y2}|{title}|{xlable}|{ylable}|{model}|{color}|{xlim[0]},{xlim[1]}|{ylim[0]},{ylim[1]}|{filename}\n')

size = 0.6
transparency = 0.3

ds = df.sort_values('instructblip_flant real conditioned r1 ent',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("prova",fontsize=11)
plt.scatter(ds.index, ds['videocon_real conditioned r1 ent'],c='g',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['instructblip_flant real conditioned r1 ent'],c='r',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['clip_flant real conditioned r1 ent'],c='b',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['llava real conditioned r1 ent'],c='y',s=size,alpha=transparency)

plt.tight_layout(pad=2.0)

plt.savefig(f'plots/models_correlations/v_r_c.png', dpi=300)
plt.clf()

