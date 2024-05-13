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
r = 1
video_set = 'llava synth'
video_type = 'unconditioned'
ff_set = 'ff_entail_diff_llava'

fig, axs = plt.subplots(3, 4,figsize=(20,15))

fig.suptitle('clip-flant vs ff_clip-flant_DIFF') # for BOX

axs[0,0].scatter(df[f'clip_flant real {video_type} r{1} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[0,0].set_ylim([0, 1])
axs[0,1].scatter(df[f'clip_flant real {video_type} r{2} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[0,1].set_ylim([0, 1])
axs[0,2].scatter(df[f'clip_flant real {video_type} r{4} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[0,2].set_ylim([0, 1])
axs[0,3].scatter(df[f'clip_flant real {video_type} r{5} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[0,3].set_ylim([0, 1])

axs[1,0].scatter(df[f'clip_flant synth {video_type} r{1} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[1,0].set_ylim([0, 1])
axs[1,1].scatter(df[f'clip_flant synth {video_type} r{2} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[1,1].set_ylim([0, 1])
axs[1,2].scatter(df[f'clip_flant synth {video_type} r{4} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[1,2].set_ylim([0, 1])
axs[1,3].scatter(df[f'clip_flant synth {video_type} r{5} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[1,3].set_ylim([0, 1])

axs[2,0].scatter(df[f'clip_flant synth {video_type} r{1} ent'] - df[f'clip_flant real {video_type} r{1} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[2,1].scatter(df[f'clip_flant synth {video_type} r{2} ent'] - df[f'clip_flant real {video_type} r{2} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[2,2].scatter(df[f'clip_flant synth {video_type} r{4} ent'] - df[f'clip_flant real {video_type} r{4} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)
axs[2,3].scatter(df[f'clip_flant synth {video_type} r{5} ent'] - df[f'clip_flant real {video_type} r{5} ent'],df['ff_entail_diff_clip_flant'],c='pink',s=0.7,alpha=0.7)


plt.savefig('plots/clip-flant/unconditioned/DIFF.png')
