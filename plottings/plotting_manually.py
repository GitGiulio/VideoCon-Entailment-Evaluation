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

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


for coso in ['clip_flant', 'instructblip_flant','llava']:
    for video_type in ['conditioned', 'unconditioned']:
        for doso in ['DIFF', 'REAL','SYNTH']:

            ff_set = ''
            color = 'pink'
            if doso == 'DIFF':
                ff_set = f'ff_entail_diff_{coso}'
                if video_type == 'conditioned':
                    color = 'magenta'
                elif video_type == 'unconditioned':
                    color = 'pink'
            elif doso == 'REAL':
                ff_set = f'ff_cap_{coso}'
                if video_type == 'conditioned':
                    color = '#2af542'
                elif video_type == 'unconditioned':
                    color = '#75fa85'
            elif doso == 'SYNTH':
                ff_set = f'ff_neg_cap_{coso}'
                if video_type == 'conditioned':
                    color = '#e84c31'
                elif video_type == 'unconditioned':
                    color = '#f58b45'


            fig, axs = plt.subplots(3, 4,figsize=(20,15))

            fig.suptitle(f'{coso} vs ff_{coso}_{doso}') # for BOX

            axs[0,0].scatter(df[f'{coso} real {video_type} r{1} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[0,0].set_ylim([0, 1])
            axs[0,1].scatter(df[f'{coso} real {video_type} r{2} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[0,1].set_ylim([0, 1])
            axs[0,2].scatter(df[f'{coso} real {video_type} r{4} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[0,2].set_ylim([0, 1])
            axs[0,3].scatter(df[f'{coso} real {video_type} r{5} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[0,3].set_ylim([0, 1])

            axs[1,0].scatter(df[f'{coso} synth {video_type} r{1} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[1,0].set_ylim([0, 1])
            axs[1,1].scatter(df[f'{coso} synth {video_type} r{2} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[1,1].set_ylim([0, 1])
            axs[1,2].scatter(df[f'{coso} synth {video_type} r{4} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[1,2].set_ylim([0, 1])
            axs[1,3].scatter(df[f'{coso} synth {video_type} r{5} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[1,3].set_ylim([0, 1])

            axs[2,0].scatter(df[f'{coso} synth {video_type} r{1} ent'] - df[f'clip_flant real {video_type} r{1} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[2,1].scatter(df[f'{coso} synth {video_type} r{2} ent'] - df[f'clip_flant real {video_type} r{2} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[2,2].scatter(df[f'{coso} synth {video_type} r{4} ent'] - df[f'clip_flant real {video_type} r{4} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)
            axs[2,3].scatter(df[f'{coso} synth {video_type} r{5} ent'] - df[f'clip_flant real {video_type} r{5} ent'],df[ff_set],c='pink',s=0.7,alpha=0.7)


            plt.savefig(f'plots/{coso}/{video_type}/{doso}.png')
