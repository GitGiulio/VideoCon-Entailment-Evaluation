import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

statistics_df = pd.read_csv('../data/complete_df.csv')

statistics_df['ff_entail_diff_llava'] = statistics_df['ff_cap_llava'] - statistics_df['ff_neg_cap_llava']
statistics_df['ff_entail_diff_clip_flant'] = statistics_df['ff_cap_clip_flant'] - statistics_df['ff_neg_cap_clip_flant']
statistics_df['ff_entail_diff_instructblip_flant'] = statistics_df['ff_cap_instructblip_flant'] - statistics_df['ff_neg_cap_instructblip_flant']

ff_sets = ['ff_entail_diff_llava', 'ff_entail_diff_clip_flant', 'ff_entail_diff_instructblip_flant',
           'ff_cap_llava', 'ff_neg_cap_llava', 'ff_cap_clip_flant', 'ff_neg_cap_clip_flant',
           'ff_cap_instructblip_flant', 'ff_neg_cap_instructblip_flant']
video_sets = ['videocon_synth', 'videocon_real',
              'clip_flant synth', 'clip_flant real',
              'instructblip_flant synth','instructblip_flant real',
              'llava synth', 'llava real']

for ff_set in ff_sets:
        for video_set in video_sets:
                for video_type in ['conditioned','unconditioned']:
                        for i in range(0,5):
                                #fig, axs = plt.subplots(2, 3,figsize=(15,5))

                                plt.title(f'Round {i+1} {video_set} {video_type} entailment vs {ff_set}')

                                plt.scatter(statistics_df[f'{video_set} {video_type} r{i+1} ent'],statistics_df[ff_set],alpha=0.3,s=1)
                                plt.xlabel(f'Round {i+1} {video_set} {video_type} entailment')
                                plt.ylabel(ff_set)
                                plt.savefig(f'../figures/Round{i+1}_{video_set}_{video_type}_vs_{ff_set}_.png')

print(statistics_df['entailments_mean'].std())

statistics_df.to_csv('../data/statistics_df.csv')