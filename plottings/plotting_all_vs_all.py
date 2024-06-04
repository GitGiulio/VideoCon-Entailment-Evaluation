import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

statistics_df = pd.read_csv('../data/complete_df.csv')


ff_models = ['llava','instructblip','clip_flant','mean_wv']
video_models = ['llava','instructblip','clip_flant','videocon','mean','mean_wv']

for ff_model in ff_models:
        for video_model in video_models:
            for v_caption in ['R','S']:
                for f_caption in ['R', 'S']:
                    for video_type in ['c','u']:
                        for r in [1,2,3,4,5]:
                                plt.title(f'{video_model}(V{video_type}_{r},{v_caption}) vs {ff_model}(F,{f_caption})')

                                plt.scatter(statistics_df[f'{ff_model}(F,{f_caption})'],statistics_df[f'{video_model}(V{video_type}_{r},{v_caption})'],alpha=0.6,s=0.6)
                                plt.xlabel(f'{ff_model}(F,{f_caption})')
                                plt.ylabel(f'{video_model}(V{video_type}_{r},{v_caption})')
                                plt.savefig(f'../figures/{video_model}(V{video_type}_{r},{v_caption})_vs_{ff_model}(F,{f_caption}).png')
# 720 grafici molto coerenti fra loro e tutti abbastanza deprimenti :(

statistics_df.to_csv('../data/df.csv')