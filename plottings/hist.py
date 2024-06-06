import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

# TODO fare cartelle per modelli e automatizzare anche questo

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta

# ciclo per creare tutti gli istogrammi per i video
for model in ['llava','clip_flant','instructblip','mean','videocon','mean_wv']:
    for video_type in ['c','u']:
        for caption in ['R','S']:
            color = '#2ad0f5'
            title = ''

            if model in ['mean','mean_wv']:
                plt.hist(df[f'{model}(V{video_type},{caption})'],bins=200,color=color)
                title = f'{model}(V{video_type},{caption})'
            else:
                plt.hist(df[f'{model}(V{video_type}_1,R)'],bins=200,color=color)
                title = f'{model}(V{video_type}_1,R)'

            plt.title(title)

            plt.savefig(f'plots/histograms/{model}/Video/{title}.png')
            plt.clf()

# ciclo per creare tutti gli istogrammi per i frame
for model in ['llava','clip_flant','instructblip','mean_wv']:
    for caption in ['R','S']:
        color = '#2ad0f5'
        title = ''

        plt.hist(df[f'{model}(F,{caption})'], bins=200, color=color)
        title = f'{model}(F,{caption})'

        plt.savefig(f'plots/histograms/{model}/Frame/{title}.png')
        plt.clf()
