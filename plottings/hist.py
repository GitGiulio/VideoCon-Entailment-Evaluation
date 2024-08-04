import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])

df = pd.read_csv('../data/complete_df.csv')


# ciclo per creare tutti gli istogrammi per i video
for model in ['llava','clip_flant','instructblip','mean','videocon','mean_wv']:
    for video_type in ['c','u']:
        for caption in ['R','S']:
            title = ''
            c = ''
            if model == 'llava':
                c = '#f13511'
            elif model == 'clip_flant':
                c = '#f79410'
            elif model == 'instructblip':
                c = '#0cb14d'
            elif model == 'videocon':
                c = '#145d9e'
            elif model == 'mean':
                c = '#4a4a4a'
            elif model == 'mean_wv':
                c = '#7d5f8d'

            if model in ['mean','mean_wv']:
                plt.hist(df[f'{model}(V{video_type},{caption})'],bins=200,color=c)
                title = f'{model}(V{video_type},{caption})'
            else:
                plt.hist(df[f'{model}(V{video_type}_1,R)'],bins=200,color=c)
                title = f'{model}(V{video_type}_1,R)'

            plt.title(title)

            plt.savefig(f'plots/histograms/{model}/Video/{title}.png')
            plt.clf()

# ciclo per creare tutti gli istogrammi per i frame
for model in ['llava','clip_flant','instructblip','mean_wv']:
    for caption in ['R','S']:
        title = ''
        c = ''
        if model == 'llava':
            c = '#f13511'
        elif model == 'clip_flant':
            c = '#f79410'
        elif model == 'instructblip':
            c = '#0cb14d'
        elif model == 'mean_wv':
            c = '#7d5f8d'

        plt.hist(df[f'{model}(F,{caption})'], bins=200,color=c)
        title = f'{model}(F,{caption})'

        plt.savefig(f'plots/histograms/{model}/Frame/{title}.png')
        plt.clf()
