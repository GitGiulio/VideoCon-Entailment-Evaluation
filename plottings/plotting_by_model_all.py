import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])

df = pd.read_csv('../data/complete_df.csv')


for model in ['clip_flant', 'instructblip', 'llava']:
    for video_type in ['c', 'u']:
        for data in ['DIFF', 'REAL', 'SYNTH']:

            ylim = 0

            ff_set = ''
            color = '#7d5f8d'
            if data == 'DIFF':
                ylim = -1
                ff_set = f'D({model}(F,R),{model}(F,S))'
                if video_type == 'c':
                    color = '#4a4a4a'
                elif video_type == 'u':
                    color = '#7d5f8d'
            elif data == 'REAL':
                ff_set = f'{model}(F,R)'
                if video_type == 'c':
                    color = '#145d9e'
                elif video_type == 'u':
                    color = '#0cb14d'
            elif data == 'SYNTH':
                ff_set = f'{model}(F,S)'
                if video_type == 'c':
                    color = '#f13511'
                elif video_type == 'u':
                    color = '#f79410'

            size = 0.7
            transparency = 0.7
            
            plt.rc('font', size=16)

            fig, axs = plt.subplots(3, 4,figsize=(20,15))

            fig.suptitle(f'{model} vs ff_{model}_{data}')

            axs[0,0].scatter(df[f'{model}(V{video_type}_1,R)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[0,0].set_ylim([ylim, 1])
            axs[0,0].set_xlim([0, 1])
            plt.setp(axs[0,0], ylabel='VIDEO REAL')
            axs[0,1].scatter(df[f'{model}(V{video_type}_2,R)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[0,1].set_ylim([ylim, 1])
            axs[0,1].set_xlim([0, 1])
            axs[0,2].scatter(df[f'{model}(V{video_type}_4,R)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[0,2].set_ylim([ylim, 1])
            axs[0,2].set_xlim([0, 1])
            axs[0,3].scatter(df[f'{model}(V{video_type}_5,R)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[0,3].set_ylim([ylim, 1])
            axs[0,3].set_xlim([0, 1])

            axs[1,0].scatter(df[f'{model}(V{video_type}_1,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[1,0].set_ylim([ylim, 1])
            axs[1,0].set_xlim([0, 1])
            plt.setp(axs[1,0], ylabel='VIDEO SYNTH')
            axs[1,1].scatter(df[f'{model}(V{video_type}_2,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[1,1].set_ylim([ylim, 1])
            axs[1,1].set_xlim([0, 1])
            axs[1,2].scatter(df[f'{model}(V{video_type}_4,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[1,2].set_ylim([ylim, 1])
            axs[1,2].set_xlim([0, 1])
            axs[1,3].scatter(df[f'{model}(V{video_type}_5,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[1,3].set_ylim([ylim, 1])
            axs[1,3].set_xlim([0, 1])

            axs[2,0].scatter(df[f'{model}(V{video_type}_1,R)'] - df[f'{model}(V{video_type}_1,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[2,0].set_ylim([ylim, 1])
            axs[2,0].set_xlim([-1, 1])
            plt.setp(axs[2,0], xlabel='R1', ylabel='VIDEO DIFFERENCE (SYNTH-REAL)')
            axs[2,1].scatter(df[f'{model}(V{video_type}_2,R)'] - df[f'{model}(V{video_type}_2,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[2,1].set_ylim([ylim, 1])
            axs[2,1].set_xlim([-1, 1])
            plt.setp(axs[2,1], xlabel='R2')
            axs[2,2].scatter(df[f'{model}(V{video_type}_4,R)'] - df[f'{model}(V{video_type}_4,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[2,2].set_ylim([ylim, 1])
            axs[2,2].set_xlim([-1, 1])
            plt.setp(axs[2,2], xlabel='R4')
            axs[2,3].scatter(df[f'{model}(V{video_type}_5,R)'] - df[f'{model}(V{video_type}_5,S)'], df[ff_set], c=color, s=size, alpha=transparency)
            axs[2,3].set_ylim([ylim, 1])
            axs[2,3].set_xlim([-1, 1])
            plt.setp(axs[2,3], xlabel='R5')

            a = 'difference'

            if video_type == 'c':
                a = 'conditioned'
            elif video_type == 'u':
                a = 'unconditioned'
            if model == 'instructblip':
                plt.savefig(f'plots/{model}_flant/{a}/{data}.png', dpi=300)
            else:
                plt.savefig(f'plots/{model}/{a}/{data}.png', dpi=300)
