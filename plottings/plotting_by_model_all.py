import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')


# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


for model in ['clip_flant', 'instructblip_flant', 'llava']:
    for video_type in ['c', 'u']:
        for data in ['DIFF', 'REAL', 'SYNTH']:

            ylim = 0

            ff_set = ''
            color = 'pink'
            if data == 'DIFF':
                ylim = -1
                ff_set = f'D({model}(F,R),{model}(F,S))'
                if video_type == 'conditioned':
                    color = 'magenta'
                elif video_type == 'unconditioned':
                    color = 'pink'
            elif data == 'REAL':
                ff_set = f'{model}(F,R)'
                if video_type == 'conditioned':
                    color = '#2af542'
                elif video_type == 'unconditioned':
                    color = '#75fa85'
            elif data == 'SYNTH':
                ff_set = f'{model}(F,S)'
                if video_type == 'conditioned':
                    color = '#e84c31'
                elif video_type == 'unconditioned':
                    color = '#f58b45'

            plt.rc('font', size=16)

            fig, axs = plt.subplots(3, 4,figsize=(20,15))

            fig.suptitle(f'{model} vs ff_{model}_{data}')

            axs[0,0].scatter(df[f'{model}(V{video_type}_1,R)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[0,0].set_ylim([ylim, 1])
            axs[0,0].set_xlim([0, 1])
            plt.setp(axs[0,0], ylabel='VIDEO REAL')
            axs[0,1].scatter(df[f'{model}(V{video_type}_2,R)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[0,1].set_ylim([ylim, 1])
            axs[0,1].set_xlim([0, 1])
            axs[0,2].scatter(df[f'{model}(V{video_type}_4,R)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[0,2].set_ylim([ylim, 1])
            axs[0,2].set_xlim([0, 1])
            axs[0,3].scatter(df[f'{model}(V{video_type}_5,R)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[0,3].set_ylim([ylim, 1])
            axs[0,3].set_xlim([0, 1])

            axs[1,0].scatter(df[f'{model}(V{video_type}_1,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[1,0].set_ylim([ylim, 1])
            axs[1,0].set_xlim([0, 1])
            plt.setp(axs[1,0], ylabel='VIDEO SYNTH')
            axs[1,1].scatter(df[f'{model}(V{video_type}_2,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[1,1].set_ylim([ylim, 1])
            axs[1,1].set_xlim([0, 1])
            axs[1,2].scatter(df[f'{model}(V{video_type}_4,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[1,2].set_ylim([ylim, 1])
            axs[1,2].set_xlim([0, 1])
            axs[1,3].scatter(df[f'{model}(V{video_type}_5,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[1,3].set_ylim([ylim, 1])
            axs[1,3].set_xlim([0, 1])

            axs[2,0].scatter(df[f'{model}(V{video_type}_1,R)'] - df[f'{model}(V{video_type}_1,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[2,0].set_ylim([ylim, 1])
            axs[2,0].set_xlim([-1, 1])
            plt.setp(axs[2,0], xlabel='R1', ylabel='VIDEO DIFFERENCE (SYNTH-REAL)')
            axs[2,1].scatter(df[f'{model}(V{video_type}_2,R)'] - df[f'{model}(V{video_type}_2,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[2,1].set_ylim([ylim, 1])
            axs[2,1].set_xlim([-1, 1])
            plt.setp(axs[2,1], xlabel='R2')
            axs[2,2].scatter(df[f'{model}(V{video_type}_4,R)'] - df[f'{model}(V{video_type}_4,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[2,2].set_ylim([ylim, 1])
            axs[2,2].set_xlim([-1, 1])
            plt.setp(axs[2,2], xlabel='R4')
            axs[2,3].scatter(df[f'{model}(V{video_type}_5,R)'] - df[f'{model}(V{video_type}_5,S)'], df[ff_set], c=color, s=0.7, alpha=0.7)
            axs[2,3].set_ylim([ylim, 1])
            axs[2,3].set_xlim([-1, 1])
            plt.setp(axs[2,3], xlabel='R5')


            plt.savefig(f'plots/{model}/{video_type}/{data}.png', dpi=300)
