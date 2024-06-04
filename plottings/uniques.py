import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')
""" # ho salvato il DF con queste modifiche apportate, quindi queste righe non è più necessario eseguirle
df['D(llava(F,R),llava(F,S))'] = df['llava(F,R)'] - df['llava(F,S)']
df['D(clip_flant(F,R),clip_flant(F,S))'] = df['clip_flant(F,R)'] - df['clip_flant(F,S)']
df['D(instructblip(F,R),instructblip(F,S))'] = df['instructblip(F,R)'] - df['instructblip(F,S)']

df['mean_wv(F,R)'] = (df['llava(F,R)'] + df['clip_flant(F,R)'] + df['instructblip(F,R)']) / 3
df['mean_wv(F,S)'] = (df['llava(F,S)'] + df['clip_flant(F,S)'] + df['instructblip(F,S)']) / 3

df['D(mean_wv(F,R),mean_wv(F,R))'] = df['mean_wv(F,R)'] - df['mean_wv(F,S)']

df['mean(Vc,R)'] = 0
df['mean(Vc,S)'] = 0
df['mean(Vu,R)'] = 0
df['mean(Vu,S)'] = 0

df['mean_wv(Vc,R)'] = 0
df['mean_wv(Vc,S)'] = 0
df['mean_wv(Vu,R)'] = 0
df['mean_wv(Vu,S)'] = 0

counter_wv = 0
counter = 0

for round in [1,2,4,5]:
    for model in ['videocon','clip_flant', 'instructblip_flant', 'llava']:
        counter = counter + 1
        if model != 'videocon':
            counter_wv = counter_wv + 1
            df['mean_wv(Vc,R)'] += df[f'{model}(Vc,R_{round})']
            df['mean_wv(Vc,S)'] += df[f'{model}(Vc,S_{round})']
            df['mean_wv(Vu,R)'] += df[f'{model}(Vu,R_{round})']
            df['mean_wv(Vu,S)'] += df[f'{model}(Vu,S_{round})']
        df['mean(Vc,R)'] += df[f'{model}(Vc,R_{round})']
        df['mean(Vc,S)'] += df[f'{model}(Vc,S_{round})']
        df['mean(Vu,R)'] += df[f'{model}(Vu,R_{round})']
        df['mean(Vu,S)'] += df[f'{model}(Vu,S_{round})']

df['mean(Vc,R)'] /= counter
df['mean(Vc,S)'] /= counter
df['mean(Vu,R)'] /= counter
df['mean(Vu,S)'] /= counter

df['mean_wv(Vc,R)'] /= counter_wv
df['mean_wv(Vc,S)'] /= counter_wv
df['mean_wv(Vu,R)'] /= counter_wv
df['mean_wv(Vu,S)'] /= counter_wv
"""

inf = open('plots_instruction', 'r')

lines = inf.readlines()
X = []
Y = []
TITLES = []
XLABELS = []
YLABELS = []
MODELS = []
COLORS = []
XLIMS = []
YLIMS = []
FILENAMES = []

for line in lines:
    splitted_line = line.split('|')

    xlims_t = splitted_line[7].split(',')
    ylims_t = splitted_line[8].split(',')

    X.append(splitted_line[0])
    Y.append(splitted_line[1])
    TITLES.append(splitted_line[2])
    XLABELS.append(splitted_line[3])
    YLABELS.append(splitted_line[4])
    MODELS.append(splitted_line[5])
    COLORS.append(splitted_line[6])
    XLIMS.append([int(xlims_t[0]),int(xlims_t[1])])
    YLIMS.append([int(ylims_t[0]),int(ylims_t[1])])
    FILENAMES.append(splitted_line[9])

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


for x,y,title,xlable,ylable,model,color,xlim,ylim,filename in zip(X,Y,TITLES,XLABELS,YLABELS,MODELS,COLORS,XLIMS,YLIMS,FILENAMES):

    size = 0.9
    transparency = 0.6

    plt.title(title,fontsize=11)
    plt.scatter(df[x], df[y], c=color, s=size, alpha=transparency)

    plt.xlabel(xlable,fontsize=9)
    plt.ylabel(ylable,fontsize=9)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout(pad=2.0)

    plt.savefig(f'plots/{model}/unique_clear/{filename}.png', dpi=300)
    plt.clf()

    if model == 'models_mean' or model == 'mean_without_videocon':
        continue

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    fig.suptitle(title, fontsize=11)

    for r in range(4):
        if r > 1:
            plt.setp(ax[r], title=f'Round {r + 2}')

            x = re.sub('_[1-6]', f'_{str(r + 2)}', x)
            y = re.sub('_[1-6]', f'_{str(r + 2)}', y)
        else:
            plt.setp(ax[r], title=f'Round {r + 1}')

            x = re.sub('_[1-6]', f'_{str(r + 1)}', x)
            y = re.sub('_[1-6]', f'_{str(r + 1)}', y)

        ax[r].scatter(df[x], df[y], c=color, s=size, alpha=transparency)

        ax[r].set_xlabel(xlable, fontsize=9)
        ax[r].set_ylabel(ylable, fontsize=9)
        ax[r].set_xlim(xlim)
        ax[r].set_ylim(ylim)
        fig.set_tight_layout(tight=True)

    plt.savefig(f'plots/{model}/multiround_clear/{filename}.png', dpi=300)
    plt.clf()

    matplotlib.pyplot.close()