import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import re
import scienceplots

plt.style.use(['science', 'ieee'])

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

for round in [1,2,4,5]:
    df[f'D(clip_flant(Vc_{round},R),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vc_{round},R)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vu_{round},S))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vu_{round},S)']
    df[f'D(clip_flant(Vu_{round},S),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vu_{round},S)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vc_{round},R))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vc_{round},R)']
    df[f'D(llava(Vc_{round},R),llava(Vc_{round},S))'] = df[f'llava(Vc_{round},R)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vu_{round},S))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vu_{round},S)']
    df[f'D(llava(Vu_{round},S),llava(Vc_{round},S))'] = df[f'llava(Vu_{round},S)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vc_{round},R))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vc_{round},R)']
    df[f'D(instructblip(Vc_{round},R),instructblip(Vc_{round},S))'] = df[f'instructblip(Vc_{round},R)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vu_{round},S))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vu_{round},S)']
    df[f'D(instructblip(Vu_{round},S),instructblip(Vc_{round},S))'] = df[f'instructblip(Vu_{round},S)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vc_{round},R))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vc_{round},R)']

df[f'D(mean(Vc,R),mean(Vc,S))'] = df[f'mean(Vc,R)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vu,S))'] = df[f'mean(Vu,R)'] - df[f'mean(Vu,S)']
df[f'D(mean(Vu,S),mean(Vc,S))'] = df[f'mean(Vu,S)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vc,R))'] = df[f'mean(Vu,R)'] - df[f'mean(Vc,R)']

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


for x,y,title,xlable,ylable,model,color1,xlim,ylim,filename in zip(X,Y,TITLES,XLABELS,YLABELS,MODELS,COLORS,XLIMS,YLIMS,FILENAMES):

    if model == 'clip_flant':
        color = '#f13511'
    elif model == 'llava':
        color = '#f79410'
    elif model == 'instructblip_flant':
        color = '#0cb14d'
    elif model == 'videocon':
        color = '#145d9e'
    elif model == 'models_mean':
        color = '#4a4a4a'
    elif model == 'mean_without_videocon':
        color = '#7d5f8d'
    else:
        color = color1
    size = 0.05
    transparency = 0.6

    plt.title(title,fontsize=9)
    plt.scatter(df[x], df[y], s=size,c=color,marker='>', alpha=transparency)

    plt.xlabel(xlable,fontsize=7)
    plt.ylabel(ylable,fontsize=7)
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