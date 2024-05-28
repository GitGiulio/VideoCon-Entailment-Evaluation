import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

df['ff_entail_diff_llava'] = df['ff_cap_llava'] - df['ff_neg_cap_llava']
df['ff_entail_diff_clip_flant'] = df['ff_cap_clip_flant'] - df['ff_neg_cap_clip_flant']
df['ff_entail_diff_instructblip_flant'] = df['ff_cap_instructblip_flant'] - df['ff_neg_cap_instructblip_flant']

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta

inf = open('plots_instruction', 'r')

lines = inf.readlines()
X = []
Y1 = []
Y2 = []
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

    xlims_t = splitted_line[8].split(',')
    ylims_t = splitted_line[9].split(',')

    X.append(splitted_line[0])
    Y1.append(splitted_line[1])
    Y2.append(splitted_line[2])
    TITLES.append(splitted_line[3])
    XLABELS.append(splitted_line[4])
    YLABELS.append(splitted_line[5])
    MODELS.append(splitted_line[6])
    COLORS.append(splitted_line[7])
    XLIMS.append([int(xlims_t[0]),int(xlims_t[1])])
    YLIMS.append([int(ylims_t[0]),int(ylims_t[1])])
    FILENAMES.append(splitted_line[10])

for x,y1,y2,title,xlable,ylable,model,color,xlim,ylim,filename in zip(X,Y1,Y2,TITLES,XLABELS,YLABELS,MODELS,COLORS,XLIMS,YLIMS,FILENAMES):

    size = 0.9
    transparency = 0.6

    plt.title(title,fontsize=11)
    if y2 == '0':
        plt.scatter(df[x], df[y1], c=color, s=size, alpha=transparency)
    else:
        plt.scatter(df[x], df[y1] - df[y2], c=color, s=size, alpha=transparency)

    plt.xlabel(xlable,fontsize=9)
    plt.ylabel(ylable,fontsize=9)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout(pad=2.0)

    plt.savefig(f'plots/{model}/unique_clear/{filename}.png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()

for x,y1,y2,title,xlable,ylable,model,color,xlim,ylim,filename in zip(X,Y1,Y2,TITLES,XLABELS,YLABELS,MODELS,COLORS,XLIMS,YLIMS,FILENAMES):

    fig, ax = plt.subplots(1, 4,figsize=(20,5))
    size = 0.9
    transparency = 0.6

    fig.suptitle(title,fontsize=11)

    for r in range(4):
        if r > 1:
            plt.setp(ax[r], title=f'Round {r+2}')

            x = re.sub('r[1-6]', f'r{str(r+2)}', x)
            y1 = re.sub('r[1-6]', f'r{str(r+2)}', y1)
        else:
            plt.setp(ax[r], title=f'Round {r + 1}')

            x = re.sub('r[1-6]', f'r{str(r+1)}', x)
            y1 = re.sub('r[1-6]', f'r{str(r+1)}', y1)
        if y2 == '0':
            ax[r].scatter(df[x], df[y1], c=color, s=size, alpha=transparency)
        else:
            if r > 1:
                y2 = re.sub('r[1-6]', f'r{str(r+2)}', y2)
            else:
                y2 = re.sub('r[1-6]', f'r{str(r+1)}', y2)
            ax[r].scatter(df[x], df[y1] - df[y2], c=color, s=size, alpha=transparency)

        ax[r].set_xlabel(xlable,fontsize=9)
        ax[r].set_ylabel(ylable,fontsize=9)
        ax[r].set_xlim(xlim)
        ax[r].set_ylim(ylim)
        fig.set_tight_layout(tight=True)

    plt.savefig(f'plots/{model}/multiround_clear/{filename}.png', dpi=300)
    plt.clf()
    matplotlib.pyplot.close()