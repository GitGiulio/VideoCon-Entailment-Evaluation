import pandas as pd
import matplotlib.pyplot as plt

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

    #inf.write(f'{x}|{y1}|{y2}|{title}|{xlable}|{ylable}|{model}|{color}|{xlim[0]},{xlim[1]}|{ylim[0]},{ylim[1]}|{filename}\n')

    size = 0.9
    transparency = 0.6

    plt.title(title,fontsize=11)
    if y2 == '0':
        plt.scatter(df[x], df[y1], c=color, s=size, alpha=transparency)
    else:
        plt.scatter(df[x], df[y1] - df[y2], c=color, s=size, alpha=transparency)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #TODO capire come funziona bene
    plt.xlabel(xlable,fontsize=9)
    plt.ylabel(ylable,fontsize=9)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout(pad=2.0)
    #plt.figtext(0.5, 0.01, 'tesatasdaa', wrap=True, horizontalalignment='center', fontsize=12) #TODO capire come funziona bene

    plt.savefig(f'plots/{model}/unique_clear/{filename}.png', dpi=300)
    plt.clf()

