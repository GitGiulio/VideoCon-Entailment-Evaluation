import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

df['ff_entail_diff_llava'] = df['ff_cap_llava'] - df['ff_neg_cap_llava']
df['ff_entail_diff_clip_flant'] = df['ff_cap_clip_flant'] - df['ff_neg_cap_clip_flant']
df['ff_entail_diff_instructblip_flant'] = df['ff_cap_instructblip_flant'] - df['ff_neg_cap_instructblip_flant']

ff_sets = ['ff_entail_diff_llava', 'ff_entail_diff_clip_flant', 'ff_entail_diff_instructblip_flant',
           'ff_cap_llava', 'ff_neg_cap_llava', 'ff_cap_clip_flant', 'ff_neg_cap_clip_flant',
           'ff_cap_instructblip_flant', 'ff_neg_cap_instructblip_flant']
video_sets = ['videocon_synth', 'videocon_real',
              'clip_flant synth', 'clip_flant real',
              'instructblip_flant synth','instructblip_flant real',
              'llava synth', 'llava real']

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


x = 'ff_cap_clip_flant'
y1 = 'clip_flant synth conditioned r1 ent'
y2 = 'clip_flant real conditioned r1 ent'
model = 'clip_flant'

size = 1.0
transparency = 1.0
color = 'magenta'
#plt.rc('font', size=16)


plt.title(f'first frame REAL vs video entailment difference (SYNTH-REAL) conditioned')

plt.scatter(df[x], df[y2]-df[y1], c=color, s=size, alpha=transparency)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #TODO capire come funziona bene
plt.xlabel('entailment first frame to real caption clip-flant')
plt.ylabel('entailment difference video (real_cap-synth_cap)')
plt.xlim([0, 1])
plt.ylim([0, 1])


plt.savefig(f'plots/{model}/unique_clear/test.png', dpi=300)
