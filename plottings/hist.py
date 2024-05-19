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

come_fare = 'videocon_synth conditioned r1 ent'

data1 = 'clip_flant real unconditioned r1 ent'
data2 = 'clip_flant real conditioned r1 ent'

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


plt.hist(df[data1]-df[data2],bins=100,color='#2ad0f5')
plt.title('clip_flant real diff r1 (unconditioned-conditioned)')

plt.savefig('plots/histograms/clip_flant_real_diff_r1.png')
