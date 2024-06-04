import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')


ff_sets = ['ff_entail_diff_llava', 'ff_entail_diff_clip_flant', 'ff_entail_diff_instructblip_flant',
           'ff_cap_llava', 'ff_neg_cap_llava', 'ff_cap_clip_flant', 'ff_neg_cap_clip_flant',
           'ff_cap_instructblip_flant', 'ff_neg_cap_instructblip_flant']
video_sets = ['videocon_synth', 'videocon_real',
              'clip_flant synth', 'clip_flant real',
              'instructblip_flant synth','instructblip_flant real',
              'llava synth', 'llava real']

come_fare = 'videocon(Vc_1,S)'

data1 = 'instructblip_flant(Vc_r1,R)'
data2 = 'clip_flant(Vc_1,R)'

# TODO fare cartelle per modelli e automatizzare anche questo

# Unconditioned::=  REAL = #75fa85    |  SYNTH = #f58b45   |  DIFF = pink
# Conditioned::=  REAL = #2af542    |  SYNTH = #e84c31   |  DIFF = magenta


plt.hist(df[data1],bins=100,color='#2ad0f5')
plt.title('instructblip_flant(Vc_r1,R)')

plt.savefig('plots/histograms/instructblip_flant(Vc_r1,R).png')
