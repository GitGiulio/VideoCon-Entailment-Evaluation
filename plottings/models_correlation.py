import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee','no-latex'])

df = pd.read_csv('../data/complete_df.csv')


size = 0.007
transparency = 0.9

ds = df.sort_values('llava(Vc_1,R)',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("Models correlation",fontsize=9)
plt.scatter(ds.index.values, ds['videocon(Vc_1,R)'],c='#145d9e',s=size/2,alpha=transparency,marker='o')
plt.scatter(ds.index.values, ds['instructblip(Vc_1,R)'],c='#0cb14d',s=size,alpha=transparency,marker='o')
plt.scatter(ds.index.values, ds['llava(Vc_1,R)'],c='#f13511',s=size,alpha=transparency,marker='o')
plt.scatter(ds.index.values, ds['clip_flant(Vc_1,R)'],c='#f79410',s=size,alpha=transparency,marker='o')

plt.tight_layout(pad=2.0)

plt.savefig(f'plots/models_correlations/v_r_c.png', dpi=600)
plt.clf()

# TODO fare un grafico che mostra il valore medio n per n come sopra cosi da eliminare un po' di rumore

