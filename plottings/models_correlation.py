import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee','scatter'])

df = pd.read_csv('../data/complete_df.csv')


size = 0.1
transparency = 0.6

ds = df.sort_values('clip_flant(Vc_1,R)',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("Models correlation",fontsize=7)
plt.scatter(ds.index.values, ds['videocon(Vc_1,R)'],c='#f79410',s=size,alpha=transparency,marker='+')
plt.scatter(ds.index.values, ds['instructblip(Vc_1,R)'],c='#0cb14d',s=size,alpha=transparency,marker='x')
plt.scatter(ds.index.values, ds['llava(Vc_1,R)'],c='#145d9e',s=size,alpha=transparency,marker='+')
plt.scatter(ds.index.values, ds['clip_flant(Vc_1,R)'],c='#f13511',s=size/10,alpha=transparency,marker='x')

plt.tight_layout()

plt.savefig(f'plots/models_correlations/v_r_c.png', dpi=1000)
plt.clf()

ds = df.sort_values('D(llava(F,R),llava(F,S))',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("Models correlation",fontsize=7)
plt.scatter(ds.index.values, ds['D(instructblip(F,R),instructblip(F,S))'],c='#0cb14d',s=size,alpha=transparency,marker='x')
plt.scatter(ds.index.values, ds['D(clip_flant(F,R),clip_flant(F,S))'],c='#f13511',s=size,alpha=transparency,marker='x')
plt.scatter(ds.index.values, ds['D(mean_wv(F,R),mean_wv(F,S))'],c='#ffff00',s=size,alpha=transparency,marker='x')
plt.scatter(ds.index.values, ds['D(llava(F,R),llava(F,S))'],c='#145d9e',s=size,alpha=transparency,marker='+')

plt.tight_layout()

plt.savefig(f'plots/models_correlations/diff_frames.png', dpi=1000)
# TODO fare un grafico che mostra il valore medio n per n come sopra cosi da eliminare un po' di rumore

