import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee','no-latex'])

df = pd.read_csv('../data/complete_df.csv')


size = 0.1
transparency = 1

ds = df.sort_values('videocon(Vc_1,R)',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("Models correlation",fontsize=9)
plt.scatter(ds.index, ds['videocon(Vc_1,R)'],c='#145d9e',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['instructblip(Vc_1,R)'],c='#0cb14d',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['clip_flant(Vc_1,R)'],c='#f79410',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['llava(Vc_1,R)'],c='#f13511',s=size,alpha=transparency)

plt.tight_layout(pad=2.0)

plt.savefig(f'plots/models_correlations/v_r_c_2.png', dpi=600)
plt.clf()

