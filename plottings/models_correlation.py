import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

df = pd.read_csv('../data/complete_df.csv')

# TODO automatizzare??

size = 0.6
transparency = 0.3

ds = df.sort_values('videocon(Vc_1,R)',ascending=True)

ds.reset_index(drop = True, inplace = True)

plt.title("prova",fontsize=11)
plt.scatter(ds.index, ds['videocon(Vc_1,R)'],c='g',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['instructblip(Vc_1,R)'],c='r',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['clip_flant(Vc_1,R)'],c='b',s=size,alpha=transparency)
plt.scatter(ds.index.values, ds['llava(Vc_1,R)'],c='y',s=size,alpha=transparency)

plt.tight_layout(pad=2.0)

plt.savefig(f'plots/models_correlations/v_r_c_2.png', dpi=300)
plt.clf()

