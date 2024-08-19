import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','scatter'])

df = pd.read_csv('../../with_mean.csv')

a = pd.DataFrame()
c = 0
df = df.sort_values('clip_flant(Vc_mean,S)',ascending=True)
df = df.reset_index(drop=True)
for index,row in df.iterrows():
    c += 1
    if c >= 23:
        b1 = (df['clip_flant(Vc_mean,S)'].at[index] + df['clip_flant(Vc_mean,S)'].at[index - 1] +
              df['clip_flant(Vc_mean,S)'].at[index - 2] +
              df['clip_flant(Vc_mean,S)'].at[index - 3] +
              df['clip_flant(Vc_mean,S)'].at[index - 4] +
              df['clip_flant(Vc_mean,S)'].at[index - 5] +
              df['clip_flant(Vc_mean,S)'].at[index - 6] +
              df['clip_flant(Vc_mean,S)'].at[index - 7] +
              df['clip_flant(Vc_mean,S)'].at[index - 8] +
              df['clip_flant(Vc_mean,S)'].at[index - 9] +
              df['clip_flant(Vc_mean,S)'].at[index - 10] +
              df['clip_flant(Vc_mean,S)'].at[index - 11] +
              df['clip_flant(Vc_mean,S)'].at[index - 12] +
              df['clip_flant(Vc_mean,S)'].at[index - 13] +
              df['clip_flant(Vc_mean,S)'].at[index - 14] +
              df['clip_flant(Vc_mean,S)'].at[index - 15] +
              df['clip_flant(Vc_mean,S)'].at[index - 16] +
              df['clip_flant(Vc_mean,S)'].at[index - 17] +
              df['clip_flant(Vc_mean,S)'].at[index - 18] +
              df['clip_flant(Vc_mean,S)'].at[index - 19] +
              df['clip_flant(Vc_mean,S)'].at[index - 20] +
              df['clip_flant(Vc_mean,S)'].at[index - 21] +
              df['clip_flant(Vc_mean,S)'].at[index - 22]
              )
        b2 = (df['videocon(Vc_mean,S)'].at[index] + df['videocon(Vc_mean,S)'].at[index - 1] +
              df['videocon(Vc_mean,S)'].at[index - 2] +
              df['videocon(Vc_mean,S)'].at[index - 3] + df['videocon(Vc_mean,S)'].at[index - 4] +
              df['videocon(Vc_mean,S)'].at[index - 5] +
              df['videocon(Vc_mean,S)'].at[index - 6] + df['videocon(Vc_mean,S)'].at[index - 7] +
              df['videocon(Vc_mean,S)'].at[index - 8] +
              df['videocon(Vc_mean,S)'].at[index - 9] +
              df['videocon(Vc_mean,S)'].at[index - 10] +
              df['videocon(Vc_mean,S)'].at[index - 11] +
              df['videocon(Vc_mean,S)'].at[index - 12] +
              df['videocon(Vc_mean,S)'].at[index - 13] +
              df['videocon(Vc_mean,S)'].at[index - 14] +
              df['videocon(Vc_mean,S)'].at[index - 15] +
              df['videocon(Vc_mean,S)'].at[index - 16] +
              df['videocon(Vc_mean,S)'].at[index - 17] +
              df['videocon(Vc_mean,S)'].at[index - 18] +
              df['videocon(Vc_mean,S)'].at[index - 19] +
              df['videocon(Vc_mean,S)'].at[index - 20] +
              df['videocon(Vc_mean,S)'].at[index - 21] +
              df['videocon(Vc_mean,S)'].at[index - 22]
              )
        b3 = (df['instructblip(Vc_mean,S)'].at[index] + df['instructblip(Vc_mean,S)'].at[index - 1] +
              df['instructblip(Vc_mean,S)'].at[index - 2] +
              df['instructblip(Vc_mean,S)'].at[index - 3] + df['instructblip(Vc_mean,S)'].at[index - 4] +
              df['instructblip(Vc_mean,S)'].at[index - 5] +
              df['instructblip(Vc_mean,S)'].at[index - 6] + df['instructblip(Vc_mean,S)'].at[index - 7] +
              df['instructblip(Vc_mean,S)'].at[index - 8] +
              df['instructblip(Vc_mean,S)'].at[index - 9] +
              df['instructblip(Vc_mean,S)'].at[index - 10] +
              df['instructblip(Vc_mean,S)'].at[index - 11] +
              df['instructblip(Vc_mean,S)'].at[index - 12] +
              df['instructblip(Vc_mean,S)'].at[index - 13] +
              df['instructblip(Vc_mean,S)'].at[index - 14] +
              df['instructblip(Vc_mean,S)'].at[index - 15] +
              df['instructblip(Vc_mean,S)'].at[index - 16] +
              df['instructblip(Vc_mean,S)'].at[index - 17] +
              df['instructblip(Vc_mean,S)'].at[index - 18] +
              df['instructblip(Vc_mean,S)'].at[index - 19] +
              df['instructblip(Vc_mean,S)'].at[index - 20] +
              df['instructblip(Vc_mean,S)'].at[index - 21] +
              df['instructblip(Vc_mean,S)'].at[index - 22]
              )
        b4 = (df['llava(Vc_mean,S)'].at[index] + df['llava(Vc_mean,S)'].at[index - 1] +
              df['llava(Vc_mean,S)'].at[index - 2] +
              df['llava(Vc_mean,S)'].at[index - 3] + df['llava(Vc_mean,S)'].at[index - 4] +
              df['llava(Vc_mean,S)'].at[index - 5] +
              df['llava(Vc_mean,S)'].at[index - 6] + df['llava(Vc_mean,S)'].at[index - 7] +
              df['llava(Vc_mean,S)'].at[index - 8] +
              df['llava(Vc_mean,S)'].at[index - 9] +
              df['llava(Vc_mean,S)'].at[index - 10] +
              df['llava(Vc_mean,S)'].at[index - 11] +
              df['llava(Vc_mean,S)'].at[index - 12] +
              df['llava(Vc_mean,S)'].at[index - 13] +
              df['llava(Vc_mean,S)'].at[index - 14] +
              df['llava(Vc_mean,S)'].at[index - 15] +
              df['llava(Vc_mean,S)'].at[index - 16] +
              df['llava(Vc_mean,S)'].at[index - 17] +
              df['llava(Vc_mean,S)'].at[index - 18] +
              df['llava(Vc_mean,S)'].at[index - 19] +
              df['llava(Vc_mean,S)'].at[index - 20] +
              df['llava(Vc_mean,S)'].at[index - 21] +
              df['llava(Vc_mean,S)'].at[index - 22]
              )
        a = pd.concat([a, pd.DataFrame([{'clip_flant(Vc_mean,S)':  b1/23,'videocon(Vc_mean,S)': b2/23
                                           ,'instructblip(Vc_mean,S)': b3/23,'llava(Vc_mean,S)': b4/23}])],ignore_index=True)


size = 0.15
transparency = 0.7
plt.rcParams.update({'font.size': 5})

plt.title("Models correlation for conditional videos",fontsize=7)
plt.scatter(a.index.values, a['videocon(Vc_mean,S)'],c='#f79410',s=size,alpha=transparency,marker='+',label="videocon")
plt.scatter(a.index.values, a['instructblip(Vc_mean,S)'],c='#0cb14d',s=size,alpha=transparency,marker='x',label="instructblip")
plt.scatter(a.index.values, a['llava(Vc_mean,S)'],c='#145d9e',s=size,alpha=transparency,marker='+',label= "llava")
plt.scatter(a.index.values, a['clip_flant(Vc_mean,S)'],c='#f13511',s=size,alpha=transparency,marker='x',label="clip_flant")

plt.xlabel('video caption pair')
plt.ylabel('$E(V_S^C,T_S)$')

legend = plt.legend(facecolor="pink", loc=2,markerscale=10)
plt.savefig(f'P1_models_correlation_filtered.png', dpi=300)
plt.clf()


