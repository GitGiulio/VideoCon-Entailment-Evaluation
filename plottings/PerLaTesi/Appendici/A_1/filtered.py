import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','scatter'])

df = pd.read_csv('../../with_mean.csv')

def filter(df,val):
    a = pd.DataFrame()
    c = 0
    df = df.sort_values('clip_flant(Vc_mean,S)',ascending=True)
    df = df.reset_index(drop=True)
    for index,row in df.iterrows():
        c += 1
        if c >= val:
            b1 = 0
            b2 = 0
            b3 = 0
            b4 = 0
            for i in range(val):
                b1 += df['clip_flant(Vc_mean,S)'].at[index-i]
                b2 += df['videocon(Vc_mean,S)'].at[index-i]
                b3 += df['instructblip(Vc_mean,S)'].at[index-i]
                b4 += df['llava(Vc_mean,S)'].at[index-i]
            a = pd.concat([a, pd.DataFrame([{'clip_flant(Vc_mean,S)':  b1/val,'videocon(Vc_mean,S)': b2/val
                                               ,'instructblip(Vc_mean,S)': b3/val,'llava(Vc_mean,S)': b4/val}])],ignore_index=True)
    return a

a = filter(df,77)

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


