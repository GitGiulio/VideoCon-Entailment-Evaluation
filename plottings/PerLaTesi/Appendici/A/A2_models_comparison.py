import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

df = pd.read_csv('../../with_mean.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size': 12})

fig.suptitle('Conditioned video alignment distribution by model', fontsize=15)

ax[0].hist(df['clip_flant(Vc_mean,S)'],color='#f79410',bins=200)
m0 = [df['clip_flant(Vc_mean,S)'].mean(),df['clip_flant(Vc_mean,S)'].mean()]
ax[0].plot(m0,[0,255],color='#ff0000',label='Mean')
ax[0].plot([0.5,0.5],[0,255],'k--')
ax[0].set_xlabel('$E(V_S^C,T_S)$')
ax[0].set_title('Clip flant')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,255])
ax[0].legend(facecolor="pink", loc=2,markerscale=5)


ax[1].hist(df['llava(Vc_mean,S)'],color='#1a6fc4',bins=200)
m1 = [df['llava(Vc_mean,S)'].mean(),df['llava(Vc_mean,S)'].mean()]
ax[1].plot(m1,[0,255],color='#ff0000',label='Mean')
ax[1].plot([0.5,0.5],[0,255],'k--')
ax[1].set_xlabel('$E(V_S^C,T_S)$')
ax[1].set_title('Llava')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,255])
ax[1].legend(facecolor="pink", loc=2,markerscale=5)

ax[2].hist(df['instructblip(Vc_mean,S)'],color='#0cb14d',bins=200)
m2 = [df['instructblip(Vc_mean,S)'].mean(),df['instructblip(Vc_mean,S)'].mean()]
ax[2].plot(m2,[0,255],color='#ff0000',label='Mean')
ax[2].plot([0.5,0.5],[0,255],'k--')
ax[2].set_xlabel('$E(V_S^C,T_S)$')
ax[2].set_title('Instructblip')
ax[2].set_xlim([0,1])
ax[2].set_ylim([0,255])
ax[2].legend(facecolor="pink", loc=2,markerscale=5)

fig.set_tight_layout(tight=True)

plt.savefig(f'A2_models_comparison.png', dpi=300)
plt.clf()