import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

df = pd.read_csv('../../with_mean.csv')
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

plt.suptitle('Clip-FlanT5',fontsize=13)

ax[0].hist(df['clip_flant(Vc_mean,S)'],color='#f79410',bins=200)
m0 = [df['clip_flant(Vc_mean,S)'].mean(),df['clip_flant(Vc_mean,S)'].mean()]
ax[0].plot(m0,[0,255],color='#ff0000',label='Mean')
ax[0].plot([0.5,0.5],[0,255],'k--')
ax[0].set_xlabel('$E(V_S^C,T_S)$')
ax[0].set_title('Conditioned')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,255])
ax[0].legend(facecolor="pink", loc=2,markerscale=5)

ax[1].hist(df['clip_flant(Vu_mean,S)'],color='#f79410',bins=200)
m0 = [df['clip_flant(Vu_mean,S)'].mean(),df['clip_flant(Vu_mean,S)'].mean()]
ax[1].plot(m0,[0,255],color='#ff0000',label='Mean')
ax[1].plot([0.5,0.5],[0,255],'k--')
ax[1].set_xlabel('$E(V_S^U,T_S)$')
ax[1].set_title('Unconditioned')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,255])
ax[1].legend(facecolor="pink", loc=2,markerscale=5)

plt.tight_layout()

plt.savefig(f'A2_clip.png', dpi=300)
plt.clf()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

plt.suptitle('Llava',fontsize=13)

ax[0].hist(df['llava(Vc_mean,S)'],color='#1a6fc4',bins=200)
m1 = [df['llava(Vc_mean,S)'].mean(),df['llava(Vc_mean,S)'].mean()]
ax[0].plot(m1,[0,255],color='#ff0000',label='Mean')
ax[0].plot([0.5,0.5],[0,255],'k--')
ax[0].set_xlabel('$E(V_S^C,T_S)$')
ax[0].set_title('Conditioned')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,255])
ax[0].legend(facecolor="pink", loc=2,markerscale=5)

ax[1].hist(df['llava(Vu_mean,S)'],color='#1a6fc4',bins=200)
m1 = [df['llava(Vu_mean,S)'].mean(),df['llava(Vu_mean,S)'].mean()]
ax[1].plot(m1,[0,255],color='#ff0000',label='Mean')
ax[1].plot([0.5,0.5],[0,255],'k--')
ax[1].set_xlabel('$E(V_S^U,T_S)$')
ax[1].set_title('Unconditioned')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,255])
ax[1].legend(facecolor="pink", loc=2,markerscale=5)

plt.tight_layout()

plt.savefig(f'A2_llava.png', dpi=300)
plt.clf()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))


plt.suptitle('Instructblip-Flant',fontsize=13)

ax[0].hist(df['instructblip(Vc_mean,S)'],color='#0cb14d',bins=200)
m2 = [df['instructblip(Vc_mean,S)'].mean(),df['instructblip(Vc_mean,S)'].mean()]
ax[0].plot(m2,[0,255],color='#ff0000',label='Mean')
ax[0].plot([0.5,0.5],[0,255],'k--')
ax[0].set_xlabel('$E(V_S^C,T_S)$')
ax[0].set_title('Conditioned')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,255])
ax[0].legend(facecolor="pink", loc=2,markerscale=5)

ax[1].hist(df['instructblip(Vu_mean,S)'],color='#0cb14d',bins=200)
m2 = [df['instructblip(Vu_mean,S)'].mean(),df['instructblip(Vu_mean,S)'].mean()]
ax[1].plot(m2,[0,255],color='#ff0000',label='Mean')
ax[1].plot([0.5,0.5],[0,255],'k--')
ax[1].set_xlabel('$E(V_S^U,T_S)$')
ax[1].set_title('Unconditioned')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,255])
ax[1].legend(facecolor="pink", loc=2,markerscale=5)

plt.tight_layout()

plt.savefig(f'A2_instructblip.png', dpi=300)
plt.clf()





