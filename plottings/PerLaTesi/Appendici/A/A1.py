import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

df = pd.read_csv('../../with_mean.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams.update({'font.size': 12})

fig.suptitle('Distribution of frames alignment', fontsize=15)

ax[0].hist(df['mean_wv(F,S)'],color='#7d5f8d',bins=200)
m0 = [df['mean_wv(F,S)'].mean(),df['mean_wv(F,S)'].mean()]
ax[0].plot(m0,[0,152],color='#ff0000',label='Mean')
ax[0].plot([0.5,0.5],[0,152],'k--')
ax[0].set_xlabel('$E(F,T_S)$')
ax[0].set_title('with synthetic caption')
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,152])
ax[0].legend(facecolor="pink", loc=2,markerscale=5)


ax[1].hist(df['mean_wv(F,R)'],color='#7d5f8d',bins=200)
m1 = [df['mean_wv(F,R)'].mean(),df['mean_wv(F,R)'].mean()]
ax[1].plot(m1,[0,152],color='#ff0000',label='Mean')
ax[1].plot([0.5,0.5],[0,152],'k--')
ax[1].set_xlabel('$E(F,T_R)$')
ax[1].set_title('with real caption')
ax[1].set_xlim([0,1])
ax[1].set_ylim([0,152])
ax[1].legend(facecolor="pink", loc=2,markerscale=5)

fig.set_tight_layout(tight=True)

plt.savefig(f'A1_frames_entailment_distribution.png', dpi=300)
plt.clf()