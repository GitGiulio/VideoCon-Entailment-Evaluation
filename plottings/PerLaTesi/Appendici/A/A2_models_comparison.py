import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

def centered_subplots(rows, figsize=None):
    grid_dim = max(rows)
    grid_shape = (len(rows), 2 * grid_dim)

    if figsize:
        fig = plt.figure(figsize=(figsize))
    else:
        fig = plt.figure(figsize=(2 * grid_dim, 3 * len(rows)))

    allaxes = []

    jrow = 0
    for row in rows:
        offset = 0
        for i in range(row):
            if row < grid_dim:
                offset = grid_dim - row

            ax_position = (jrow, 2 * i + offset)
            ax = plt.subplot2grid(grid_shape, ax_position, fig=fig, colspan=2)
            allaxes.append(ax)

        jrow += 1

    return allaxes


plt.style.use(['science'])

df = pd.read_csv('../../with_mean.csv')
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax = centered_subplots([2,1],figsize=(10, 10))

plt.suptitle('Conditioned video alignment distribution by model',fontsize=13)

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

plt.tight_layout()

plt.savefig(f'A2_models_comparison.png', dpi=300)
plt.clf()