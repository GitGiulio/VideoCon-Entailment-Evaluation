import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

statistics_df = pd.read_csv('../data/complete_df.csv')

statistics_df['entailments_sum'] = (statistics_df['conditioned round1 entailment'] + statistics_df['conditioned round2 entailment'] +
                                    statistics_df['conditioned round3 entailment'] + statistics_df['conditioned round4 entailment'] +
                                    statistics_df['conditioned round5 entailment'] + statistics_df['unconditioned round1 entailment'] +
                                    statistics_df['unconditioned round2 entailment'] + statistics_df['unconditioned round3 entailment'] +
                                    statistics_df['unconditioned round4 entailment'] + statistics_df['unconditioned round5 entailment'])

statistics_df['conditioned_sum'] = (statistics_df['conditioned round1 entailment'] + statistics_df['conditioned round2 entailment'] +
                                    statistics_df['conditioned round3 entailment'] + statistics_df['conditioned round4 entailment'] +
                                    statistics_df['conditioned round5 entailment'])

statistics_df['unconditioned_sum'] = (statistics_df['unconditioned round1 entailment'] + statistics_df['unconditioned round2 entailment'] +
                                      statistics_df['unconditioned round3 entailment'] + statistics_df['unconditioned round4 entailment'] +
                                      statistics_df['unconditioned round5 entailment'])


statistics_df['entailments_mean'] = statistics_df['entailments_sum'] / 10
statistics_df['conditioned_mean'] = statistics_df['conditioned_sum'] / 10
statistics_df['unconditioned_mean'] = statistics_df['unconditioned_sum'] / 10

statistics_df['entailments_standard_deviation'] = (
        (statistics_df['conditioned round1 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['conditioned round2 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['conditioned round3 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['conditioned round4 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['conditioned round5 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['unconditioned round1 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['unconditioned round2 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['unconditioned round3 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['unconditioned round4 entailment'] - statistics_df['entailments_mean'])**2
        + (statistics_df['unconditioned round5 entailment'] - statistics_df['entailments_mean'])**2
        ) / 9

#plt.scatter(statistics_df['conditioned round1 entailment'],statistics_df['unconditioned round1 entailment'],c='blue',alpha=0.5)
#plt.bar([statistics_df['conditioned round1 entailment'].std(),
#         statistics_df['conditioned round2 entailment'].std(),
#         statistics_df['conditioned round3 entailment'].std(),
#         statistics_df['conditioned round4 entailment'].std(),
#         statistics_df['conditioned round5 entailment'].std(),
#         statistics_df['unconditioned round1 entailment'].std(),
#         statistics_df['unconditioned round2 entailment'].std(),
#         statistics_df['unconditioned round3 entailment'].std(),
#         statistics_df['unconditioned round4 entailment'].std(),
#         statistics_df['unconditioned round5 entailment'].std()],
#        ['cr1','cr2','cr3','cr4','cr5','ur1','ur2','ur3','ur4','ur5'])

statistics_df['ff_entail_diff_llava'] = statistics_df['ff_cap_llava'] - statistics_df['ff_neg_cap_llava']
statistics_df['ff_entail_diff_clip_flant'] = statistics_df['ff_cap_clip_flant'] - statistics_df['ff_neg_cap_clip_flant']
statistics_df['ff_entail_diff_instructblip_flant'] = statistics_df['ff_cap_instructblip_flant'] - statistics_df['ff_neg_cap_instructblip_flant']

statistics_df['mean entailment diff'] = statistics_df['unconditioned_mean'] - statistics_df['conditioned_mean']


fig, axs = plt.subplots(1, 3,figsize=(15,5))

fig.suptitle('ff entailment diff vs mean entailment diff')

axs[0].scatter(statistics_df['ff_entail_diff_clip_flant'],statistics_df['mean entailment diff'],alpha=0.3,s=1)
axs[0].set_title('ff entailment diff vs mean entailment diff')
axs[0].set(xlabel='first frame entailment difference (clip-flant)', ylabel='unconditioned video mean - conditioned video mean)')

axs[1].scatter(statistics_df['ff_entail_diff_clip_flant'],statistics_df['mean entailment diff'],alpha=0.3,s=1)
axs[1].set_title('ff entailment diff vs mean entailment diff')
axs[1].set(xlabel='first frame entailment difference (clip-flant)', ylabel='unconditioned video mean - conditioned video mean)')

axs[2].scatter(statistics_df['ff_entail_diff_clip_flant'],statistics_df['mean entailment diff'],alpha=0.3,s=1)
axs[2].set_title('ff entailment diff vs mean entailment diff')
axs[2].set(xlabel='first frame entailment difference (clip-flant)', ylabel='unconditioned video mean - conditioned video mean)')

plt.show()

print(statistics_df['entailments_mean'].std())

#statistics_df.to_csv('../data/statistics_df.csv')