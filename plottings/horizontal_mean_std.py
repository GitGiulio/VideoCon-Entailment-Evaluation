import pandas as pd
import matplotlib.pyplot as plt

statistics_df = pd.read_csv('final_dataset.csv')

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
plt.bar([statistics_df['conditioned round1 entailment'].std(),
         statistics_df['conditioned round2 entailment'].std(),
         statistics_df['conditioned round3 entailment'].std(),
         statistics_df['conditioned round4 entailment'].std(),
         statistics_df['conditioned round5 entailment'].std(),
         statistics_df['unconditioned round1 entailment'].std(),
         statistics_df['unconditioned round2 entailment'].std(),
         statistics_df['unconditioned round3 entailment'].std(),
         statistics_df['unconditioned round4 entailment'].std(),
         statistics_df['unconditioned round5 entailment'].std()],
        ['cr1','cr2','cr3','cr4','cr5','ur1','ur2','ur3','ur4','ur5'])
plt.show()

print(statistics_df['entailments_mean'].std())

#statistics_df.to_csv('statistics_df.csv')