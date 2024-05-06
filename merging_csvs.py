import pandas as pd

dataset = pd.read_csv('my_new_dataset.csv')

csv_path = {
    'conditioned': {'r1': 'synthetic_video_entailment_results/final_videocon_s001_conditioned_scores.csv',
                    'r2': 'synthetic_video_entailment_results/final_videocon_s002_conditioned_scores.csv',
                    'r3': 'synthetic_video_entailment_results/final_videocon_s003_conditioned_scores.csv',
                    'r4': 'synthetic_video_entailment_results/final_videocon_s004_conditioned_scores.csv',
                    'r5': 'synthetic_video_entailment_results/final_videocon_s005_conditioned_scores.csv'},
    'unconditioned': {'r1': 'synthetic_video_entailment_results/final_videocon_s001_unconditioned_scores.csv',
                      'r2': 'synthetic_video_entailment_results/final_videocon_s002_unconditioned_scores.csv',
                      'r3': 'synthetic_video_entailment_results/final_videocon_s003_unconditioned_scores.csv',
                      'r4': 'synthetic_video_entailment_results/final_videocon_s004_unconditioned_scores.csv',
                      'r5': 'synthetic_video_entailment_results/final_videocon_s005_unconditioned_scores.csv'}
}

csv_scores = {
    'conditioned': [pd.read_csv(csv_path['conditioned']['r1']),
                    pd.read_csv(csv_path['conditioned']['r2']),
                    pd.read_csv(csv_path['conditioned']['r3']),
                    pd.read_csv(csv_path['conditioned']['r4']),
                    pd.read_csv(csv_path['conditioned']['r5'])],
    'unconditioned': [pd.read_csv(csv_path['unconditioned']['r1']),
                      pd.read_csv(csv_path['unconditioned']['r2']),
                      pd.read_csv(csv_path['unconditioned']['r3']),
                      pd.read_csv(csv_path['unconditioned']['r4']),
                      pd.read_csv(csv_path['unconditioned']['r5'])]
}

dataset[f'conditioned round1 entailment'] = 0
dataset[f'conditioned round2 entailment'] = 0
dataset[f'conditioned round3 entailment'] = 0
dataset[f'conditioned round4 entailment'] = 0
dataset[f'conditioned round5 entailment'] = 0
dataset[f'unconditioned round1 entailment'] = 0
dataset[f'unconditioned round2 entailment'] = 0
dataset[f'unconditioned round3 entailment'] = 0
dataset[f'unconditioned round4 entailment'] = 0
dataset[f'unconditioned round5 entailment'] = 0

for set in csv_scores:
    j = 0
    for round in csv_scores[set]:
        j += 1
        for index,row in dataset.iterrows():
            if len((round.loc[(round['videopath'] == row.videopath) & (round['neg_caption'] == row.neg_caption)]['entailment']).values) != 0:
                dataset[f'{set} round{j} entailment'].at[index] = (round.loc[(round['videopath'] == row.videopath) & (round['neg_caption'] == row.neg_caption)]['entailment']).values[0]
            else:
                dataset[f'{set} round{j} entailment'].at[index] = -1


dataset.to_csv('final_dataset.csv', index=False)