import pandas as pd

csv_path = {
    'conditioned': {'r1': 'data/synthetic_video_entailment_results/final_videocon_s001_conditioned_scores.csv',
                    'r2': 'data/synthetic_video_entailment_results/final_videocon_s002_conditioned_scores.csv',
                    'r3': 'data/synthetic_video_entailment_results/final_videocon_s003_conditioned_scores.csv',
                    'r4': 'data/synthetic_video_entailment_results/final_videocon_s004_conditioned_scores.csv',
                    'r5': 'data/synthetic_video_entailment_results/final_videocon_s005_conditioned_scores.csv'},
    'unconditioned': {'r1': 'data/synthetic_video_entailment_results/final_videocon_s001_unconditioned_scores.csv',
                      'r2': 'data/synthetic_video_entailment_results/final_videocon_s002_unconditioned_scores.csv',
                      'r3': 'data/synthetic_video_entailment_results/final_videocon_s003_unconditioned_scores.csv',
                      'r4': 'data/synthetic_video_entailment_results/final_videocon_s004_unconditioned_scores.csv',
                      'r5': 'data/synthetic_video_entailment_results/final_videocon_s005_unconditioned_scores.csv'}
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

for set in csv_scores:
    i = 0
    for round in csv_scores[set]:
        i += 1
        for index, row in round.iterrows():
            pos = row['videopath'].find('highres_')
            videopath_temp = row['videopath']
            round.loc[index,'videopath'] = videopath_temp[pos+8:len(videopath_temp)-10]
            conversation_temp = row['neg_caption']
            round.loc[index, 'neg_caption'] = conversation_temp[222:len(conversation_temp)-7]
        round.to_csv(csv_path[set][f'r{i}'], index=False)
