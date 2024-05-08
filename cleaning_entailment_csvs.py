import pandas as pd

csv_path = {
    'conditioned':{
        'videocon_synth':{
            'r1': 'data/entailment/final_videocon_s001_conditioned_synth-synth_scores.csv',
            'r2': 'data/entailment/final_videocon_s002_conditioned_synth-synth_scores.csv',
            'r3': 'data/entailment/final_videocon_s003_conditioned_synth-synth_scores.csv',
            'r4': 'data/entailment/final_videocon_s004_conditioned_synth-synth_scores.csv',
            'r5': 'data/entailment/final_videocon_s005_conditioned_synth-synth_scores.csv'},
        'videocon_real':{
            'r1': 'data/entailment/final_videocon_s001_conditioned_synth-real_scores.csv',
            'r2': 'data/entailment/final_videocon_s002_conditioned_synth-real_scores.csv',
            'r3': 'data/entailment/final_videocon_s003_conditioned_synth-real_scores.csv',
            'r4': 'data/entailment/final_videocon_s004_conditioned_synth-real_scores.csv',
            'r5': 'data/entailment/final_videocon_s005_conditioned_synth-real_scores.csv'},
        'clip_flant':{
            'r1': 'data/entailment/videocon_s001_conditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_conditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_conditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_conditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_conditioned_clip-flant5-xxl_sample_4_frame.csv'},
        'instructblip_flant':{
            'r1': 'data/entailment/videocon_s001_conditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_conditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_conditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_conditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_conditioned_instructblip-flant5-xxl_sample_4_frame.csv'},
        'llava':{
            'r1': 'data/entailment/videocon_s001_conditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_conditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_conditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_conditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_conditioned_llava-v1.5-13b_sample_4_frame.csv'},
    },
    'unconditioned':{
        'videocon_synth':{
            'r1': 'data/entailment/final_videocon_s001_unconditioned_synth-synth_scores.csv',
            'r2': 'data/entailment/final_videocon_s002_unconditioned_synth-synth_scores.csv',
            'r3': 'data/entailment/final_videocon_s003_unconditioned_synth-synth_scores.csv',
            'r4': 'data/entailment/final_videocon_s004_unconditioned_synth-synth_scores.csv',
            'r5': 'data/entailment/final_videocon_s005_unconditioned_synth-synth_scores.csv'},
        'videocon_real':{
            'r1': 'data/entailment/final_videocon_s001_unconditioned_synth-real_scores.csv',
            'r2': 'data/entailment/final_videocon_s002_unconditioned_synth-real_scores.csv',
            'r3': 'data/entailment/final_videocon_s003_unconditioned_synth-real_scores.csv',
            'r4': 'data/entailment/final_videocon_s004_unconditioned_synth-real_scores.csv',
            'r5': 'data/entailment/final_videocon_s005_unconditioned_synth-real_scores.csv'},
        'clip_flant':{
            'r1': 'data/entailment/videocon_s001_unconditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_unconditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_unconditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_unconditioned_clip-flant5-xxl_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_unconditioned_clip-flant5-xxl_sample_4_frame.csv'},
        'instructblip_flant':{
            'r1': 'data/entailment/videocon_s001_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv'},
        'llava':{
            'r1': 'data/entailment/videocon_s001_unconditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r2': 'data/entailment/videocon_s002_unconditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r3': 'data/entailment/videocon_s003_unconditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r4': 'data/entailment/videocon_s004_unconditioned_llava-v1.5-13b_sample_4_frame.csv',
            'r5': 'data/entailment/videocon_s005_unconditioned_llava-v1.5-13b_sample_4_frame.csv'},
    }
}

csv_videocon_scores = {
    'conditioned': {
        'videocon_synth': [
            pd.read_csv(csv_path['conditioned']['videocon_synth']['r1']),
            pd.read_csv(csv_path['conditioned']['videocon_synth']['r2']),
            pd.read_csv(csv_path['conditioned']['videocon_synth']['r3']),
            pd.read_csv(csv_path['conditioned']['videocon_synth']['r4']),
            pd.read_csv(csv_path['conditioned']['videocon_synth']['r5'])],
        'videocon_real': [
            pd.read_csv(csv_path['conditioned']['videocon_real']['r1']),
            pd.read_csv(csv_path['conditioned']['videocon_real']['r2']),
            pd.read_csv(csv_path['conditioned']['videocon_real']['r3']),
            pd.read_csv(csv_path['conditioned']['videocon_real']['r4']),
            pd.read_csv(csv_path['conditioned']['videocon_real']['r5'])]
    },
    'unconditioned': {
        'videocon_synth': [
            pd.read_csv(csv_path['unconditioned']['videocon_synth']['r1']),
            pd.read_csv(csv_path['unconditioned']['videocon_synth']['r2']),
            pd.read_csv(csv_path['unconditioned']['videocon_synth']['r3']),
            pd.read_csv(csv_path['unconditioned']['videocon_synth']['r4']),
            pd.read_csv(csv_path['unconditioned']['videocon_synth']['r5'])],
        'videocon_real': [
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r1']),
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r2']),
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r3']),
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r4']),
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r5'])]
    }
}

for set in csv_videocon_scores:
    for type in csv_videocon_scores[set]:
        i = 0
        for round in csv_videocon_scores[set][type]:
            i += 1
            for index, row in round.iterrows():
                pos = row['videopath'].find('highres_')
                videopath_temp = row['videopath']
                round.loc[index,'videopath'] = videopath_temp[pos+8:len(videopath_temp)-10]
                conversation_temp = row['text']
                round.loc[index, 'text'] = conversation_temp[222:len(conversation_temp)-7]
            round.to_csv(csv_path[set][type][f'r{i}'], index=False)
