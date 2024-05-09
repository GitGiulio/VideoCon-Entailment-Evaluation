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

csv_scores = {
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
            pd.read_csv(csv_path['conditioned']['videocon_real']['r5'])],
        'clip_flant': [
            pd.read_csv(csv_path['conditioned']['clip_flant']['r1']),
            pd.read_csv(csv_path['conditioned']['clip_flant']['r2']),
            pd.read_csv(csv_path['conditioned']['clip_flant']['r3']),
            pd.read_csv(csv_path['conditioned']['clip_flant']['r4']),
            pd.read_csv(csv_path['conditioned']['clip_flant']['r5'])],
        'instructblip_flant': [
            pd.read_csv(csv_path['conditioned']['instructblip_flant']['r1']),
            pd.read_csv(csv_path['conditioned']['instructblip_flant']['r2']),
            pd.read_csv(csv_path['conditioned']['instructblip_flant']['r3']),
            pd.read_csv(csv_path['conditioned']['instructblip_flant']['r4']),
            pd.read_csv(csv_path['conditioned']['instructblip_flant']['r5'])],
        'llava': [
            pd.read_csv(csv_path['conditioned']['llava']['r1']),
            pd.read_csv(csv_path['conditioned']['llava']['r2']),
            pd.read_csv(csv_path['conditioned']['llava']['r3']),
            pd.read_csv(csv_path['conditioned']['llava']['r4']),
            pd.read_csv(csv_path['conditioned']['llava']['r5'])]
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
            pd.read_csv(csv_path['unconditioned']['videocon_real']['r5'])],
        'clip_flant': [
            pd.read_csv(csv_path['unconditioned']['clip_flant']['r1']),
            pd.read_csv(csv_path['unconditioned']['clip_flant']['r2']),
            pd.read_csv(csv_path['unconditioned']['clip_flant']['r3']),
            pd.read_csv(csv_path['unconditioned']['clip_flant']['r4']),
            pd.read_csv(csv_path['unconditioned']['clip_flant']['r5'])],
        'instructblip_flant': [
            pd.read_csv(csv_path['unconditioned']['instructblip_flant']['r1']),
            pd.read_csv(csv_path['unconditioned']['instructblip_flant']['r2']),
            pd.read_csv(csv_path['unconditioned']['instructblip_flant']['r3']),
            pd.read_csv(csv_path['unconditioned']['instructblip_flant']['r4']),
            pd.read_csv(csv_path['unconditioned']['instructblip_flant']['r5'])],
        'llava': [
            pd.read_csv(csv_path['unconditioned']['llava']['r1']),
            pd.read_csv(csv_path['unconditioned']['llava']['r2']),
            pd.read_csv(csv_path['unconditioned']['llava']['r3']),
            pd.read_csv(csv_path['unconditioned']['llava']['r4']),
            pd.read_csv(csv_path['unconditioned']['llava']['r5'])]
    }
}

dataset = pd.read_csv('data/my_new_dataset.csv')

dataset[f'videocon_synth conditioned r1 ent'] = 0
dataset[f'videocon_synth conditioned r2 ent'] = 0
dataset[f'videocon_synth conditioned r3 ent'] = 0
dataset[f'videocon_synth conditioned r4 ent'] = 0
dataset[f'videocon_synth conditioned r5 ent'] = 0

dataset[f'videocon_synth unconditioned r1 ent'] = 0
dataset[f'videocon_synth unconditioned r2 ent'] = 0
dataset[f'videocon_synth unconditioned r3 ent'] = 0
dataset[f'videocon_synth unconditioned r4 ent'] = 0
dataset[f'videocon_synth unconditioned r5 ent'] = 0

dataset[f'videocon_real conditioned r1 ent'] = 0
dataset[f'videocon_real conditioned r2 ent'] = 0
dataset[f'videocon_real conditioned r3 ent'] = 0
dataset[f'videocon_real conditioned r4 ent'] = 0
dataset[f'videocon_real conditioned r5 ent'] = 0

dataset[f'videocon_real unconditioned r1 ent'] = 0
dataset[f'videocon_real unconditioned r2 ent'] = 0
dataset[f'videocon_real unconditioned r3 ent'] = 0
dataset[f'videocon_real unconditioned r4 ent'] = 0
dataset[f'videocon_real unconditioned r5 ent'] = 0

dataset[f'clip_flant synth conditioned r1 ent'] = 0
dataset[f'clip_flant synth conditioned r2 ent'] = 0
dataset[f'clip_flant synth conditioned r3 ent'] = 0
dataset[f'clip_flant synth conditioned r4 ent'] = 0
dataset[f'clip_flant synth conditioned r5 ent'] = 0

dataset[f'clip_flant synth unconditioned r1 ent'] = 0
dataset[f'clip_flant synth unconditioned r2 ent'] = 0
dataset[f'clip_flant synth unconditioned r3 ent'] = 0
dataset[f'clip_flant synth unconditioned r4 ent'] = 0
dataset[f'clip_flant synth unconditioned r5 ent'] = 0

dataset[f'clip_flant real conditioned r1 ent'] = 0
dataset[f'clip_flant real conditioned r2 ent'] = 0
dataset[f'clip_flant real conditioned r3 ent'] = 0
dataset[f'clip_flant real conditioned r4 ent'] = 0
dataset[f'clip_flant real conditioned r5 ent'] = 0

dataset[f'clip_flant real unconditioned r1 ent'] = 0
dataset[f'clip_flant real unconditioned r2 ent'] = 0
dataset[f'clip_flant real unconditioned r3 ent'] = 0
dataset[f'clip_flant real unconditioned r4 ent'] = 0
dataset[f'clip_flant real unconditioned r5 ent'] = 0

dataset[f'instructblip_flant synth conditioned r1 ent'] = 0
dataset[f'instructblip_flant synth conditioned r2 ent'] = 0
dataset[f'instructblip_flant synth conditioned r3 ent'] = 0
dataset[f'instructblip_flant synth conditioned r4 ent'] = 0
dataset[f'instructblip_flant synth conditioned r5 ent'] = 0

dataset[f'instructblip_flant synth unconditioned r1 ent'] = 0
dataset[f'instructblip_flant synth unconditioned r2 ent'] = 0
dataset[f'instructblip_flant synth unconditioned r3 ent'] = 0
dataset[f'instructblip_flant synth unconditioned r4 ent'] = 0
dataset[f'instructblip_flant synth unconditioned r5 ent'] = 0

dataset[f'instructblip_flant real conditioned r1 ent'] = 0
dataset[f'instructblip_flant real conditioned r2 ent'] = 0
dataset[f'instructblip_flant real conditioned r3 ent'] = 0
dataset[f'instructblip_flant real conditioned r4 ent'] = 0
dataset[f'instructblip_flant real conditioned r5 ent'] = 0

dataset[f'instructblip_flant real unconditioned r1 ent'] = 0
dataset[f'instructblip_flant real unconditioned r2 ent'] = 0
dataset[f'instructblip_flant real unconditioned r3 ent'] = 0
dataset[f'instructblip_flant real unconditioned r4 ent'] = 0
dataset[f'instructblip_flant real unconditioned r5 ent'] = 0

dataset[f'llava synth conditioned r1 ent'] = 0
dataset[f'llava synth conditioned r2 ent'] = 0
dataset[f'llava synth conditioned r3 ent'] = 0
dataset[f'llava synth conditioned r4 ent'] = 0
dataset[f'llava synth conditioned r5 ent'] = 0

dataset[f'llava synth unconditioned r1 ent'] = 0
dataset[f'llava synth unconditioned r2 ent'] = 0
dataset[f'llava synth unconditioned r3 ent'] = 0
dataset[f'llava synth unconditioned r4 ent'] = 0
dataset[f'llava synth unconditioned r5 ent'] = 0

dataset[f'llava real conditioned r1 ent'] = 0
dataset[f'llava real conditioned r2 ent'] = 0
dataset[f'llava real conditioned r3 ent'] = 0
dataset[f'llava real conditioned r4 ent'] = 0
dataset[f'llava real conditioned r5 ent'] = 0

dataset[f'llava real unconditioned r1 ent'] = 0
dataset[f'llava real unconditioned r2 ent'] = 0
dataset[f'llava real unconditioned r3 ent'] = 0
dataset[f'llava real unconditioned r4 ent'] = 0
dataset[f'llava real unconditioned r5 ent'] = 0

print(dataset.shape)


for type in csv_scores:
    for set in csv_scores[type]:
        j = 0
        for round in csv_scores[type][set]:
            print(f"{type} {set}")
            j += 1
            for index,row in dataset.iterrows():
                if set == 'videocon_synth':
                    synth_cap_entail = round.loc[(round['videopath'] == row.videopath) & (round['text'] == row.neg_caption)]['entailment'].values
                    if len(synth_cap_entail) != 0:
                        dataset[f'{set} {type} r{j} ent'].at[index] = synth_cap_entail[0]
                    else:
                        print(f'ACHTUNG{index}  set {set}  type {type}  r{j}')
                        print(f'cap {synth_cap_entail}')
                        print(f'row {row}')
                        dataset[f'{set} {type} r{j} ent'].at[index] = -1
                elif set == 'videocon_real':
                    real_cap_entail = round.loc[(round['videopath'] == row.videopath) & (round['text'] == row.caption)]['entailment'].values
                    if len(real_cap_entail) != 0:
                        dataset[f'{set} {type} r{j} ent'].at[index] = real_cap_entail[0]
                    else:
                        print(f'ACHTUNG{index}  set {set}  type {type}  r{j}')
                        print(f'cap {real_cap_entail}')
                        print(f'row {row}')
                        dataset[f'{set} {type} r{j} ent'].at[index] = -1
                else:
                    synth_caption_entail = round.loc[(round['videopath'] == row.videopath) & (round['text'] == row.neg_caption)]['entailment'].values
                    real_caption_entail = round.loc[(round['videopath'] == row.videopath) & (round['text'] == row.caption)]['entailment'].values
                    if len(synth_caption_entail) != 0:
                        dataset[f'{set} synth {type} r{j} ent'].at[index] = synth_caption_entail[0]
                    else:
                        print(f'ACHTUNG{index}  set {set}  type {type}  r{j}')
                        print(f'synth {synth_caption_entail}')
                        dataset[f'{set} synth {type} r{j} ent'].at[index] = -1
                    if len(real_caption_entail) != 0:
                        dataset[f'{set} real {type} r{j} ent'].at[index] = real_caption_entail[0]
                    else:
                        print(f'ACHTUNG{index}  set {set}  type {type}  r{j}')
                        print(f'real {real_caption_entail}')
                        dataset[f'{set} real {type} r{j} ent'].at[index] = -1

print(dataset.shape)


dataset.to_csv('final_dataset.csv', index=False)

print(dataset.shape)
