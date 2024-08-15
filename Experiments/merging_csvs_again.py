import pandas as pd

conditioned_clip = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_clip-flant5-xxl_sample_4_frame.csv')
conditioned_llava = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_llava-v1.5-13b_sample_4_frame.csv')
conditioned_instructblip = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_instructblip-flant5-xxl_sample_4_frame.csv')

unconditioned_clip = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_clip-flant5-xxl_sample_4_frame.csv')
unconditioned_instructblip = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv')
unconditioned_llava = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_llava-v1.5-13b_sample_4_frame.csv')

complete_df = pd.read_csv('../data/complete_df.csv')

conditioned_clip.rename(columns={"entailment": "conditioned_clip"})

conditioned_clip['conditioned_llava'] = 0
conditioned_clip['conditioned_instructblip'] = 0
conditioned_clip['unconditioned_clip'] = 0
conditioned_clip['unconditioned_llava'] = 0
conditioned_clip['unconditioned_clip'] = 0
conditioned_clip['D(mean_wv(F,R),mean_wv(F,S))'] = 0
conditioned_clip['caption'] = 0
conditioned_clip['unconditioned_videopath'] = 0

for index, row in conditioned_clip.iterrows():

    videopath = row['videopath']
    text = row['text']

    trimed_vp = videopath[73:-10]

    conditioned_clip['conditioned_llava'].at_index[index] = conditioned_llava.loc[
        (conditioned_llava['videopath'] == videopath) & (conditioned_llava['text'] == text)
        ]['entailment'].values
    conditioned_clip['conditioned_instructblip'].at_index[index] = conditioned_instructblip.loc[
        (conditioned_instructblip['videopath'] == videopath) & (conditioned_instructblip['text'] == text)
        ]['entailment'].values

    conditioned_clip['unconditioned_clip'].at_index[index] = unconditioned_instructblip.loc[
        (unconditioned_instructblip['videopath'][0:-10] == videopath[0:-10]) & (unconditioned_instructblip['text'] == text)
        ]['entailment'].values
    conditioned_clip['unconditioned_llava'].at_index[index] = unconditioned_llava.loc[
        (unconditioned_llava['videopath'][0:-10] == videopath[0:-10]) & (unconditioned_llava['text'] == text)
        ]['entailment'].values
    conditioned_clip['unconditioned_clip'].at_index[index] = unconditioned_clip.loc[
        (unconditioned_clip['videopath'][0:-10] == videopath[0:-10]) & (unconditioned_clip['text'] == text)
        ]['entailment'].values

    conditioned_clip['D(mean_wv(F,R),mean_wv(F,S))'].at_index[index] = complete_df.loc[(complete_df['videopath'] == trimed_vp) & ('neg_caption' == text)]['D(mean_wv(F,R),mean_wv(F,S))']
    conditioned_clip['caption'].at_index[index] = complete_df.loc[(complete_df['videopath'] == trimed_vp) & ('neg_caption' == text)]['caption']
    conditioned_clip['unconditioned_videopath'].at_index[index] = unconditioned_clip.loc[
        (unconditioned_clip['videopath'][0:-10] == videopath[0:-10]) & (unconditioned_clip['text'] == text)
        ]['videopath'].values

conditioned_clip.rename(columns={"text": "neg_caption","videopath": "conditioned_videopath"})

conditioned_clip.to_csv('merged_for_training_paths.csv', index=False)
