import pandas as pd

conditioned_clip = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_clip-flant5-xxl_sample_4_frame.csv')
conditioned_llava = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_llava-v1.5-13b_sample_4_frame.csv')
conditioned_instructblip = pd.read_csv('entailment_csv_for_paths/videocon_s001_conditioned_instructblip-flant5-xxl_sample_4_frame.csv')

unconditioned_clip = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_clip-flant5-xxl_sample_4_frame.csv')
unconditioned_instructblip = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv')
unconditioned_llava = pd.read_csv('entailment_csv_for_paths/videocon_s001_unconditioned_llava-v1.5-13b_sample_4_frame.csv')

complete_df = pd.read_csv('../data/complete_df.csv')

conditioned_clip.rename(columns={"entailment": "conditioned_clip"})

unconditioned_clip['trimed_videopath'] = 0
unconditioned_instructblip['trimed_videopath'] = 0
unconditioned_llava['trimed_videopath'] = 0
for index,row in unconditioned_clip.iterrows():
    unconditioned_clip['trimed_videopath'].at[index] = row.videopath[73:-10]
for index,row in unconditioned_instructblip.iterrows():
    unconditioned_instructblip['trimed_videopath'].at[index] = row.videopath[73:-10]
for index,row in unconditioned_llava.iterrows():
    unconditioned_llava['trimed_videopath'].at[index] =row.videopath[73:-10]

conditioned_clip['conditioned_llava'] = -2
conditioned_clip['conditioned_instructblip'] = -2
conditioned_clip['unconditioned_clip'] = -2
conditioned_clip['unconditioned_llava'] = -2
conditioned_clip['unconditioned_clip'] = -2
conditioned_clip['D(mean_wv(F,R),mean_wv(F,S))'] = -2
conditioned_clip['caption'] = -2
conditioned_clip['unconditioned_videopath'] = -2

for index, row in conditioned_clip.iterrows():

    videopath = row.videopath
    text = row.text

    trimed_vp = videopath[73:-10]

    if len(conditioned_llava.loc[
        (conditioned_llava['videopath'] == videopath) & (conditioned_llava['text'] == text)
        ]['entailment'].values) == 0:
        print(videopath)
        print(text)
        continue
    conditioned_clip['conditioned_llava'].at[index] = conditioned_llava.loc[
        (conditioned_llava['videopath'] == videopath) & (conditioned_llava['text'] == text)
        ]['entailment'].values[0]
    if len(conditioned_instructblip.loc[
        (conditioned_instructblip['videopath'] == videopath) & (conditioned_instructblip['text'] == text)
        ]['entailment'].values) == 0:
        print(videopath)
        print(text)
        continue
    conditioned_clip['conditioned_instructblip'].at[index] = conditioned_instructblip.loc[
        (conditioned_instructblip['videopath'] == videopath) & (conditioned_instructblip['text'] == text)
        ]['entailment'].values[0]


    if len(unconditioned_instructblip.loc[
        (unconditioned_instructblip['trimed_videopath'] == trimed_vp) & (unconditioned_instructblip['text'] == text)
        ]['entailment'].values) == 0:
        print(videopath)
        print(text)
        continue
    conditioned_clip['unconditioned_clip'].at[index] = unconditioned_instructblip.loc[
        (unconditioned_instructblip['trimed_videopath'] == trimed_vp) & (unconditioned_instructblip['text'] == text)
        ]['entailment'].values[0]
    if len(unconditioned_llava.loc[
        (unconditioned_llava['trimed_videopath'] == trimed_vp) & (unconditioned_llava['text'] == text)
        ]['entailment'].values) == 0:
        print(videopath)
        print(text)
        continue
    conditioned_clip['unconditioned_llava'].at[index] = unconditioned_llava.loc[
        (unconditioned_llava['trimed_videopath'] == trimed_vp) & (unconditioned_llava['text'] == text)
        ]['entailment'].values[0]
    if len(unconditioned_clip.loc[
        (unconditioned_clip['trimed_videopath'] == trimed_vp) & (unconditioned_clip['text'] == text)
        ]['entailment'].values) == 0:
        print(videopath)
        print(text)
        continue
    conditioned_clip['unconditioned_clip'].at[index] = unconditioned_clip.loc[
        (unconditioned_clip['trimed_videopath'] == trimed_vp) & (unconditioned_clip['text'] == text)
        ]['entailment'].values[0]


    if len(complete_df.loc[(complete_df['videopath'] == trimed_vp) & (complete_df['neg_caption'] == text)]['D(mean_wv(F,R),mean_wv(F,S))'].values) != 1:
        #print(complete_df.loc[(complete_df['videopath'] == trimed_vp) & (complete_df['neg_caption'] == text)]['D(mean_wv(F,R),mean_wv(F,S))'].values)

        continue
    #print("NON SKIPPATOOOOOOOO________________________________________")
    #print(complete_df.loc[(complete_df['videopath'] == trimed_vp) & (complete_df['neg_caption'] == text)]['D(mean_wv(F,R),mean_wv(F,S))'].values)
    conditioned_clip['D(mean_wv(F,R),mean_wv(F,S))'].at[index] = complete_df.loc[(complete_df['videopath'] == trimed_vp) & (complete_df['neg_caption'] == text)]['D(mean_wv(F,R),mean_wv(F,S))']
    conditioned_clip['caption'].at[index] = complete_df.loc[(complete_df['videopath'] == trimed_vp) & (complete_df['neg_caption'] == text)]['caption']
    conditioned_clip['unconditioned_videopath'].at[index] = unconditioned_clip.loc[
        (unconditioned_clip['trimed_videopath'] == trimed_vp) & (unconditioned_clip['text'] == text)
        ]['videopath'].values

conditioned_clip.rename(columns={"text": "neg_caption","videopath": "conditioned_videopath"})

conditioned_clip.to_csv('merged_for_training_paths.csv', index=False)
