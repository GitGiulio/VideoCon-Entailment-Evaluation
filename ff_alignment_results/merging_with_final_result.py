import pandas as pd

ff_aligments = pd.read_csv('../data/ff_aligments.csv')
complete_df = pd.read_csv('../data/final_dataset.csv')

complete_df['ff_cap_llava'] = -1
complete_df['ff_cap_instructblip_flant'] = -1
complete_df['ff_cap_clip_flant'] = -1

complete_df['ff_neg_cap_llava'] = -1
complete_df['ff_neg_cap_instructblip_flant'] = -1
complete_df['ff_neg_cap_clip_flant'] = -1

for index, row in complete_df.iterrows():
    caption_temp = ff_aligments.loc[(ff_aligments['videopath'] == row['videopath']) & (ff_aligments['text'] == row['caption'])]
    neg_caption_temp = ff_aligments.loc[(ff_aligments['videopath'] == row['videopath']) & (ff_aligments['text'] == row['neg_caption'])]
    if caption_temp.shape[0] != 1:
        print('ERRORE')
        print(f'caption_temp{caption_temp}')
    if neg_caption_temp.shape[0] != 1:
        print('ERRORE')
        print(f'neg_caption_temp{neg_caption_temp}')
    complete_df['ff_cap_llava'].at[index] = caption_temp['llava']
    complete_df['ff_cap_instructblip_flant'].at[index] = caption_temp['instructblip_flant']
    complete_df['ff_cap_clip_flant'].at[index] = caption_temp['clip_flant']

    complete_df['ff_neg_cap_llava'].at[index] = neg_caption_temp['llava']
    complete_df['ff_neg_cap_instructblip_flant'].at[index] = neg_caption_temp['instructblip_flant']
    complete_df['ff_neg_cap_clip_flant'].at[index] = neg_caption_temp['clip_flant']

complete_df.to_csv('../data/complete_df.csv', index=False)