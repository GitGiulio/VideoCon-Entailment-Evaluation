import pandas as pd

ff_aligments = pd.read_csv('ff_aligments.csv')
final_df = pd.read_csv('../final_dataset.csv')

final_df['ff_cap_llava'] = -1
final_df['ff_cap_instructblip_flant'] = -1
final_df['ff_cap_clip_flant'] = -1

final_df['ff_neg_cap_llava'] = -1
final_df['ff_neg_cap_instructblip_flant'] = -1
final_df['ff_neg_cap_clip_flant'] = -1

for index, row in final_df.iterrows():
    caption_temp = ff_aligments.loc[(ff_aligments['videopath'] == row['videopath']) & (ff_aligments['text'] == row['caption'])]
    neg_caption_temp = ff_aligments.loc[(ff_aligments['videopath'] == row['videopath']) & (ff_aligments['text'] == row['neg_caption'])]
    if caption_temp.shape[0] != 1:
        print('ERRORE')
        print(f'caption_temp{caption_temp}')
    if neg_caption_temp.shape[0] != 1:
        print('ERRORE')
        print(f'neg_caption_temp{neg_caption_temp}')
    final_df['ff_cap_llava'].at[index] = caption_temp['llava']
    final_df['ff_cap_instructblip_flant'].at[index] = caption_temp['instructblip_flant']
    final_df['ff_cap_clip_flant'].at[index] = caption_temp['clip_flant']

    final_df['ff_neg_cap_llava'].at[index] = neg_caption_temp['llava']
    final_df['ff_neg_cap_instructblip_flant'].at[index] = neg_caption_temp['instructblip_flant']
    final_df['ff_neg_cap_clip_flant'].at[index] = neg_caption_temp['clip_flant']

final_df.to_csv('complete_df.csv',index=False)