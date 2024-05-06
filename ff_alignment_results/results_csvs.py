import pandas as pd

clip_flant = pd.read_csv('videocon_clip-flant5-xxl.csv')
instructblip_flant = pd.read_csv('videocon_instructblip-flant5-xxl.csv')
llava = pd.read_csv('videocon_llava-v1.5-13b.csv')


#clip_flant['videopath'] = clip_flant['videopath'].str[9:-11]
#instructblip_flant['videopath'] = instructblip_flant['videopath'].str[64:-11]
#llava['videopath'] = llava['videopath'].str[64:-11]
#
#clip_flant.to_csv('videocon_clip-flant5-xxl.csv', index=False)
#instructblip_flant.to_csv('videocon_instructblip-flant5-xxl.csv', index=False)
#llava.to_csv('videocon_llava-v1.5-13b.csv', index=False)

clip_flant = clip_flant.rename(columns={'entailment': 'clip_flant'})
clip_flant['instructblip_flant'] = -1
clip_flant['llava'] = -1

for index,row in clip_flant.iterrows():
    clip_flant['instructblip_flant'].at[index] = instructblip_flant.loc[(instructblip_flant['videopath'] == row['videopath']) & (instructblip_flant['text'] == row['text'])]['entailment'].values[0]
    clip_flant['llava'].at[index] = llava.loc[(llava['videopath'] == row['videopath']) & (llava['text'] == row['text'])]['entailment'].values[0]

clip_flant.drop_duplicates(subset=['videopath','text'], keep='first',inplace=True)

clip_flant.to_csv('ff_aligments.csv', index=False)
