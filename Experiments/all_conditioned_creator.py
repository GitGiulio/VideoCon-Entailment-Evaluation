import pandas as pd

df = pd.read_csv('ilmiocsv.csv')


# columns = ['videopath','caption','split']

# come :  ['msr-vtt/TrainValVideo/video5920.mp4','
# The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: <|video|>
# Human: What is the misalignment between this video and the description: "a man offers to explain and demonstrate how he made it and talks about his new bicycle cart that he built"?
# AI: First, a man built his new bicycle cart, then he talks about it and offers to explain and demonstrate how he made it, not offers to explain and demonstrate how he made it before building his new bicycle cart',
# 'train']

# in questo devo mettere tutti quelli che ci sono gia in quello originale + tutti i video sintetici condizionati

for index, row in df.iterrows():
    pos = row['videopath'].find('highres_')
    videopath_temp = row['videopath']
    round.loc[index,'videopath'] = videopath_temp[pos+8:len(videopath_temp)-10]
    if set == 'videocon_synth' or set == 'videocon_real':
        conversation_temp = row['text']
        round.loc[index, 'text'] = conversation_temp[222:len(conversation_temp)-7]
round.to_csv(csv_path[type][set][f'r{i}'], index=False)



df.to_csv('all_conditioned_creator.csv', index=False)