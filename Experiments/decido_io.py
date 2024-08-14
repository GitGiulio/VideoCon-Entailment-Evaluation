import pandas as pd

df = pd.read_csv('ilmiocsv.csv')

input = pd.read_csv('input.csv')


# columns = ['videopath','caption','split']

# come :  ['msr-vtt/TrainValVideo/video5920.mp4','
# The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: <|video|>
# Human: Does this video entail the description: "A group of children stroke a small but calm python."?
# AI: No',
# 'train']

# in questo devo mettere tutti quelli che ci sono gia in quello originale + tutti i video sintetici condizionati

HYPERPARAMETER = 0.0

caption = ""
SI = "SI"
NO = "NO"
si_o_no = NO

for index, row in input.iterrows():
    if row['D(mean_wv(F,R),mean_wv(F,S))'] > HYPERPARAMETER:
        # prima coppia (V_S_U,T_R)
        caption = row['caption']
        si_o_no = NO
        df.append({"videopath": f"{videopath_unconditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"})

        # prima coppia (V_S_U,T_S)
        caption = row['neg_caption']
        si_o_no = SI
        df.append({"videopath": f"{videopath_unconditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"})
    else:
        # prima coppia (V_S_C,T_R)
        caption = row['caption']
        si_o_no = NO
        df.append({"videopath": f"{videopath_conditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"})

        # prima coppia (V_S_C,T_S)
        caption = row['neg_caption']
        si_o_no = SI
        df.append({"videopath": f"{videopath_conditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"})

df.to_csv('chosed_by_me_H_0.csv', index=False)