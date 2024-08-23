import pandas as pd

df = pd.read_csv('train_llm_mix_entail_feedback.csv')

input = pd.read_csv('merged_for_training_paths.csv')


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
SI = "Yes"
NO = "No"
si_o_no = NO

for index, row in input.iterrows():
    if row['D(instructblip(F,R),instructblip(F,S))'] == -2:
        continue
    if row['D(instructblip(F,R),instructblip(F,S))'] > HYPERPARAMETER:
        # prima coppia (V_S_U,T_R)
        caption = row['caption']
        si_o_no = NO
        videopath_unconditioned = row['unconditioned_videopath']
        df = pd.concat([df, pd.DataFrame([{"videopath": f"{videopath_unconditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"}])], ignore_index=True)

        # prima coppia (V_S_U,T_S)
        caption = row['text']
        si_o_no = SI
        df = pd.concat([df, pd.DataFrame([{"videopath": f"{videopath_unconditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"}])], ignore_index=True)
    else:
        # prima coppia (V_S_C,T_R)
        caption = row['caption']
        videopath_conditioned = row['videopath']
        si_o_no = NO
        df = pd.concat([df, pd.DataFrame([{"videopath": f"{videopath_conditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"}])], ignore_index=True)

        # prima coppia (V_S_C,T_S)
        caption = row['text']
        si_o_no = SI
        df = pd.concat([df, pd.DataFrame([{"videopath": f"{videopath_conditioned}",
                   "caption": f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video entail the description: \"{caption}\"?\nAI: {si_o_no}",
                   "split": "train"}])], ignore_index=True)

df.to_csv('training_csvs/instructblip_H_0.csv', index=False)