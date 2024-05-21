import os
import pandas as pd

import t2v_metrics

# clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
smaller_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')  # usa questo se crasha
# smaller_llava_score = t2v_metrics.VQAScore(model='llava-v1.5-7b') # alternativa piccola
# llava_score = t2v_metrics.VQAScore(model='llava-v1.5-13b')
# instructblip_score = t2v_metrics.VQAScore(model='instructblip-flant5-xxl')
# clip_score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336')
# blip_itm_score = t2v_metrics.ITMScore(model='blip2-itm')
# pick_score = t2v_metrics.CLIPScore(model='pickscore-v1')
# hpsv2_score = t2v_metrics.CLIPScore(model='hpsv2')
# image_reward_score = t2v_metrics.ITMScore(model='image-reward-v1')

path = 'data/videocon'

first_frames_dictionary = {}

""" I create a dictionary containig all the videos path as a key and as a value a dictionary containing:
        1) 'image' -> the path of the first frame;
        2) 'captions' -> a list of the captions;
        3) 'neg_captions' -> a list of the negated captions.
"""
for root, dirs, files in os.walk(path, topdown=False):
    for dirname in dirs:
        first_frames_dictionary[dirname] = {}
        for root2, dirs2, files2 in os.walk(os.path.join(root, dirname), topdown=False):
            for filename2 in files2:
                if filename2 == '000000.png':
                    first_frames_dictionary[dirname]['image'] = os.path.join(os.path.join(root, dirname), filename2)
                elif filename2 == 'captions.txt':
                    with open(os.path.join(os.path.join(root, dirname), filename2), 'r') as f:
                        first_frames_dictionary[dirname]['caption'] = []
                        for line in f:
                            first_frames_dictionary[dirname]['caption'].append(line)
                elif filename2 == 'neg_captions.txt':
                    with open(os.path.join(os.path.join(root, dirname), filename2), 'r') as f:
                        first_frames_dictionary[dirname]['neg_caption'] = []
                        for line in f:
                            first_frames_dictionary[dirname]['neg_caption'].append(line)
                else:
                    print('QUALCOSA DI INASPETTATO')

dataframe = pd.read_csv('data/final_dataset.csv')

dataframe['alignment(ff-caption)'] = -1
dataframe['alignment(ff-neg_cap)'] = -1
dataframe['alignment_difference'] = -1

for index, row in dataframe.iterrows():
    if first_frames_dictionary[row['videopath']] != {}:

        img_path = first_frames_dictionary[row['videopath']]['image']

        images = [img_path]
        texts = [row['caption'], row['neg_caption']]
        score = smaller_flant5_score(images=images, texts=texts)

        dataframe['alignment(ff-caption)'].at[index] = score[0][0].cpu().numpy()
        dataframe['alignment(ff-neg_cap)'].at[index] = score[0][1].cpu().numpy()
        dataframe['alignment_difference'].at[index] = score[0][0].cpu().numpy() - score[0][1].cpu().numpy()
    else:
        print('QUALCOSA DI INASPETTATO')

dataframe.to_csv('results.csv', index=False)