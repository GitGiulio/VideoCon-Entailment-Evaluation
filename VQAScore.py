import os
import pandas as pd

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

import t2v_metrics

clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')
llava_score = t2v_metrics.VQAScore(model='llava-v1.5-13b')
instructblip_score = t2v_metrics.VQAScore(model='instructblip-flant5-xxl')

for index, row in dataframe.iterrows():
    if first_frames_dictionary[row['videopath']] != {}:

        img_path = first_frames_dictionary[row['videopath']]['image']

        images = [img_path] #
        texts = [row['caption'], row['contrasted_caption']]

        clip_results = clip_flant5_score(images=images, texts=texts)
        llava_results = llava_score(images=images, texts=texts)
        instructblip_results = instructblip_score(images=images, texts=texts)

        # I save the values obtained by the model in the pandas dataframe
        dataframe['clip_flant(F,Tr)'].at[index] = clip_results[0][0].cpu().numpy()
        dataframe['clip_flant(F,Ts)'].at[index] = clip_results[0][1].cpu().numpy()
        dataframe['clip_flant_difference'].at[index] = clip_results[0][0].cpu().numpy() - clip_results[0][1].cpu().numpy()

        dataframe['llava(F,Tr)'].at[index] = llava_results[0][0].cpu().numpy()
        dataframe['llava(F,Ts)'].at[index] = llava_results[0][1].cpu().numpy()
        dataframe['llava_difference'].at[index] = llava_results[0][0].cpu().numpy() - llava_results[0][1].cpu().numpy()

        dataframe['instructblip(F,Tr)'].at[index] = instructblip_results[0][0].cpu().numpy()
        dataframe['instructblip(F,Ts)'].at[index] = instructblip_results[0][1].cpu().numpy()
        dataframe['instructblip_difference'].at[index] = instructblip_results[0][0].cpu().numpy() - instructblip_results[0][1].cpu().numpy()

    else:
        print('EXCEPTION')

dataframe.to_csv('results.csv', index=False)