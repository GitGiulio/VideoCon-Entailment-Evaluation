import os
import pandas as pd
import json

from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import openai
from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
pipeline = get_llama2_pipeline("tifa-benchmark/llama2_tifa_question_generation")
unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
vqa_model = VQAModel("mplug-large")

path = 'videocon'

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

dataframe['results(ff-caption)'] = -1
dataframe['results(ff-neg_cap)'] = -1
dataframe['alignment(ff-caption)'] = -1
dataframe['alignment(ff-neg_cap)'] = -1
dataframe['alignment_difference'] = -1

for index, row in dataframe.iterrows():
    if first_frames_dictionary[row['videopath']] != {}:
        print(f'neg_caption --------------------------')
        print(first_frames_dictionary[row['videopath']]['neg_caption'])
        print(row['neg_caption'])
        print(f'caption --------------------------')
        print(first_frames_dictionary[row['videopath']]['caption'])
        print(row['caption'])

        llama2_questions_caption = get_llama2_question_and_answers(pipeline, row['caption'])  # generating questions
        llama2_questions_neg_caption = get_llama2_question_and_answers(pipeline, row['neg_caption'])  # generating questions

        filtered_questions_caption = filter_question_and_answers(unifiedqa_model,llama2_questions_caption)  # verifing and filtering quesations
        filtered_questions_neg_caption = filter_question_and_answers(unifiedqa_model,llama2_questions_neg_caption)  # verifing and filtering quesations

        img_path = first_frames_dictionary[row['videopath']]['image']

        result_caption = tifa_score_single(vqa_model, filtered_questions_caption,img_path)  # quella che mi ha detto di calcolare LZ
        result_caption = json.loads(result_caption)

        result_neg_caption = tifa_score_single(vqa_model, filtered_questions_neg_caption,img_path)  # quella che vorrei calcolare io per fare la differenza
        result_neg_caption = json.loads(result_neg_caption)

        alignment_difference = result_caption['TIFA score'] - result_neg_caption['TIFA score']

        dataframe['results(ff-caption)'] = result_caption
        dataframe['results(ff-neg_cap)'] = result_neg_caption
        dataframe['alignment(ff-caption)'].at[index] = result_caption['TIFA score']
        dataframe['alignment(ff-neg_cap)'].at[index] = result_neg_caption['TIFA score']
        dataframe['alignment_difference'].at[index] = alignment_difference
    else:
        print('QUALCOSA DI INASPETTATO')

dataframe.to_csv('results.csv', index=False)