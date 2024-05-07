import pandas as pd


videocon_llm_entailment_path = 'data/videocon_llm_entailment.csv'  # (videopath,source,caption,neg_caption,youtube_key,split,misalignment)

videocon_llm_entailment = pd.read_csv(videocon_llm_entailment_path)

videocon_llm_entailment = videocon_llm_entailment[
    (videocon_llm_entailment["misalignment"].isin(["action", "flip"]))
    & (videocon_llm_entailment["split"] == "train")
    # I keep only the ones with misalignment type (action|flip) and from the train set
    ]

for index, row in videocon_llm_entailment.iterrows():
    videopathtemp = row['videopath']
    videocon_llm_entailment.loc[index, 'videopath'] = videopathtemp.replace('/', '_')[0:len(videopathtemp)-4]
videocon_llm_entailment.to_csv('data/my_new_dataset.csv', index=False)


""" My brutal way of reading the csv file without pandas

csv_videocon = open('videocon_llm_entailment.csv', 'r') # (videopath,source,caption,neg_caption,youtube_key,split,misalignment)

c = 0

list = []

for line in csv_videocon:
    line_split = line.split(',')

    line_split[len(line_split)-1] = line_split[len(line_split)-1].rstrip('\n')
    #print(line_split[6])
    if line_split[len(line_split)-2] == 'train' and (line_split[len(line_split)-1] == 'action' or line_split[len(line_split)-1] == 'flip'):
        c += 1
        line_split[0] = line_split[0].replace('_', '/')[:-4] # formatting the path to be equal to the video name i have form the directories names\
        list.append(line_split)
        #print(line_split[0])

"""