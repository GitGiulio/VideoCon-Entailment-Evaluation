import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\giuli\PycharmProjects\VideoCon-Entailment-Evaluation\plottings\PerLaTesi\with_mean.csv')


for round in [1,2,4,5]:
    df[f'D(clip_flant(Vc_{round},R),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vc_{round},R)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vu_{round},S))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vu_{round},S)']
    df[f'D(clip_flant(Vu_{round},S),clip_flant(Vc_{round},S))'] = df[f'clip_flant(Vu_{round},S)'] - df[f'clip_flant(Vc_{round},S)']
    df[f'D(clip_flant(Vu_{round},R),clip_flant(Vc_{round},R))'] = df[f'clip_flant(Vu_{round},R)'] - df[f'clip_flant(Vc_{round},R)']
    df[f'D(llava(Vc_{round},R),llava(Vc_{round},S))'] = df[f'llava(Vc_{round},R)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vu_{round},S))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vu_{round},S)']
    df[f'D(llava(Vu_{round},S),llava(Vc_{round},S))'] = df[f'llava(Vu_{round},S)'] - df[f'llava(Vc_{round},S)']
    df[f'D(llava(Vu_{round},R),llava(Vc_{round},R))'] = df[f'llava(Vu_{round},R)'] - df[f'llava(Vc_{round},R)']
    df[f'D(instructblip(Vc_{round},R),instructblip(Vc_{round},S))'] = df[f'instructblip(Vc_{round},R)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vu_{round},S))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vu_{round},S)']
    df[f'D(instructblip(Vu_{round},S),instructblip(Vc_{round},S))'] = df[f'instructblip(Vu_{round},S)'] - df[f'instructblip(Vc_{round},S)']
    df[f'D(instructblip(Vu_{round},R),instructblip(Vc_{round},R))'] = df[f'instructblip(Vu_{round},R)'] - df[f'instructblip(Vc_{round},R)']

df[f'D(mean(Vc,R),mean(Vc,S))'] = df[f'mean(Vc,R)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vu,S))'] = df[f'mean(Vu,R)'] - df[f'mean(Vu,S)']
df[f'D(mean(Vu,S),mean(Vc,S))'] = df[f'mean(Vu,S)'] - df[f'mean(Vc,S)']
df[f'D(mean(Vu,R),mean(Vc,R))'] = df[f'mean(Vu,R)'] - df[f'mean(Vc,R)']

df[f'D(llava(Vc,R),llava(Vc,S))'] = df[f'llava(Vc_mean,R)'] - df[f'llava(Vc_mean,S)']
df[f'D(llava(Vu,R),llava(Vu,S))'] = df[f'llava(Vu_mean,R)'] - df[f'llava(Vu_mean,S)']
df[f'D(llava(Vu,S),llava(Vc,S))'] = df[f'llava(Vu_mean,S)'] - df[f'llava(Vc_mean,S)']
df[f'D(llava(Vu,R),llava(Vc,R))'] = df[f'llava(Vu_mean,R)'] - df[f'llava(Vc_mean,R)']

df[f'D(instructblip(Vc,R),instructblip(Vc,S))'] = df[f'instructblip(Vc_mean,R)'] - df[f'instructblip(Vc_mean,S)']
df[f'D(instructblip(Vu,R),instructblip(Vu,S))'] = df[f'instructblip(Vu_mean,R)'] - df[f'instructblip(Vu_mean,S)']
df[f'D(instructblip(Vu,S),instructblip(Vc,S))'] = df[f'instructblip(Vu_mean,S)'] - df[f'instructblip(Vc_mean,S)']
df[f'D(instructblip(Vu,R),instructblip(Vc,R))'] = df[f'instructblip(Vu_mean,R)'] - df[f'instructblip(Vc_mean,R)']

df[f'D(clip_flant(Vc,R),clip_flant(Vc,S))'] = df[f'clip_flant(Vc_mean,R)'] - df[f'clip_flant(Vc_mean,S)']
df[f'D(clip_flant(Vu,R),clip_flant(Vu,S))'] = df[f'clip_flant(Vu_mean,R)'] - df[f'clip_flant(Vu_mean,S)']
df[f'D(clip_flant(Vu,S),clip_flant(Vc,S))'] = df[f'clip_flant(Vu_mean,S)'] - df[f'clip_flant(Vc_mean,S)']
df[f'D(clip_flant(Vu,R),clip_flant(Vc,R))'] = df[f'clip_flant(Vu_mean,R)'] - df[f'clip_flant(Vc_mean,R)']

df = df.sort_values('D(mean_wv(F,R),mean_wv(F,S))', ascending=False)
df = df.reset_index(drop=True)

c = 0
for i,r in df.iterrows():
    if r['mean_wv(F,R)'] > 0.5 and r['misalignment'] == 'flip':
        c += 1
        if c > 140:
            print(f"path: {r['videopath']}  |  T_R: {r['caption']}  |  T_S: {r['neg_caption']}")
            if c > 160:
                exit(1)