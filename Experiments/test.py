import pandas as pd

a = pd.read_csv('training_csvs/clip_H_0.csv')
b = pd.read_csv('training_csvs/llava_H_0.csv')
c = pd.read_csv('training_csvs/instructblip_H_0.csv')
d = pd.read_csv('training_csvs/chosed_by_me_H_0.csv')
e = pd.read_csv('training_csvs/all_unconditioned.csv')
f = pd.read_csv('train_llm_mix_entail_feedback.csv')

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)

cond = 0
unc = 0
n_m = 0
for index,row in a.iterrows():
    #print(row.videopath[-8:-5])
    if row.videopath[-8:-5] == "_1_":
        cond += 1
    elif row.videopath[-8:-5] == '_0_':
        unc += 1
    else:
        n_m += 1
print(f"A cond = {cond}")
print(f"unc = {unc}")
print(f"n_m = {n_m}")
print(cond / (cond + unc))

cond = 0
unc = 0
n_m = 0
for index, row in b.iterrows():
    # print(row.videopath[-8:-5])
    if row.videopath[-8:-5] == "_1_":
        cond += 1
    elif row.videopath[-8:-5] == '_0_':
        unc += 1
    else:
        n_m += 1
print(f"B cond = {cond}")
print(f"unc = {unc}")
print(f"n_m = {n_m}")
print(cond / (cond + unc))

cond = 0
unc = 0
n_m = 0
for index, row in c.iterrows():
    # print(row.videopath[-8:-5])
    if row.videopath[-8:-5] == "_1_":
        cond += 1
    elif row.videopath[-8:-5] == '_0_':
        unc += 1
    else:
        n_m += 1
print(f"C cond = {cond}")
print(f"unc = {unc}")
print(f"n_m = {n_m}")
print(cond / (cond + unc))

cond = 0
unc = 0
n_m = 0
for index, row in d.iterrows():
    # print(row.videopath[-8:-5])
    if row.videopath[-8:-5] == "_1_":
        cond += 1
    elif row.videopath[-8:-5] == '_0_':
        unc += 1
    else:
        n_m += 1
print(f"D cond = {cond}")
print(f"unc = {unc}")
print(f"n_m = {n_m}")
print(cond / (cond + unc))

cond = 0
unc = 0
n_m = 0
for index, row in e.iterrows():
    # print(row.videopath[-8:-5])
    if row.videopath[-8:-5] == "_1_":
        cond += 1
    elif row.videopath[-8:-5] == '_0_':
        unc += 1
    else:
        n_m += 1
print(f"E cond = {cond}")
print(f"unc = {unc}")
print(f"n_m = {n_m}")
print(cond / (cond + unc))


