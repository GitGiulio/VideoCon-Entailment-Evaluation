import pandas as pd

conditioned_clip =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_conditioned_clip-flant5-xxl_sample_4_frame.csv'),
unconditioned_clip =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_unconditioned_clip-flant5-xxl_sample_4_frame.csv'),
conditioned_llava =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_conditioned_llava-v1.5-13b_sample_4_frame.csv'),
unconditioned_llava =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_unconditioned_llava-v1.5-13b_sample_4_frame.csv'),
conditioned_instructblip =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_conditioned_instructblip-flant5-xxl_sample_4_frame.csv'),
unconditioned_instructblip =  pd.read_csv('\entailment_csv_for_paths\\videocon_s001_unconditioned_instructblip-flant5-xxl_sample_4_frame.csv')

conditioned_clip.rename(columns={"entailment": "conditioned_clip"})

for index, row in conditioned_clip.iterrows():
    row['unconditioned_clip'] = unconditioned_instructblip.loc['videopath' == row['video_path']]['entailment']
    row['conditioned_llava'] = conditioned_llava.loc['videopath' == row['video_path']]['entailment']
    row['unconditioned_llava'] = unconditioned_llava.loc['videopath' == row['video_path']]['entailment']
    row['conditioned_instructblip'] = conditioned_instructblip.loc['videopath' == row['video_path']]['entailment']
    row['unconditioned_instructblip'] = unconditioned_instructblip.loc['videopath' == row['video_path']]['entailment']



conditioned_clip.to_csv('merged_entailment_scores_with_means.csv', index=False)
