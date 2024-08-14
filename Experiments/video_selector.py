import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.special import softmax


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_files_videocon", nargs="+")
    parser.add_argument("--csv_files_vqascore", nargs="+")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def extract_description(conversation):
    pattern = r'Does this video entail the description: "(.*?)"\?'
    match = re.search(pattern, conversation)
    if match:
        return match.group(1)
    else:
        raise ValueError("Description not found")


def load_videocon_scores(csv_files):
    df_synth_synth_list = []
    df_synth_real_list = []

    for csv_file in csv_files:
        print(f"Processing {csv_file}")
        df = pd.read_csv(csv_file, header=None, names=["videopath", "text", "entailment"])
        df["model"] = "owl-con"
        df["text"] = df["text"].apply(extract_description)
        df_synth_synth_list.append(df)

        df_real_file = csv_file.replace("synth-synth", "synth-real")
        df_real = pd.read_csv(df_real_file, header=None, names=["videopath", "text", "entailment"])
        df_real["model"] = "owl-con"
        df_real["text"] = df_real["text"].apply(extract_description)
        df_synth_real_list.append(df_real)

    return df_synth_synth_list, df_synth_real_list


def load_train_data_from_videocon_csv(csv_files):
    df_train_list = []

    for csv_file in csv_files:
        df_name = "_".join(Path(csv_file).stem.split("_")[2:4])
        df_train_file = f"data/train_videocon_{df_name}.csv"
        df_train = pd.read_csv(df_train_file)
        df_train["text"] = df_train["caption"].apply(extract_description)
        df_train["original_index"] = df_train.index
        df_train_list.append(df_train)

    return df_train_list


def load_vqascore_scores(csv_files):
    df_synth_synth_list = []
    df_synth_real_list = []

    for csv_file in csv_files:
        print(f"Processing {csv_file}")
        df = pd.read_csv(csv_file)
        df["model"] = Path(csv_file).stem.split("_")[3]
        df_synth_synth = df.iloc[::2]
        df_synth_real = df.iloc[1::2]
        df_synth_synth_list.append(df_synth_synth)
        df_synth_real_list.append(df_synth_real)

    return df_synth_synth_list, df_synth_real_list


def save_config(df_name, data_dir="data"):
    # with open(f"libs/videocon/training/configs/video_{df_name}.yaml", "w") as f:
    with open(f"{data_dir}/video_{df_name}.yaml", "w") as f:
        yaml.dump(
            {
                "data_files": [
                    f"{data_dir}/{df_name}.csv",
                ],
                "pretrained_ckpt": "/leonardo_scratch/fast/IscrC_UTUVLM/checkpoints/MAGAer13/mplug-owl-llama-7b-video",
            },
            f,
            default_flow_style=False,
        )


def merge(file3, data_dir="data"):
    ## merge train
    file1 = 'data/train_llm_entailment.csv'
    file2 = 'data/train_llm_feedback.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    df2['videopath'] = df2['videopath'].apply(lambda x: x.replace('/leonardo_work/IscrC_UTUVLM/', '/leonardo_scratch/fast/IscrB_FANDANGO/'))

    df1['entailment'] = 1.0
    df1['delta'] = 1.0
    df2['entailment'] = 1.0
    df2['delta'] = 1.0
    df1['domain'] = 'real'
    df2['domain'] = 'real'
    df3['domain'] = 'synthetic'

    assert set(df1.columns) == set(df2.columns) == set(df3.columns)

    print(df1.columns, df2.columns, df3.columns)
    df = pd.concat([df1, df2, df3])
    df = df.sample(frac = 1)
    print(len(df))

    filename = os.path.basename(file3)  # 'train_videocon_example.video.mp4'
    name_without_extension = os.path.splitext(filename)[0]  # 'train_videocon_example.video'
    df3_name = name_without_extension.removeprefix('train_videocon_')  # 'example.video'
    df_name = f'train_llm_mix_entail_feedback_{df3_name}'
    df.to_csv(f'{data_dir}/{df_name}.csv', index = False)

    save_config(df_name, data_dir)


def save_top_ranked(df, df_name, data_dir="data2"):
    print(f"{data_dir}/train_videocon_{df_name}.csv")
    # print(f"Saving libs/videocon/training/configs/video_{df_name}.yaml")
    df = df.sort_values(by="original_index").drop(columns=["original_index"])
    df.to_csv(f"{data_dir}/train_videocon_{df_name}.csv", index=False)
    # with open(f"libs/videocon/training/configs/video_{df_name}.yaml", "w") as f:
    #     yaml.dump(
    #         {
    #             "data_files": [
    #                 "data/train_llm_mix_entail_feedback_FAST.csv",
    #                 f"data/train_videocon_{df_name}.csv",
    #                 "data/val_llm_entailment_FAST.csv",
    #                 "data/val_llm_entailment_action_FAST.csv",
    #                 "data/val_llm_entailment_flip_FAST.csv",
    #             ],
    #             "pretrained_ckpt": "/leonardo_scratch/fast/IscrC_UTUVLM/checkpoints/MAGAer13/mplug-owl-llama-7b-video",
    #         },
    #         f,
    #         default_flow_style=False,
    #     )
    
    merge(f"{data_dir}/train_videocon_{df_name}.csv", data_dir)


def select_top_ranked_from_df(
    df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]
):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    idx = df_synth_synth.groupby("video_name")["entailment"].idxmax()
    df_filtered = df_synth_synth.loc[idx].reset_index(drop=True)

    df_filtered = df_filtered.drop(columns=["video_name", "model"])
    # df_filtered["videopath"] = df_filtered["videopath"].apply(lambda x: Path(x).stem)

    if df_synth_real is not None:
        df_synth_real = df_synth_real.drop(columns=["video_name", "model"])
        df_synth_real["videopath"] = df_synth_real["videopath"].apply(lambda x: Path(x).stem)
        df_filtered = pd.concat(
            [
                df_filtered,
                df_synth_real[df_synth_real["videopath"].isin(df_filtered["videopath"].values)],
            ]
        )
        df_filtered = df_filtered.reset_index(drop=True)

    if df_synth_real is None:
        merge_columns = ["videopath"]
    else:
        merge_columns = ["videopath", "text"]
    df_filtered = pd.merge(df_filtered, df_train, on=merge_columns, how="inner")
    df_filtered = df_filtered.drop(columns=merge_columns).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered = df_filtered.rename(columns={"text": "caption"})
    df_filtered = df_filtered[
        ["videopath"] + [col for col in df_filtered.columns if col != "videopath"]
    ]
    df_filtered = df_filtered.sort_values(by="original_index")

    for threshold in thresholds:
        if df_synth_real is not None:
            df_filtered_threshold = df_filtered[
                ((df_filtered["label"] == 1) & (df_filtered["entailment"] >= threshold))
                | ((df_filtered["label"] == 0) & (df_filtered["entailment"] < threshold))
            ]
            if not df_filtered_threshold.empty:
                df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
                save_top_ranked(df_filtered_threshold, f"{dataset_name}_threshold_{threshold}")

        filtered_videopaths = df_filtered[
            (df_filtered["label"] == 1) & (df_filtered["entailment"] >= threshold)
        ]["videopath"].values
        df_filtered_threshold = df_filtered[df_filtered["videopath"].isin(filtered_videopaths)]
        if not df_filtered_threshold.empty:
            df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
            save_top_ranked(
                df_filtered_threshold, f"{dataset_name}_threshold_{threshold}_positive"
            )

    save_top_ranked(df_filtered, dataset_name)


def select_random(df_synth_synth, df_train, dataset_name, n=3):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    df_synth_synth = df_synth_synth.groupby("video_name")
    for i in range(n):
        random.seed(i)
        np.random.seed(i)
        df_sample = df_synth_synth.apply(lambda x: x.sample(n=1)).reset_index(drop=True)
        df_sample = df_sample.drop(columns=["text", "video_name"])
        df_sample = pd.merge(df_sample, df_train, on="videopath", how="inner")
        df_sample = df_sample.drop(columns=["videopath", "text", "entailment", "model"]).rename(
            columns={"original_videopath": "videopath"}
        )
        df_sample = df_sample.rename(columns={"text": "caption"})
        df_sample = df_sample[
            ["videopath"] + [col for col in df_sample.columns if col != "videopath"]
        ]
        df_sample = df_sample.sort_values(by="original_index")
        save_top_ranked(df_sample, f"{dataset_name}_random_seed_{i}")


def select_top_ranked_softmax(
    df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]
):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    df_synth_synth["entailment_mean"] = df_synth_synth.groupby("videopath")[
        "entailment"
    ].transform("mean")
    df_synth_synth["entailment_probs"] = df_synth_synth.groupby(["video_name", "model"])[
        "entailment"
    ].transform(lambda x: softmax(x))
    df_synth_synth["mean_entailment_probs"] = df_synth_synth.groupby("videopath")[
        "entailment_probs"
    ].transform(lambda x: np.mean(x))
    max_indices = df_synth_synth.groupby("video_name")["mean_entailment_probs"].idxmax()
    result_df = df_synth_synth.loc[max_indices].reset_index(drop=True)

    if df_synth_real is not None:
        df_synth_real["entailment_mean"] = df_synth_real.groupby("videopath")[
            "entailment"
        ].transform("mean")
        df_synth_real = df_synth_real.drop_duplicates(subset=["videopath"])
        result_df = pd.concat(
            [
                result_df,
                df_synth_real[df_synth_real["videopath"].isin(result_df["videopath"].values)],
            ]
        )
        result_df = result_df.reset_index(drop=True)

    result_df = result_df.drop(
        columns=["entailment", "video_name", "model", "entailment_probs", "mean_entailment_probs"]
    )
    if df_synth_real is None:
        merge_columns = ["videopath"]
    else:
        merge_columns = ["videopath", "text"]
    result_df = pd.merge(result_df, df_train, on=merge_columns, how="inner")
    result_df = result_df.drop(columns=merge_columns).rename(
        columns={"original_videopath": "videopath"}
    )
    result_df = result_df.rename(columns={"text": "caption"})
    result_df = result_df.rename(columns={"entailment_mean": "entailment"})
    result_df = result_df[["videopath"] + [col for col in result_df.columns if col != "videopath"]]
    result_df = result_df.sort_values(by="original_index")

    for threshold in thresholds:
        if df_synth_real is not None:
            df_filtered_threshold = result_df[
                ((result_df["label"] == 1) & (result_df["entailment"] >= threshold))
                | ((result_df["label"] == 0) & (result_df["entailment"] < threshold))
            ]
            if not df_filtered_threshold.empty:
                df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
                save_top_ranked(df_filtered_threshold, f"{dataset_name}_threshold_{threshold}")

        filtered_videopaths = result_df[
            (result_df["label"] == 1) & (result_df["entailment"] >= threshold)
        ]["videopath"].values
        df_filtered_threshold = result_df[result_df["videopath"].isin(filtered_videopaths)]
        if not df_filtered_threshold.empty:
            df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
            save_top_ranked(
                df_filtered_threshold, f"{dataset_name}_threshold_{threshold}_positive"
            )

    save_top_ranked(result_df, f"{dataset_name}")


def select_top_ranked_from_df_hybrid(#NON USATA
    df_synth_synth,
    df_synth_real,
    df_synth_synth_owl_con,
    df_synth_real_owl_con,
    df_train,
    dataset_name,
    thresholds=[0.5],
):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    idx = df_synth_synth.groupby("video_name")["entailment"].idxmax()
    df_filtered = df_synth_synth.loc[idx].reset_index(drop=True)

    df_filtered = df_filtered.drop(columns="video_name")
    df_synth_synth_owl_con = df_synth_synth_owl_con.drop(columns=["video_name", "model"])
    df_filtered = pd.merge(
        df_filtered, df_synth_synth_owl_con, on=["videopath", "text"], how="inner"
    )
    df_filtered = df_filtered.drop_duplicates(subset=["videopath"])
    df_filtered["entailment"] = df_filtered["entailment_y"]
    df_filtered = df_filtered.drop(columns=["entailment_x", "entailment_y"])
    df_filtered = df_filtered.reset_index(drop=True)

    if df_synth_real is not None:
        df_synth_real = df_synth_real.drop(columns="video_name")
        df_synth_real = df_synth_real[
            df_synth_real["videopath"].isin(df_filtered["videopath"].values)
        ]
        df_synth_real = df_synth_real.reset_index(drop=True)
        df_synth_real_owl_con = df_synth_real_owl_con.drop(columns=["video_name", "model"])
        df_synth_real = pd.merge(
            df_synth_real, df_synth_real_owl_con, on=["videopath", "text"], how="inner"
        )
        df_synth_real = df_synth_real.drop_duplicates(subset=["videopath"])
        df_synth_real["entailment"] = df_synth_real["entailment_y"]
        df_synth_real = df_synth_real.drop(columns=["entailment_x", "entailment_y"])
        df_synth_real = df_synth_real.reset_index(drop=True)
        df_filtered = pd.concat([df_filtered, df_synth_real])
        df_filtered = df_filtered.reset_index(drop=True)

    if df_synth_real is None:
        merge_columns = ["videopath"]
    else:
        merge_columns = ["videopath", "text"]

    df_filtered = pd.merge(df_filtered, df_train, on=merge_columns, how="inner")
    df_filtered = df_filtered.drop(columns=merge_columns).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered = df_filtered.rename(columns={"text": "caption"})
    df_filtered = df_filtered[
        ["videopath"] + [col for col in df_filtered.columns if col != "videopath"]
    ]
    df_filtered = df_filtered.sort_values(by="original_index")

    for threshold in thresholds:
        if df_synth_real is not None:
            df_filtered_threshold = df_filtered[
                ((df_filtered["label"] == 1) & (df_filtered["entailment"] >= threshold))
                | ((df_filtered["label"] == 0) & (df_filtered["entailment"] < threshold))
            ]
            if not df_filtered_threshold.empty:
                df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
                save_top_ranked(df_filtered_threshold, f"{dataset_name}_threshold_{threshold}")

        filtered_videopaths = df_filtered[
            (df_filtered["label"] == 1) & (df_filtered["entailment"] >= threshold)
        ]["videopath"].values
        df_filtered_threshold = df_filtered[df_filtered["videopath"].isin(filtered_videopaths)]
        if not df_filtered_threshold.empty:
            df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
            save_top_ranked(
                df_filtered_threshold, f"{dataset_name}_threshold_{threshold}_positive"
            )

    save_top_ranked(df_filtered, dataset_name)


def select_top_ranked_delta(#NON USATA
    df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]
):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    # drop duplicates in videopath column
    # df_synth_synth = df_synth_synth.drop_duplicates(subset=["videopath"])
    # df_synth_real = df_synth_real.drop_duplicates(subset=["videopath"])
    # df_synth_synth['delta'] = df_synth_synth.apply(lambda row: row['entailment'] - df_synth_real[df_synth_real['videopath'] == row['videopath']]['entailment'].values[0], axis=1)
    merged_df = pd.merge(
        df_synth_synth,
        df_synth_real[["videopath", "entailment"]],
        on="videopath",
        suffixes=("_synth", "_real"),
    )
    merged_df["delta"] = merged_df["entailment_synth"] - merged_df["entailment_real"]
    merged_df.drop("entailment_real", axis=1, inplace=True)
    merged_df.rename(columns={"entailment_synth": "entailment"}, inplace=True)

    merged_df_by_entailment = merged_df.copy(deep=True)
    merged_df_by_entailment["delta"] = (
        merged_df_by_entailment["delta"] * merged_df_by_entailment["entailment"]
    )
    idx = merged_df_by_entailment.groupby("video_name")["delta"].idxmax()
    df_filtered_by_entailment = merged_df_by_entailment.loc[idx].reset_index(drop=True)
    df_filtered_by_entailment = df_filtered_by_entailment.drop(columns=["video_name", "model"])

    idx = merged_df.groupby("video_name")["delta"].idxmax()
    df_filtered = merged_df.loc[idx].reset_index(drop=True)
    df_filtered = df_filtered.drop(columns=["video_name", "model"])
    videos_with_positive_delta = df_filtered[df_filtered["delta"] > 0]["videopath"].values

    # videopaths_split_0 = df_filtered[df_filtered["videopath"].str.split('_').str[-2] == '0']
    # videopaths_split_1 = df_filtered[df_filtered["videopath"].str.split('_').str[-2] == '1']

    # # Plot histograms
    # plt.figure(figsize=(10, 6))

    # plt.subplot(1, 2, 1)
    # plt.hist(videopaths_split_0["delta"], bins=20, color='blue', alpha=0.7)
    # plt.title("Unconditional")
    # plt.xlabel("Delta")
    # plt.ylabel("Frequency")

    # plt.subplot(1, 2, 2)
    # plt.hist(videopaths_split_1["delta"], bins=20, color='red', alpha=0.7)
    # plt.title("Conditional")
    # plt.xlabel("Delta")
    # plt.ylabel("Frequency")

    # # put the title
    # plt.suptitle(f"{dataset_name.split('_')[2]} synthetic-real entailment")
    # plt.tight_layout()
    # plt.savefig(f"plots/{dataset_name}_delta_histograms.png")

    if df_synth_real is not None:
        df_synth_real = df_synth_real.drop(columns=["video_name", "model"])
        df_synth_real["videopath"] = df_synth_real["videopath"].apply(lambda x: Path(x).stem)

        df_filtered_by_entailment = pd.concat(
            [
                df_filtered_by_entailment,
                df_synth_real[
                    df_synth_real["videopath"].isin(df_filtered_by_entailment["videopath"].values)
                ],
            ]
        )
        df_filtered_by_entailment = df_filtered_by_entailment.reset_index(drop=True)

        df_filtered = pd.concat(
            [
                df_filtered,
                df_synth_real[df_synth_real["videopath"].isin(df_filtered["videopath"].values)],
            ]
        )
        df_filtered = df_filtered.reset_index(drop=True)

    df_filtered_by_entailment = pd.merge(
        df_filtered_by_entailment, df_train, on=["videopath", "text"], how="inner"
    )
    df_filtered = pd.merge(df_filtered, df_train, on=["videopath", "text"], how="inner")

    df_filtered_positive_delta = df_filtered[
        df_filtered["videopath"].isin(videos_with_positive_delta)
    ]
    df_filtered_positive_delta = df_filtered_positive_delta.reset_index(drop=True)
    df_filtered_positive_delta = df_filtered_positive_delta.drop(
        columns=["videopath", "text"]
    ).rename(columns={"original_videopath": "videopath"})
    df_filtered_positive_delta = df_filtered_positive_delta.rename(columns={"text": "caption"})
    df_filtered_positive_delta = df_filtered_positive_delta[
        ["videopath"] + [col for col in df_filtered_positive_delta.columns if col != "videopath"]
    ]
    df_filtered_positive_delta = df_filtered_positive_delta.sort_values(by="original_index")
    save_top_ranked(df_filtered_positive_delta, f"{dataset_name}_positive")

    df_filtered = df_filtered.drop(columns=["videopath", "text"]).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered = df_filtered.rename(columns={"text": "caption"})
    df_filtered = df_filtered[
        ["videopath"] + [col for col in df_filtered.columns if col != "videopath"]
    ]
    df_filtered = df_filtered.sort_values(by="original_index")

    save_top_ranked(df_filtered, dataset_name)

    df_filtered_by_entailment = df_filtered_by_entailment.drop(
        columns=["videopath", "text"]
    ).rename(columns={"original_videopath": "videopath"})
    df_filtered_by_entailment = df_filtered_by_entailment.rename(columns={"text": "caption"})
    df_filtered_by_entailment = df_filtered_by_entailment[
        ["videopath"] + [col for col in df_filtered_by_entailment.columns if col != "videopath"]
    ]
    df_filtered_by_entailment = df_filtered_by_entailment.sort_values(by="original_index")

    save_top_ranked(df_filtered_by_entailment, f"{dataset_name}-by-entailment")


def select_top_ranked_delta_only_synth( #NON USATA
    df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]
):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)
    df_train.drop(columns=["text"], inplace=True)

    merged_df = pd.merge(
        df_synth_synth,
        df_synth_real[["videopath", "entailment"]],
        on="videopath",
        suffixes=("_synth", "_real"),
    )
    merged_df["delta"] = merged_df["entailment_synth"] - merged_df["entailment_real"]
    merged_df.drop("entailment_real", axis=1, inplace=True)
    merged_df.rename(columns={"entailment_synth": "entailment"}, inplace=True)

    merged_df_by_entailment = merged_df.copy(deep=True)
    merged_df_by_entailment["delta"] = (
        merged_df_by_entailment["delta"] * merged_df_by_entailment["entailment"]
    )
    idx = merged_df_by_entailment.groupby("video_name")["delta"].idxmax()
    df_filtered_by_entailment = merged_df_by_entailment.loc[idx].reset_index(drop=True)
    df_filtered_by_entailment = df_filtered_by_entailment.drop(
        columns=["video_name", "model", "text"]
    )

    idx = merged_df.groupby("video_name")["delta"].idxmax()
    df_filtered = merged_df.loc[idx].reset_index(drop=True)
    df_filtered = df_filtered.drop(columns=["video_name", "model", "text"])
    videos_with_positive_delta = df_filtered[df_filtered["delta"] > 0]["videopath"].values

    df_filtered_by_entailment = pd.merge(
        df_filtered_by_entailment, df_train, on="videopath", how="inner"
    )
    df_filtered = pd.merge(df_filtered, df_train, on=["videopath"], how="inner")
    df_filtered = df_filtered.reset_index(drop=True)

    df_filtered_positive_delta = df_filtered[
        df_filtered["videopath"].isin(videos_with_positive_delta)
    ]
    df_filtered_positive_delta = df_filtered_positive_delta.reset_index(drop=True)
    df_filtered_positive_delta = df_filtered_positive_delta.drop(columns=["videopath"]).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered_positive_delta = df_filtered_positive_delta[
        ["videopath"] + [col for col in df_filtered_positive_delta.columns if col != "videopath"]
    ]
    df_filtered_positive_delta = df_filtered_positive_delta.sort_values(by="original_index")
    save_top_ranked(df_filtered_positive_delta, f"{dataset_name}_positive")

    df_filtered = df_filtered.drop(columns=["videopath"]).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered = df_filtered[
        ["videopath"] + [col for col in df_filtered.columns if col != "videopath"]
    ]
    df_filtered = df_filtered.sort_values(by="original_index")

    save_top_ranked(df_filtered, dataset_name)

    df_filtered_by_entailment = df_filtered_by_entailment.drop(columns=["videopath"]).rename(
        columns={"original_videopath": "videopath"}
    )
    df_filtered_by_entailment = df_filtered_by_entailment[
        ["videopath"] + [col for col in df_filtered_by_entailment.columns if col != "videopath"]
    ]
    df_filtered_by_entailment = df_filtered_by_entailment.sort_values(by="original_index")

    save_top_ranked(df_filtered_by_entailment, f"{dataset_name}-by-entailment")

    for threshold in thresholds:
        filtered_videopaths = df_filtered[
            (df_filtered["label"] == 1) & (df_filtered["entailment"] >= threshold)
        ]["videopath"].values
        df_filtered_threshold = df_filtered[df_filtered["videopath"].isin(filtered_videopaths)]
        if not df_filtered_threshold.empty:
            df_filtered_threshold = df_filtered_threshold.reset_index(drop=True)
            save_top_ranked(
                df_filtered_threshold, f"{dataset_name}_threshold_{threshold}_positive"
            )


def select_all(df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)
    df_train.drop(columns=["text"], inplace=True)

    merged_df = pd.merge(
        df_synth_synth,
        df_synth_real[["videopath", "entailment"]],
        on="videopath",
        suffixes=("_synth", "_real"),
    )
    merged_df["delta"] = merged_df["entailment_synth"] - merged_df["entailment_real"]
    merged_df.drop("entailment_real", axis=1, inplace=True)
    merged_df.rename(columns={"entailment_synth": "entailment"}, inplace=True)
    videos_with_positive_delta = merged_df[merged_df["delta"] > 0]["videopath"].values

    merged_df = merged_df.drop(columns=["video_name", "model", "text"])
    merged_df = merged_df.reset_index(drop=True)
    merged_df = pd.merge(merged_df, df_train, on=["videopath"], how="inner")
    merged_df = merged_df.reset_index(drop=True)

    merged_df_positive_delta = merged_df[merged_df["videopath"].isin(videos_with_positive_delta)]
    merged_df_positive_delta = merged_df_positive_delta.reset_index(drop=True)
    merged_df_positive_delta = merged_df_positive_delta.drop(columns=["videopath"]).rename(
        columns={"original_videopath": "videopath"}
    )
    merged_df_positive_delta = merged_df_positive_delta[
        ["videopath"] + [col for col in merged_df_positive_delta.columns if col != "videopath"]
    ]
    merged_df_positive_delta = merged_df_positive_delta.sort_values(by="original_index")
    # save_top_ranked(merged_df_positive_delta, f"{dataset_name}-positive-delta")

    merged_df = merged_df.drop(columns=["videopath"]).rename(
        columns={"original_videopath": "videopath"}
    )
    merged_df = merged_df[["videopath"] + [col for col in merged_df.columns if col != "videopath"]]
    merged_df = merged_df.sort_values(by="original_index")

    save_top_ranked(merged_df, dataset_name)


def select_all_with_real(df_synth_synth, df_synth_real, df_train, dataset_name, thresholds=[0.5]):
    df_train["original_videopath"] = df_train["videopath"]
    df_train["videopath"] = df_train["videopath"].apply(lambda x: Path(x).stem)

    df_concat = pd.concat([df_synth_synth, df_synth_real])
    df_concat = df_concat.reset_index(drop=True)

    delta_df = pd.merge(
        df_synth_synth,
        df_synth_real[["videopath", "entailment"]],
        on="videopath",
        suffixes=("_synth", "_real"),
    )
    delta_df["delta"] = delta_df["entailment_synth"] - delta_df["entailment_real"]
    delta_df = delta_df[["videopath", "delta"]]

    merged_df = pd.merge(df_concat, delta_df, on="videopath")

    merged_df = merged_df.drop(columns=["model", "video_name"])
    merged_df = merged_df.reset_index(drop=True)
    merged_df = pd.merge(merged_df, df_train, on=["videopath", "text"], how="inner")
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.drop(columns=["videopath", "text"]).rename(
        columns={"original_videopath": "videopath"}
    )
    merged_df = merged_df[["videopath"] + [col for col in merged_df.columns if col != "videopath"]]
    merged_df = merged_df.sort_values(by="original_index")

    save_top_ranked(merged_df, dataset_name)


def select_top_ranked(videocon_csv_files, vqascore_csv_files, dataset_name):
    df_synth_synth_list_videocon, df_synth_real_list_videocon = load_videocon_scores(
        videocon_csv_files
    )
    df_train_list = load_train_data_from_videocon_csv(videocon_csv_files)
    df_synth_synth_list_vqascore, df_synth_real_list_vqascore = load_vqascore_scores(
        vqascore_csv_files
    )

    df_synth_synth_list = df_synth_synth_list_videocon + df_synth_synth_list_vqascore
    df_synth_real_list = df_synth_real_list_videocon + df_synth_real_list_vqascore

    df_synth_synth_concat = pd.concat(df_synth_synth_list)
    df_synth_synth_concat["videopath"] = df_synth_synth_concat["videopath"].apply(
        lambda x: Path(x).stem
    )
    df_synth_synth_concat = df_synth_synth_concat[
        ~df_synth_synth_concat["videopath"].str.contains("highres_tempo_videos_2522_0_0_2")
    ]
    df_synth_synth_concat["video_name"] = df_synth_synth_concat["videopath"].apply(
        lambda x: "_".join(x.split("_")[:-2])
    )
    df_synth_synth_concat = df_synth_synth_concat.reset_index(drop=True)

    df_synth_real_concat = pd.concat(df_synth_real_list)
    df_synth_real_concat["videopath"] = df_synth_real_concat["videopath"].apply(
        lambda x: Path(x).stem
    )
    df_synth_real_concat = df_synth_real_concat[
        ~df_synth_real_concat["videopath"].str.contains("highres_tempo_videos_2522_0_0_2")
    ]
    df_synth_real_concat["video_name"] = df_synth_real_concat["videopath"].apply(
        lambda x: "_".join(x.split("_")[:-2])
    )
    df_synth_real_concat = df_synth_real_concat.reset_index(drop=True)

    df_train_concat = pd.concat(df_train_list)

    for model in df_synth_synth_concat["model"].unique():
        df_synth_synth_concat_model = df_synth_synth_concat[
            df_synth_synth_concat["model"] == model
        ]
        df_synth_synth_concat_model = df_synth_synth_concat_model.reset_index(drop=True)
        df_synth_real_concat_model = df_synth_real_concat[df_synth_real_concat["model"] == model]
        df_synth_real_concat_model = df_synth_real_concat_model.reset_index(drop=True)

        select_all_with_real(
            df_synth_synth_concat_model.copy(deep=True),
            df_synth_real_concat_model.copy(deep=True),
            df_train_concat.copy(deep=True),
            f"{dataset_name}_{model}-all-with-real",
        )
        select_all(
            df_synth_synth_concat_model.copy(deep=True),
            df_synth_real_concat_model.copy(deep=True),
            df_train_concat.copy(deep=True),
            f"{dataset_name}_{model}-all",
        )
        # select_top_ranked_delta(
        #     df_synth_synth_concat_model.copy(deep=True),
        #     df_synth_real_concat_model.copy(deep=True),
        #     df_train_concat.copy(deep=True),
        #     f"{dataset_name}_{model}-delta",
        # )
        # select_top_ranked_delta_only_synth(
        #     df_synth_synth_concat_model.copy(deep=True),
        #     df_synth_real_concat_model.copy(deep=True),
        #     df_train_concat.copy(deep=True),
        #     f"{dataset_name}_{model}-delta-only-synth",
        # )
        # select_top_ranked_from_df(
        #     df_synth_synth_concat_model.copy(deep=True),
        #     df_synth_real_concat_model.copy(deep=True),
        #     df_train_concat.copy(deep=True),
        #     f"{dataset_name}_{model}",
        # )
        # select_top_ranked_from_df(
        #     df_synth_synth_concat_model.copy(deep=True),
        #     None,
        #     df_train_concat.copy(deep=True),
        #     f"{dataset_name}_{model}-only-synth",
        # )

    df_synth_synth_mean = (
        df_synth_synth_concat.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    df_synth_synth_mean["model"] = "mean"
    df_synth_real_mean = (
        df_synth_real_concat.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    df_synth_real_mean["model"] = "mean"
    select_all_with_real(
        df_synth_synth_mean.copy(deep=True),
        df_synth_real_mean.copy(deep=True),
        df_train_concat.copy(deep=True),
        f"{dataset_name}_mean-all-with-real",
    )
    select_all(
        df_synth_synth_mean.copy(deep=True),
        df_synth_real_mean.copy(deep=True),
        df_train_concat.copy(deep=True),
        f"{dataset_name}_mean-all",
    )
    # select_top_ranked_delta(
    #     df_synth_synth_mean.copy(deep=True),
    #     df_synth_real_mean.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-delta",
    # )
    # select_top_ranked_delta_only_synth(
    #     df_synth_synth_mean.copy(deep=True),
    #     df_synth_real_mean.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-delta-only-synth",
    # )
    # select_top_ranked_from_df(
    #     df_synth_synth_mean.copy(deep=True),
    #     df_synth_real_mean.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean",
    # )
    # select_top_ranked_from_df(
    #     df_synth_synth_mean.copy(deep=True),
    #     None,
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-only-synth",
    # )

    df_synth_synth_vqascore = df_synth_synth_concat[df_synth_synth_concat["model"] != "owl-con"]
    df_synth_synth_mean_vqascore = (
        df_synth_synth_vqascore.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    df_synth_synth_mean_vqascore["model"] = "mean-vqascore"
    df_synth_real_vqascore = df_synth_real_concat[df_synth_real_concat["model"] != "owl-con"]
    df_synth_real_mean_vqascore = (
        df_synth_real_vqascore.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    df_synth_real_mean_vqascore["model"] = "mean-vqascore"
    select_all_with_real(
        df_synth_synth_mean_vqascore.copy(deep=True),
        df_synth_real_mean_vqascore.copy(deep=True),
        df_train_concat.copy(deep=True),
        f"{dataset_name}_mean-vqascore-all-with-real",
    )
    select_all(
        df_synth_synth_mean_vqascore.copy(deep=True),
        df_synth_real_mean_vqascore.copy(deep=True),
        df_train_concat.copy(deep=True),
        f"{dataset_name}_mean-vqascore-all",
    )
    # select_top_ranked_delta(
    #     df_synth_synth_mean_vqascore.copy(deep=True),
    #     df_synth_real_mean_vqascore.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-vqascore-delta",
    # )
    # select_top_ranked_delta_only_synth(
    #     df_synth_real_mean_vqascore.copy(deep=True),
    #     df_synth_real_mean_vqascore.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-vqascore-delta-only-synth",
    # )
    # select_top_ranked_from_df(
    #     df_synth_synth_mean_vqascore.copy(deep=True),
    #     df_synth_real_mean_vqascore.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-vqascore",
    # )
    # select_top_ranked_from_df(
    #     df_synth_synth_mean_vqascore.copy(deep=True),
    #     None,
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_mean-vqascore-only-synth",
    # )

    # select_top_ranked_softmax(
    #     df_synth_synth_concat.copy(deep=True),
    #     df_synth_real_concat.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_softmax",
    # )
    # select_top_ranked_softmax(
    #     df_synth_synth_concat.copy(deep=True),
    #     None,
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_softmax-only-synth",
    # )

    df_synth_synth_vqascore = df_synth_synth_concat[df_synth_synth_concat["model"] != "owl-con"]
    df_synth_synth_vqascore = df_synth_synth_vqascore.reset_index(drop=True)
    df_synth_real_vqascore = df_synth_real_concat[df_synth_real_concat["model"] != "owl-con"]
    df_synth_real_vqascore = df_synth_real_vqascore.reset_index(drop=True)
    # select_top_ranked_softmax(
    #     df_synth_synth_vqascore.copy(deep=True),
    #     df_synth_real_vqascore.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_softmax-vqascore",
    # )
    # select_top_ranked_softmax(
    #     df_synth_synth_vqascore.copy(deep=True),
    #     None,
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_softmax-vqascore-only-synth",
    # )

    df_synth_synth_vqascore = df_synth_synth_concat[df_synth_synth_concat["model"] != "owl-con"]
    df_synth_synth_owl_con = df_synth_synth_concat[df_synth_synth_concat["model"] == "owl-con"]
    df_synth_synth_mean_vqascore = (
        df_synth_synth_vqascore.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    df_synth_real_vqascore = df_synth_real_concat[df_synth_real_concat["model"] != "owl-con"]
    df_synth_real_owl_con = df_synth_real_concat[df_synth_real_concat["model"] == "owl-con"]
    df_synth_real_mean_vqascore = (
        df_synth_real_vqascore.groupby(["videopath", "video_name", "text"])["entailment"]
        .mean()
        .reset_index()
    )
    # select_top_ranked_from_df_hybrid(
    #     df_synth_synth_mean_vqascore.copy(deep=True),
    #     df_synth_real_mean_vqascore.copy(deep=True),
    #     df_synth_synth_owl_con.copy(deep=True),
    #     df_synth_real_owl_con.copy(deep=True),
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_hybrid-vqascore-owl-con",
    # )
    # select_top_ranked_from_df_hybrid(
    #     df_synth_synth_mean_vqascore.copy(deep=True),
    #     None,
    #     df_synth_synth_owl_con.copy(deep=True),
    #     None,
    #     df_train_concat.copy(deep=True),
    #     f"{dataset_name}_hybrid-vqascore-owl-con-only-synth",
    # )

    # select_random(df_synth_synth_concat, df_train_concat, dataset_name, n=3)


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    select_top_ranked(args.csv_files_videocon, args.csv_files_vqascore, args.dataset_name)
