#!/bin/bash

# Activate the virtual environment
VENV_DIR="/leonardo_scratch/fast/IscrC_UTUVLM/venvs/videocon"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

dataset_name="s("
csv_files_videocon=()
# csv_files_synth_real=()
conditions=("conditioned" "unconditioned")
for i in {1..5}; do
    if [ "$i" -eq 3 ]; then
        continue
    fi
    dataset_name+=$(printf "%03d|" "$i")
    for image_conditioning in "${conditions[@]}"; do
        csv_files_videocon+=("data/final_videocon_s00${i}_${image_conditioning}_synth-synth_scores.csv")
        # csv_files_synth_real+=("data/final_videocon_s00${i}_${image_conditioning}_synth-real_scores.csv")
    done
done
dataset_name="${dataset_name%|})"
if [ "${#conditions[@]}" -eq 1 ]; then
    dataset_name+="_${conditions[0]}"
else
    dataset_name+="_(${conditions[0]}|${conditions[1]})"
fi

csv_files_videocon_string="${csv_files_videocon[*]}"

# csv_files_synth_synth_string="${csv_files_synth_synth[*]}"
# csv_files_synth_real_string="${csv_files_synth_real[*]}"

# echo "${csv_files_synth_synth_string}"

# select_mode="top"
# shellcheck disable=SC2086
# python src/video_selector.py --csv_files_synth_synth ${csv_files_synth_synth_string} --csv_files_synth_real ${csv_files_synth_real_string} --select_mode ${select_mode} --dataset_name ${dataset_name}

# select_mode="random"
# # shellcheck disable=SC2086
# for seed in {0..2}; do
#     python src/video_selector.py --csv_files_synth_synth ${csv_files_synth_synth_string} --csv_files_synth_real ${csv_files_synth_real_string} --select_mode ${select_mode} --dataset_name ${dataset_name} --seed ${seed}
# done

# dataset_name="s("
csv_files_vqascore=()
models=("clip-flant5-xxl" "instructblip-flant5-xxl" "llava-v1.5-13b")
eval_mode=("sample_4_frame")
for i in {1..5}; do
    if [ "$i" -eq 3 ]; then
        continue
    fi
    for image_conditioning in "${conditions[@]}"; do
        # dataset_name+=$(printf "%03d|" "$i")
        for model in "${models[@]}"; do
            for eval_mode in ${eval_mode}; do
                # csv_files+=("data/s00${i}_${model}_${eval_mode}.csv")
                csv_files_vqascore+=("data/videocon_s00${i}_${image_conditioning}_${model}_${eval_mode}.csv")
            done
        done
    done
done

csv_files_vqascore_string="${csv_files_vqascore[*]}"

# dataset_name="${dataset_name%|})"
# for model in ${model}; do
#     for eval_mode in ${eval_mode}; do
#         dataset_name+="_${model}_${eval_mode}"
#     done
# done

# csv_files_string="${csv_files[*]}"
# echo "${csv_files_string}"

# select_mode="top_vqascore"

# python src/video_selector.py --csv_files_synth_synth ${csv_files_string} --select_mode ${select_mode} --dataset_name ${dataset_name}

# csv_files_synth_synth="${csv_files_synth_synth[*]}"
# echo "${csv_files_synth_synth}"
# select_mode="mean"
# python src/video_selector.py --csv_files_synth_synth ${csv_files_synth_synth} --select_mode ${select_mode} --dataset_name ${dataset_name}

# shellcheck disable=SC2086
python src/video_selector.py --csv_files_videocon ${csv_files_videocon_string} --csv_files_vqascore ${csv_files_vqascore_string} --dataset_name ${dataset_name}
