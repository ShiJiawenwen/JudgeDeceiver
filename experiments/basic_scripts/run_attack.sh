#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=6
export m=0
export model=$1 # llama2,llama3,openchat35 or mistral2
export qid=$2 
export device=$3
export dataset=$4  # llmbar or mtbench

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
fi

if [ ! -d "../results/${model}" ]; then
    mkdir "../results/${model}"
    echo "Folder '../results/${model}' created."
else
    echo "Folder '../results/${model}' already exists."
fi

CUDA_VISIBLE_DEVICES=${device} python -u ../main.py \
    --config="../configs/${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../dataset/data_for_train/basic/${dataset}/${dataset}_train_${qid}.csv" \
    --config.result_prefix="../results/${model}/${model}_${dataset}_${qid}" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=600 \
    --config.test_steps=1 \
    --config.align_weight=1.0 \
    --config.enhance_weight=1.0 \
    --config.perplexity_weight=0.1 \
    --config.control_weight=0.1 \
    --config.batch_size=128 \
    --config.data_offset=0

