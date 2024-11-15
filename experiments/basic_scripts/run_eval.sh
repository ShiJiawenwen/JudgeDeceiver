#!/bin/bash

export model=$1  # llama2,llama3,openchat35 or mistral2
export dataset=$2  # llmbar or mtbench


if [ ! -d "../eval" ]; then
    mkdir "../eval"
    echo "Folder '../eval' created."
fi

if [ ! -d "../eval/${model}" ]; then
    mkdir "../eval/${model}"
    echo "Folder '../eval/${model}' created."
else
    echo "Folder '../eval/${model}' already exists."
fi

for i in {1,2,3,4,5,6,7,8,9,10,}; do
    python -u ../evaluate.py \
        --config="../configs/${model}.py" \
        --config.train_data="../../dataset/data_for_eval/basic/${dataset}/${dataset}_test_${i}.csv" \
        --config.logfile="../results/${model}/${model}_${dataset}_${i}.json" \
        --config.n_train_data=50 \
        --config.n_test_data=100 \
        --config.eval_model="${model}"
done
