#!/bin/bash
input_data_path=""
output_data_path=""
for aggr in "sum" "max" "mean" "min"; do
    echo "aggr = $aggr"
    python gnn_train.py \
        --Aggregate $aggr \
        --Num_layers 2 \
        --Num_epochs 10 \
        --Batch_size 512 \
        --selective_sampling False \
        --additional_heuristics True \
        --input_data_path $input_data_path \
        --output_data_path $output_data_path
    done
