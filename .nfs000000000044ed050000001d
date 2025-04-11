#!/bin/bash
input_data_path="/mnt/nas/deekshak/ir3/GNN-based-Retrieval-for-QA/data/train_graph_dict_sampler.pt"
output_data_path="/mnt/nas/mohitsinghtomar/project_experiments/ir_assignment/assignment_3/saved_model"
for aggr in "sum" "max" "mean" "min"; do
    echo "aggr = $aggr"
    python gnn_train.py \
        --Aggregate $aggr \
        --Num_layers 2 \
        --Num_epochs 10 \
        --Batch_size 512 \
        --selective_sampling True \
        --input_data_path $input_data_path \
        --output_data_path $output_data_path
    done
