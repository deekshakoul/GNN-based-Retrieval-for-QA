#!/bin/bash
set -e
aggregate=$1
num_layers=$2
cuda_device=$3
gnn_retrived_file_name="data/output/retrieval_gnn_${aggregate}_layers_${num_layers}_heuristics21.json"
model_path="/mnt/nas/mohitsinghtomar/project_experiments/ir_assignment/assignment_3/saved_model/gnn_model_num_layer_${num_layers}_aggregate_${aggregate}_additional_heuristics.pt"
CUDA_VISIBLE_DEVICES=$cuda_device python3 eval.py \
                            --gcn_model_path $model_path \
                            --data_path "data/dev_graph_dict_Heuristics21.pt" \
                            --num_layers $num_layers \
                            --aggregate "$aggregate" \
                            --gnn_retrived_file_name $gnn_retrived_file_name

CUDA_VISIBLE_DEVICES=$cuda_device python3 baseline_eval.py \
                            --gnn_file $gnn_retrived_file_name\
                            --model_type "gnn" 
