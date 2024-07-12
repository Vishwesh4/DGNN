#!/bin/bash
conda activate graph
cd ../

ONCO_CODE="BRCA"
splitdir="brca"
ONCO_CODE1="${ONCO_CODE}_tumor_stroma_v2"
ONCO_CODE2="${ONCO_CODE}_tumor_rest_v2"
ONCO_CODE3="${ONCO_CODE}_stroma_rest_v2"
cust_string="dgnn_interaction"

#DGNN
CUDA_VISIBLE_DEVICES=0 nohup python main.py --data_root_dir /localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_no_sample_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type dgnn --mode graph --max_epochs 20 --num_gcn_layers 2 --hidden_dim 128 --add_edge_attr --add_pe > ./logs/${cust_string}_$ONCO_CODE.out &
sleep 60
CUDA_VISIBLE_DEVICES=1 nohup python main.py --data_root_dir /localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_no_sample_$ONCO_CODE1/ --split_dir tcga_$splitdir --model_type dgnn --mode graph --max_epochs 20 --num_gcn_layers 2 --hidden_dim 128 --add_edge_attr --add_pe > ./logs/${cust_string}_$ONCO_CODE1.out &
sleep 60
CUDA_VISIBLE_DEVICES=2 nohup python main.py --data_root_dir /localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_no_sample_$ONCO_CODE2/ --split_dir tcga_$splitdir --model_type dgnn --mode graph --max_epochs 20 --num_gcn_layers 2 --hidden_dim 128 --add_edge_attr --add_pe > ./logs/${cust_string}_$ONCO_CODE2.out &
sleep 60
CUDA_VISIBLE_DEVICES=3 nohup python main.py --data_root_dir /localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_no_sample_$ONCO_CODE3/ --split_dir tcga_$splitdir --model_type dgnn --mode graph --max_epochs 20 --num_gcn_layers 2 --hidden_dim 128 --add_edge_attr --add_pe > ./logs/${cust_string}_$ONCO_CODE3.out &

