#!/bin/bash
conda activate graph
cd ../

ONCO_CODE="STAD"
splitdir="stad"
namesave="stad_exp"
hvtspecific="STAD_2_0.5_random"
GRAPH_DIR="./dataset/TCGA_processed/graphs"

#AMIL
CUDA_VISIBLE_DEVICES=6 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_PatchGCN_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type amil --mode path --max_epochs 20 > ./logs/amil_$namesave.out &
#PATCHGCN
CUDA_VISIBLE_DEVICES=0 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_PatchGCN_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type patchgcn --mode graph --max_epochs 20 > ./logs/patchgcn_$namesave.out &
#TMIL
CUDA_VISIBLE_DEVICES=4 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_PatchGCN_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type tmil --mode path --max_epochs 20 > ./logs/tmil_$namesave.out &
#GTNORIG
CUDA_VISIBLE_DEVICES=7 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_PatchGCN_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type gtn_orig --mode graph --max_epochs 20 --num_gcn_layers 2 --hidden_dim 256 > ./logs/gtn_orig_$namesave.out &
#HVT
CUDA_VISIBLE_DEVICES=5 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_HvtSurv_$hvtspecific/ --split_dir tcga_$splitdir --model_type hvtsurv --mode hvt --max_epochs 20 > ./logs/hvt_$namesave.out &
#Cluster
CUDA_VISIBLE_DEVICES=1 nohup python main.py --data_root_dir $GRAPH_DIR/TCGA_MIL_CLUSTER_$ONCO_CODE/ --split_dir tcga_$splitdir --model_type mifcn --mode cluster --max_epochs 20 > ./logs/mifcn_$namesave.out &
