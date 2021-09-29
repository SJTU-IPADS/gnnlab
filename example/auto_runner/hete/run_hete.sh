#!/bin/bash 
dgl_dir=../../dgl/multi_gpu/
sam_dir=../../samgraph/multi_gpu/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run_logs/hete/${TIME_STAMPS}

# data_set=reddit
# num_epoch=1
data_set=papers100M
num_epoch=5

mkdir -p $log_dir

export CUDA_VISIBLE_DEVICES=2,3
# dgl hete GPU
python ${dgl_dir}/train_gcn.py --use-gpu-sampling --num-epoch ${num_epoch} --dataset ${data_set} --fanout 5 10 15 > ${log_dir}/hete_dgl_gcn.log 2> ${log_dir}/hete_dgl_gcn.err.log
python ${dgl_dir}/train_graphsage.py --use-gpu-sampling --num-epoch  ${num_epoch} --dataset ${data_set} --fanout 25 10 > ${log_dir}/hete_dgl_graphsage.log 2> ${log_dir}/hete_dgl_graphsage.err.log
python ${dgl_dir}/train_pinsage.py --num-epoch ${num_epoch} --dataset ${data_set} --random-walk-length 3 --random-walk-restart-prob 0.5 --num-random-walk 4 --num-neighbor 5 > ${log_dir}/hete_dgl_pinsage.log 2> ${log_dir}/hete_dgl_pinsage.err.log

# FGNN hete GPU
python ${sam_dir}/train_gcn.py --num-epoch ${num_epoch} --dataset ${data_set} --fanout 5 10 15 --cache-percentage 0.20 > ${log_dir}/hete_sam_gcn.log 2> ${log_dir}/hete_sam_gcn.err.log
python ${sam_dir}/train_graphsage.py --num-epoch  ${num_epoch} --dataset ${data_set} --fanout 25 10 --cache-percentage 0.24 > ${log_dir}/hete_sam_graphsage.log 2> ${log_dir}/hete_sam_graphsage.err.log
python ${sam_dir}/train_pinsage.py --num-epoch ${num_epoch} --dataset ${data_set} --random-walk-length 3 --random-walk-restart-prob 0.5 --num-random-walk 4 --num-neighbor 5 --cache-percentage 0.19 > ${log_dir}/hete_sam_pinsage.log 2> ${log_dir}/hete_sam_pinsage.err.log

# FGNN hete GPU pipeline
python ${sam_dir}/train_gcn.py --num-epoch ${num_epoch} --dataset ${data_set} --fanout 5 10 15 --cache-percentage 0.20 --omp-thread-num 20 --pipeline > ${log_dir}/hete_sam_gcn_pipe.log 2> ${log_dir}/hete_sam_gcn_pipe.err.log
python ${sam_dir}/train_graphsage.py --num-epoch  ${num_epoch} --dataset ${data_set} --fanout 25 10 --cache-percentage 0.24 --pipeline > ${log_dir}/hete_sam_graphsage_pipe.log 2> ${log_dir}/hete_sam_graphsage_pipe.err.log
python ${sam_dir}/train_pinsage.py --num-epoch ${num_epoch} --dataset ${data_set} --random-walk-length 3 --random-walk-restart-prob 0.5 --num-random-walk 4 --num-neighbor 5 --cache-percentage 0.19 --pipeline > ${log_dir}/hete_sam_pinsage_pipe.log 2> ${log_dir}/hete_sam_pinsage_pipe.err.log
