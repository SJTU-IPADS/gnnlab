#!/bin/bash 
log_dir=./run_logs/single/2022-01-30_00-45-28/
# dgl_dir=../../example/dgl/
# dgl_sgnn_dir=../../example/samgraph/sgnn_dgl/
# sam_dir=../../example/samgraph/multi_gpu/
# 
# TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
# log_dir=./run_logs/single/${TIME_STAMPS}
# num_epoch=3
# 
# mkdir -p $log_dir
# 
# #dgl
# 
# python ${dgl_dir}/train_gcn.py --dataset products --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GCN_PR_single.log
# python ${dgl_dir}/train_gcn.py --dataset papers100M --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GCN_PA_single.log
# python ${dgl_dir}/train_gcn.py --dataset twitter --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GCN_TW_single.log
# 
# python ${dgl_dir}/train_graphsage.py --dataset products --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GraphSAGE_PR_single.log
# python ${dgl_dir}/train_graphsage.py --dataset papers100M --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GraphSAGE_PA_single.log
# python ${dgl_dir}/train_graphsage.py --dataset twitter --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_GraphSAGE_TW_single.log
# 
# python ${dgl_sgnn_dir}/train_pinsage.py --dataset products --pipeline --num-epoch ${num_epoch} --num-worker 1 2>&1 | tee ${log_dir}/dgl_PinSAGE_PR_single.log
# python ${dgl_sgnn_dir}/train_pinsage.py --dataset papers100M --pipeline --num-epoch ${num_epoch} --num-worker 1 2>&1 | tee ${log_dir}/dgl_PinSAGE_PA_single.log
# python ${dgl_sgnn_dir}/train_pinsage.py --dataset twitter --pipeline --num-epoch ${num_epoch} --num-worker 1 2>&1 | tee ${log_dir}/dgl_PinSAGE_TW_single.log
# 
# 
# # fgnn
# python ${sam_dir}/train_gcn.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_GCN_PR_single.log
# python ${sam_dir}/train_gcn.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.03 2>&1 | tee ${log_dir}/sam_GCN_PA_single.log
# python ${sam_dir}/train_gcn.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.04 2>&1 | tee ${log_dir}/sam_GCN_TW_single.log
# 
# python ${sam_dir}/train_graphsage.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_GraphSAGE_PR_single.log
# python ${sam_dir}/train_graphsage.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.09 2>&1 | tee ${log_dir}/sam_GraphSAGE_PA_single.log
# python ${sam_dir}/train_graphsage.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.15 2>&1 | tee ${log_dir}/sam_GraphSAGE_TW_single.log
# 
# python ${sam_dir}/train_pinsage.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_PinSAGE_PR_single.log
# python ${sam_dir}/train_pinsage.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.06 2>&1 | tee ${log_dir}/sam_PinSAGE_PA_single.log
# python ${sam_dir}/train_pinsage.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.05 2>&1 | tee ${log_dir}/sam_PinSAGE_TW_single.log


# parse data
parse_data() {
  dataset=$1 # PR, TW, PA
  app=$2 # GCN, GraphSAGE or PinSAGE
  log_dir=$3
  data_file=$4
  log_file=${log_dir}/dgl_${app}_${dataset}_single.log
  dgl_time=$(cat ${log_file} | grep -P "test_result:epoch_time" | tail -n 1 | grep -P "\d*\.\d*" -o)
  log_file=${log_dir}/sam_${app}_${dataset}_single.log
  sam_sample_time=$(cat ${log_file} | grep -P "] Avg Sample Total Time \d*\.\d*" -o | grep -P "\d*\.\d*" -o)
  sam_train_time=$(cat ${log_file} | grep -P "\].*Epoch Time \d*\.\d*" -o | grep -P "\d*\.\d*" -o)
  sam_time=$(echo "scale=2; (${sam_sample_time} + ${sam_train_time}) / 1" | bc)
  echo -e "${dataset}\t${dgl_time}\t${sam_time}\t${app}" >> ${data_file}
}

touch fig16b.dat
echo -e "dataset\tdgl\tfgnn\tapp" > fig16b.dat
parse_data PR GCN ${log_dir} fig16b.dat
parse_data TW GCN ${log_dir} fig16b.dat
parse_data PA GCN ${log_dir} fig16b.dat
parse_data PR GraphSAGE ${log_dir} fig16b.dat
parse_data TW GraphSAGE ${log_dir} fig16b.dat
parse_data PA GraphSAGE ${log_dir} fig16b.dat
parse_data PR PinSAGE ${log_dir} fig16b.dat
parse_data TW PinSAGE ${log_dir} fig16b.dat
parse_data PA PinSAGE ${log_dir} fig16b.dat

gnuplot fig16b.plt
