#!/bin/bash 
dgl_dir=../../dgl/
sam_dir=../../samgraph/multi_gpu/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run_logs/single/${TIME_STAMPS}
num_epoch=10

mkdir -p $log_dir

#dgl

# python ${dgl_dir}/train_gcn.py --dataset products --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_gcn_products_single.log
# python ${dgl_dir}/train_gcn.py --dataset papers100M --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_gcn_paper_single.log
# python ${dgl_dir}/train_gcn.py --dataset twitter --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_gcn_twitter_single.log
# 
# python ${dgl_dir}/train_graphsage.py --dataset products --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_graphsage_products_single.log
# python ${dgl_dir}/train_graphsage.py --dataset papers100M --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_graphsage_papers_single.log
# python ${dgl_dir}/train_graphsage.py --dataset twitter --pipelining --num-epoch ${num_epoch} --use-gpu-sampling 2>&1 | tee ${log_dir}/dgl_graphsage_twitter_single.log

# multi_gpu for single gpu run
python ${sam_dir}/train_gcn.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_gcn_products_single.log
python ${sam_dir}/train_gcn.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.03 2>&1 | tee ${log_dir}/sam_gcn_papers_single.log
python ${sam_dir}/train_gcn.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.04 2>&1 | tee ${log_dir}/sam_gcn_twitter_single.log

python ${sam_dir}/train_graphsage.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_graphsage_products_single.log
python ${sam_dir}/train_graphsage.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.07 2>&1 | tee ${log_dir}/sam_graphsage_papers_single.log
python ${sam_dir}/train_graphsage.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.13 2>&1 | tee ${log_dir}/sam_graphsage_twitter_single.log
python ${sam_dir}/train_graphsage.py --dataset uk-2006-05 --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.0 2>&1 | tee ${log_dir}/sam_graphsage_uk_single.log

python ${sam_dir}/train_pinsage.py --dataset products --single-gpu --num-epoch ${num_epoch} --cache-percentage 1.0 2>&1 | tee ${log_dir}/sam_pinsage_products_single.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.05 2>&1 | tee ${log_dir}/sam_pinsage_papers_single.log
python ${sam_dir}/train_pinsage.py --dataset twitter --single-gpu --num-epoch ${num_epoch} --cache-percentage 0.06 2>&1 | tee ${log_dir}/sam_pinsage_twitter_single.log
