#!/bin/bash 
dgl_dir=../../example/dgl/multi_gpu/
sam_dir=../../example/samgraph/multi_gpu/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/acc_test/one/${TIME_STAMPS}

# TODO: need change these configs
dgl_devices="0 1 2 3 4 5 6 7"
num_sam_sampler=2
num_sam_trainer=6

mkdir -p $log_dir

# papers100M acc: 56%
python ${dgl_dir}/train_graphsage.py --dataset papers100M --pipelining --report-acc 151 --num-epoch 200 --use-gpu-sampling --devices ${dgl_devices} > ${log_dir}/dgl_papers.log 2> ${log_dir}/dgl_papers.err.log
python ${sam_dir}/train_graphsage.py --dataset papers100M --cache-percentage 0.20 --pipeline --report-acc 151 --num-epoch 200 --num-sample-worker ${num_sam_sampler} --num-train-worker ${num_sam_trainer} > ${log_dir}/sam_papers.log 2> ${log_dir}/sam_papers.err.log

num_sam_sampler=4
num_sam_trainer=4

# products   acc: 91%
python ${dgl_dir}/train_graphsage.py --dataset products --pipelining --report-acc 25 --num-epoch 200 --use-gpu-sampling --devices ${dgl_devices} > ${log_dir}/dgl_products.log 2> ${log_dir}/dgl_products.err.log
python ${sam_dir}/train_graphsage.py --dataset products --cache-percentage 1.0 --pipeline --report-acc 25 --num-epoch 200 --num-sample-worker ${num_sam_sampler} --num-train-worker ${num_sam_trainer} > ${log_dir}/sam_products.log 2> ${log_dir}/sam_products.err.log


# parse data
touch acc_one.res
echo -e "system\tdataset\tbatch_size\ttime\tacc" >> acc_one.res
python ./parse_acc.py -f ${log_dir}/dgl_papers.log --system dgl --dataset papers --batch-size 8000 >> acc_one.res
python ./parse_acc.py -f ${log_dir}/sam_papers.log --system fgnn --dataset papers --batch-size 8000 >> acc_one.res

python ./parse_acc.py -f ${log_dir}/dgl_products.log --system dgl --dataset products --batch-size 8000 >> acc_one.res
python ./parse_acc.py -f ${log_dir}/sam_products.log --system fgnn --dataset products --batch-size 8000 >> acc_one.res

# gnuplot
gnuplot ./acc-timeline-pa.plt
gnuplot ./acc-timeline-pr.plt
