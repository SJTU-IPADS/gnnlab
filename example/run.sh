#!/bin/bash

LOG_DIR="run-logs"
TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="logs_${TIMESTAMP}"

if [ ! -d $LOG_DIR/$OUTPUT_DIR ]; then
    mkdir -p $LOG_DIR/$OUTPUT_DIR
fi

#app="gcn graphsage pinsage"
apps="pinsage"
datasets="reddit products papers100M com-friendster"
# datasets="reddit products"
epochs=3

echo "Runing evaluations for dgl..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}.py --dataset $dataset --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/dgl_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Runing evaluations for dgl pipelining..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}.py --dataset $dataset --pipelining --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/dgl_pipelining_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Runing multi-GPU evaluations for dgl..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}_multi_gpu.py --dataset $dataset --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/dgl_mutli_gpu_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Runing multi-GPU evaluations for dgl pipelining..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}_multi_gpu.py --dataset $dataset --pipelining --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/dgl_mutli_gpu_pipelining_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Runing evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
    done
done
