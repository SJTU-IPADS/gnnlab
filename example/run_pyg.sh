#!/bin/bash

LOG_DIR="run-logs"
TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="logs_pyg_${TIMESTAMP}"

if [ ! -d $LOG_DIR/$OUTPUT_DIR ]; then
    mkdir -p $LOG_DIR/$OUTPUT_DIR
fi

apps="gcn graphsage"

datasets="reddit products papers100M com-friendster"
# datasets="reddit"
epochs=2


echo "Running evaluations for single-gpu pyg..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python pyg/train_${app}.py --dataset $dataset --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/pyg_single_gpu_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Running evaluations for multi-gpu pyg"
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python pyg/multi_gpu/train_${app}.py --dataset $dataset --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/pyg_multi_gpu_${app}_${dataset}.log" 2>&1
    done
done
echo
echo