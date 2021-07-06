#!/bin/bash

LOG_DIR="run-logs"
TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="logs_${TIMESTAMP}"

if [ ! -d $LOG_DIR/$OUTPUT_DIR ]; then
    mkdir -p $LOG_DIR/$OUTPUT_DIR
fi

apps="graphsage gcn"
datasets="reddit products papers100M com-friendster"

echo "Runing evaluations for dgl..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}.py --parse-args --dataset $dataset > "$LOG_DIR/$OUTPUT_DIR/dgl_${app}.log" 2>&1
    done
done

echo
echo

echo "Runing multi-GPU evaluations for dgl..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python dgl/train_${app}_multi_gpu.py --parse-args --dataset $dataset > "$LOG_DIR/$OUTPUT_DIR/dgl_${app}_mutli_gpu.log" 2>&1
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
