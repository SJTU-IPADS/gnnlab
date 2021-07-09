#!/bin/bash

LOG_DIR="run-logs"
TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="logs_samgraph_${TIMESTAMP}"

if [ ! -d $LOG_DIR/$OUTPUT_DIR ]; then
    mkdir -p $LOG_DIR/$OUTPUT_DIR
fi

app="gcn graphsage pinsage"
# apps="pinsage"
datasets="reddit products papers100M com-friendster"
# datasets="reddit products"
epochs=10
# epochs=1

echo "Runing evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --dataset $dataset --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/dgl_${app}_${dataset}.log" 2>&1
    done
done
echo
echo