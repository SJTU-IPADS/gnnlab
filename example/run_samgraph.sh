#!/bin/bash

LOG_DIR="run-logs"
TIMESTAMP=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="logs_samgraph_${TIMESTAMP}"

if [ ! -d $LOG_DIR/$OUTPUT_DIR ]; then
    mkdir -p $LOG_DIR/$OUTPUT_DIR
fi

apps="gcn graphsage pinsage"
apps0="gcn"
apps1="graphsage"
apps2="pinsage"
root_path="/graph-learning/samgraph/"
datasets="reddit products papers100M com-friendster"
datasets0="reddit products"
datasets1="papers100M"
datasets2="com-friendster"
epochs=3
cache_by_degree=0
cache_by_heuristic=1

echo "Runing evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --cache-percentage 0.0 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_${app}_${dataset}.log" 2>&1
    done
done
echo
echo

echo "Runing pipeline evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --cache-percentage 0.0 --pipeline --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_pipeline_${app}_${dataset}.log" 2>&1
    done
done
echo
echo



echo "Runing cache evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets0; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --cache-policy ${cache_by_degree} --cache-percentage 1 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_${app}_${dataset}.log" 2>&1
    done

    for dataset in $datasets1; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --cache-policy ${cache_by_heuristic} --cache-percentage 0.25 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_${app}_${dataset}.log" 2>&1
    done

    for dataset in $datasets2; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --cache-policy ${cache_by_degree} --cache-percentage 0.25 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_${app}_${dataset}.log" 2>&1
    done
done
echo
echo


echo "Runing cache+pipeline evaluations for samgraph..."
for app in $apps; do
    echo "  eval $app..."
    for dataset in $datasets0; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --max-copying-jobs 5 --pipeline --cache-policy ${cache_by_degree} --cache-percentage 1 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_pipeline_${app}_${dataset}.log" 2>&1
    done

    for dataset in $datasets1; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --pipeline --cache-policy ${cache_by_heuristic} --cache-percentage 0.25 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_pipeline_${app}_${dataset}.log" 2>&1
    done

    for dataset in $datasets2; do
        echo "    running $dataset..."
        python samgraph/train_${app}.py --arch arch3 --max-copying-jobs 1 --pipeline --cache-policy ${cache_by_degree} --cache-percentage 0.25 --dataset-path  ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_pipeline_${app}_${dataset}.log" 2>&1
    done
done

# for app in $apps1; do
#     echo "  eval $app..."
#     for dataset in $datasets1; do
#         echo "    running $dataset..."
#         python samgraph/train_${app}.py --arch arch3 --pipeline --cache-policy ${cache_by_heuristic} --cache-percentage 0.4 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_pipeline_${app}_${dataset}.log" 2>&1
#     done

#     for dataset in $datasets2; do
#         echo "    running $dataset..."
#         python samgraph/train_${app}.py --arch arch3 --max-copying-jobs 10 --pipeline --cache-policy ${cache_by_heuristic} --cache-percentage 0.4 --dataset-path ${root_path}${dataset} --num-epoch ${epochs} > "$LOG_DIR/$OUTPUT_DIR/samgraph_cache_pipeline_${app}_${dataset}.log" 2>&1
#     done
# done
# echo
# echo