log_path=./run_logs/

dataset=com-friendster
# dataset=papers100M
# dataset=reddit

sample_type=khop0
sampler=gpu


if [[ ! -z "${DATA_SET}" ]]; then
    dataset=$DATA_SET
fi
if [[ ! -z "${SAMPLER}" ]]; then
    sampler=$SAMPLER
fi

log_file=${log_path}gcn_${dataset}_`date +%m-%d#%H-%M-%S`.log

cmd="python example/samgraph/train_gcn.py \
    --empty-feat 4 -ll info \
    --cache-policy degree \
    --sample-type ${sample_type} \
    --dataset  ${dataset}"

if [ $sampler = "cpu" ]; then
    cmd="${cmd} --arch arch0"
else
    cmd="${cmd} \
        --override-device \
        --override-train-device cuda:0 \
        --override-sample-device cuda:1"
fi


if [ $# -ge 1 ] && [ $1 = "-log" ]; then
    echo $cmd > $log_file
    nvidia-smi >> $log_file
    cmd="$cmd >>${log_file} 2>&1"
fi

echo $cmd
eval $cmd

if [ -f "${log_file}" ]; then
    cat $log_file | grep "test_result:epoch_time:sample_time"
fi

    