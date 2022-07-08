log_path=./run_logs/single-sampler-um-multi-gpu/

# dataset=com-friendster
# dataset=papers100M
# dataset=reddit
dataset=uk-2006-05

sample_type=khop2
sampler=gpu
um=1
um_policy=default
um_percent=1
ctx="cuda:1 cpu"


if [[ ! -z "${DATA_SET}" ]]; then
    dataset=$DATA_SET
fi
if [[ ! -z "${SAMPLER}" ]]; then
    sampler=$SAMPLER
fi
if [[ ! -z "${UM}" ]]; then
    um=$UM
fi
if [[ ! -z "${UM_PERCENT}" ]]; then
    um_percent=$UM_PERCENT
fi
if [[ ! -z "${UM_POLICY}" ]]; then
    um_policy=$UM_POLICY
fi
if [[ ! -z "${UM_CTX}" ]]; then
    ctx=$UM_CTX
fi

log_file=${log_path}gcn_`echo ${ctx} | tr " " -`_${dataset}_${um_percent}_`date +%m-%d#%H-%M-%S`.log

cmd="python example/samgraph/train_gcn.py \
    --empty-feat 4 -ll info \
    --cache-policy degree \
    --sample-type ${sample_type} \
    --num-epoch 5 \
    --dataset  ${dataset} \
    --override-device \
    --override-train-device cuda:0 \
    --override-sample-device cuda:1 \
    --um-policy ${um_policy} \
    --unified-memory \
    --unified-memory-ctx ${ctx} \
    --unified-memory-percentage ${um_percent}"


if [ $# -ge 1 ] && [ $1 = "-log" ]; then
    echo $cmd > $log_file
    nvidia-smi >> $log_file
    cmd="$cmd >> ${log_file} 2>&1"
fi

echo $cmd
eval $cmd

if [ -f "${log_file}" ]; then
    cat $log_file | grep "test_result:epoch_time:sample_time"
    cat $log_file | grep "test_result:epoch_time:sample_coo_time"
    cat $log_file | grep "test_result:epoch_time:sample_kernel_time"
    
    cat $log_file | grep "test_result:step_time:sample_time"
    cat $log_file | grep "test_result:step_time:core_sample_time"
    cat $log_file | grep "test_result:step_time:fill_sample_input_time"
    cat $log_file | grep "test_result:step_time:remap_time"
    cat $log_file | grep "test_result:step_time:sample_coo_time"
    cat $log_file | grep "test_result:step_time:sample_kernel_time"
    cat $log_file | grep "test_result:step_time:sample_compact_edge_time"

    cat $log_file | grep "test_result:um_sample_hit_rate"
    cat $log_file | grep "test_result:num_nodes"
    cat $log_file | grep "test_result:num_samples"
    cat $log_file | grep "test_result:sample_thpts(node/sec)"
    cat $log_file | grep "test_result:sample_thpts(gb/sec)"
fi

    