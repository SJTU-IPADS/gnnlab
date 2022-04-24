log_path=./run_logs/performance-check/

datatag=`date +%m-%d#%H-%M-%S`
dataset=papers100M
mm_type=""

if [[ ! -z "${DATA_SET}" ]]; then
    dataset=${DATA_SET}
fi
if [[ ! -z "${MM_TYPE}" ]]; then
    mm_type=${MM_TYPE}
else
    exit 1
fi

export ${MM_TYPE}=1
python setup.py clean > /dev/null
python setup.py install > /dev/null 2>&1

log_file=${log_path}${dataset}_${mm_type}_${datatag}.log

# papers
python example/samgraph/train_gcn.py \
--empty-feat 4 -ll info \
--cache-policy degree \
--sample-type khop0 \
--dataset ${dataset} \
--override-device --override-train-device cuda:0 --override-sample-device cuda:1 \
--num-epoch 5 \
>> ${log_file} 2>&1

cat ${log_file} | grep "test_result:epoch_time:sample_time"
cat ${log_file} | grep "step_time:sample_time"
cat ${log_file} | grep "step_time:core_sample_time"
cat ${log_file} | grep "step_time:fill_sample_input_time"
cat ${log_file} | grep "step_time:remap_time"
cat ${log_file} | grep "step_time:sample_kernel_time"
cat ${log_file} | grep "step_time:sample_compact_edge_time"