log_path=./run_logs/performance-check/

datatag=`date +%m-%d#%H-%M-%S`

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1,2,3"

# papers
python example/samgraph/train_gcn.py \
--empty-feat 4 -ll info \
--cache-policy degree \
--sample-type khop0 \
--dataset papers100M \
--override-device --override-train-device cuda:0 --override-sample-device cuda:1 \
--num-epoch 5 \
>> ${log_path}papers100M_p2p_${datatag}.log 2>&1

cat ${log_path}papers100M_p2p_${datatag}.log 2>&1 | grep "test_result:epoch_time:sample_time"

# uk
python example/samgraph/train_gcn.py \
--empty-feat 4 -ll info \
--cache-policy degree \
--sample-type khop0 \
--dataset uk-2006-05 \
--override-device --override-train-device cuda:0 --override-sample-device cuda:1 \
--num-epoch 5 \
>> ${log_path}uk-2006-05_p2p_${datatag}.log 2>&1

cat ${log_path}uk-2006-05_p2p_${datatag}.log 2>&1 | grep "test_result:epoch_time:sample_time"
