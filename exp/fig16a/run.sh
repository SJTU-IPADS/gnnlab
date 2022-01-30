#!/bin/bash 
switcher_dir=../../example/samgraph/balance_switcher/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/switch/${TIME_STAMPS}
num_epoch=10

mkdir -p $log_dir

# original FGNN
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 1 2>&1 | tee ${log_dir}/switch_origin_1.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 2 2>&1 | tee ${log_dir}/switch_origin_2.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 3 2>&1 | tee ${log_dir}/switch_origin_3.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.18 --num-sample-worker 1 --num-train-worker 4 2>&1 | tee ${log_dir}/switch_origin_4.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.18 --num-sample-worker 1 --num-train-worker 5 2>&1 | tee ${log_dir}/switch_origin_5.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.17 --num-sample-worker 1 --num-train-worker 6 2>&1 | tee ${log_dir}/switch_origin_6.log
python ${switcher_dir}/train_pinsage_no_switch_async.py --dataset papers100M --pipeline --cache-percentage 0.17 --num-sample-worker 1 --num-train-worker 7 2>&1 | tee ${log_dir}/switch_origin_7.log

# dynamic switch
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 1 2>&1 | tee ${log_dir}/switch_balance_1.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 2 2>&1 | tee ${log_dir}/switch_balance_2.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 3 2>&1 | tee ${log_dir}/switch_balance_3.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 4 2>&1 | tee ${log_dir}/switch_balance_4.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 5 2>&1 | tee ${log_dir}/switch_balance_5.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 6 2>&1 | tee ${log_dir}/switch_balance_6.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.05 --pipeline --num-sample-worker 1 --num-train-worker 7 2>&1 | tee ${log_dir}/switch_balance_7.log

# parse log
parse_switch_log() {
  num_trainer=$1
  log_dir=$2
  data_file=$3
  log_file=${log_dir}/switch_balance_${num_trainer}.log
  switch_time=$(cat $log_file | grep -P "] Epoch Time \d*\.\d*" -o | grep -P "\d*\.\d*" -o | head -n 1)
  log_file=${log_dir}/switch_origin_${num_trainer}.log
  origin_time=$(cat $log_file | grep -P "test_result:pipeline_train_epoch_time=\d*\.\d*" -o | grep -P "\d*\.\d*" -o)
  echo "\"1S ${num_trainer}T\" ${origin_time} ${switch_time}" >> $data_file
}

touch fig16a.dat
echo -e "Config\t\"w/o DS\"\t\"w/ DS\"" > fig16a.dat
parse_switch_log 1 ${log_dir} fig16a.dat
parse_switch_log 2 ${log_dir} fig16a.dat
parse_switch_log 3 ${log_dir} fig16a.dat
parse_switch_log 4 ${log_dir} fig16a.dat
parse_switch_log 5 ${log_dir} fig16a.dat
parse_switch_log 6 ${log_dir} fig16a.dat
parse_switch_log 7 ${log_dir} fig16a.dat

# gnuplot
gnuplot fig16a.plt
