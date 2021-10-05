#!/bin/bash 
sam_dir=../../samgraph/multi_gpu/
switcher_dir=../../samgraph/balance_switcher/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run_logs/switch/${TIME_STAMPS}
num_epoch=10

mkdir -p $log_dir

# original FGNN
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 1 2>&1 | tee ${log_dir}/switch_no_ddp_origin_1.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 2 2>&1 | tee ${log_dir}/switch_no_ddp_origin_2.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 3 2>&1 | tee ${log_dir}/switch_no_ddp_origin_3.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.19 --num-sample-worker 1 --num-train-worker 4 2>&1 | tee ${log_dir}/switch_no_ddp_origin_4.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.18 --num-sample-worker 1 --num-train-worker 5 2>&1 | tee ${log_dir}/switch_no_ddp_origin_5.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.17 --num-sample-worker 1 --num-train-worker 6 2>&1 | tee ${log_dir}/switch_no_ddp_origin_6.log
python ${sam_dir}/train_pinsage.py --dataset papers100M --no-ddp --pipeline --cache-percentage 0.17 --num-sample-worker 1 --num-train-worker 7 2>&1 | tee ${log_dir}/switch_no_ddp_origin_7.log

# dynamic switch
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 1 2>&1 | tee ${log_dir}/switch_balance_1.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 2 2>&1 | tee ${log_dir}/switch_balance_2.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 3 2>&1 | tee ${log_dir}/switch_balance_3.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 4 2>&1 | tee ${log_dir}/switch_balance_4.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 5 2>&1 | tee ${log_dir}/switch_balance_5.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 6 2>&1 | tee ${log_dir}/switch_balance_6.log
python ${switcher_dir}/train_pinsage.py --dataset papers100M --cache-percentage 0.17 --switch-cache-percentage 0.015 --pipeline --num-sample-worker 1 --num-train-worker 7 2>&1 | tee ${log_dir}/switch_balance_7.log
