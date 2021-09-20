import os
import sys
import time
import re

dataset_root = '/graph-learning/samgraph/'
log_dir = '/home/tangdahai.tdh/dahai-logs/{}/'.format(time.strftime("%Y-%m-%d_%H-%M-%S"))

dataset_list = ['papers100M']
# dataset_list = ['reddit']
omp_thread_list = [4,8,12,16,24,32,40,48,64,72,86,96]
# omp_thread_list = [4]
num_sampler = [1]
num_trainer = [1]
cache_policy = [1]
max_copying_jobs = [1]
mock = False

def run_app():
    if (not os.path.exists(log_dir)):
        cmd = f'mkdir -p {log_dir}'
        if (mock):
            print(cmd)
        else:
            os.system(cmd)
    for dataset in dataset_list:
        log_file_prefix = log_dir + dataset
        dataset = dataset_root + dataset
        for omp_num_thread in omp_thread_list:
            log_file = log_file_prefix + f'_{omp_num_thread}'
            cmd = f'export SAMGRAPH_OMP_NUM_THREADS={omp_num_thread}; '
            cmd += f'python train_gcn_multi_gpu.py --dataset {dataset}'
            cmd += f' --num-epoch 3 --cache-percentage 0.22 --num-sample-worker 1 --num-train-worker 1 --cache-policy 1 --max-copying-jobs 1'
            cmd += f' --log-file {log_file}'
            if (mock):
                print(cmd)
            else:
                os.system(cmd)

def run_parse_log(log_dir, pick_map):
    config_pattern = r'config:(.+)=(.+)\n'
    result_pattern = r'test_result:(.+)=(.+)\n'
    files = os.listdir(log_dir)
    for file_name in files:
        if (os.path.isdir(file_name)):
            continue
        with open(log_dir + "/" + file_name, 'r', encoding='utf8') as log_file:
            for line in log_file:
                config = re.match(config_pattern, line)
                result = re.match(result_pattern, line)
                if config:
                    key = config.group(1)
                    value = config.group(2)
                if result:
                    key = result.group(1)
                    value = result.group(2)
                if (key in pick_map.keys()):
                    pick_map[key] = value
            for (k,v) in pick_map.items():
                print(v, '\t', end="")
            print('')


if __name__ == '__main__':
    # run_app()
    pick_map = { "omp_num_threads" : "", "sample_time" : "", "copy_time" : "",
            "convert_time" : "", "train_time" : "", "epoch_time" : ""}
    run_parse_log('/home/tangdahai.tdh/dahai-logs/2021-09-20_22-43-46/', pick_map)
