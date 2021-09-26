import os
import sys
import time
import re

log_dir = '/home/tangdahai.tdh/dahai-logs/{}/'.format(time.strftime("%Y-%m-%d_%H-%M-%S"))
file_dir = log_dir + 'log/'
out_dir = log_dir + 'out/'

run_config_list = {}
run_config_list[' export SAMGRAPH_OMP_NUM_THREADS={}; '] = [12, 20, 24, 28, 32, 40]
run_config_list[' python ./{}'] = ['train_gcn_multi_gpu.py']
run_config_list[' --dataset-path /graph-learning/samgraph/{}'] = ['papers100M']
run_config_list[' --num-sample-worker {}'] =         [1]
run_config_list[' --num-train-worker {}'] =          [1, 2, 3, 4]
run_config_list[' --cache-policy {}'] =              [1]
run_config_list[' --max-copying-jobs {}'] =          [1]
run_config_list[' --num-epoch {}'] =                 [3]
run_config_list[' --cache-percentage {}'] =          [0.22]
run_config_list[' --pipeline '] =                    [1]
mock = True

def run_app():
    if (not os.path.exists(file_dir)):
        cmd = f'mkdir -p {file_dir}; mkdir -p {out_dir}'
        if (mock):
            print(cmd)
        else:
            os.system(cmd)
    total_comb = int(1)
    for k,l in run_config_list.items():
        total_comb *= len(l)
    print('total_comb: ', total_comb)
    config_idx_file = log_dir + f'config_list.txt'
    for comb_idx in range(total_comb):
        file_name = file_dir + f'/{comb_idx}.log'
        out_name = out_dir + f'/{comb_idx}.out.log'
        old_comb_idx = comb_idx
        cmd_str = ''
        config_str = ''
        for k,l in run_config_list.items():
            len_l = len(l)
            mod = comb_idx % len_l
            comb_idx = comb_idx // len_l
            cmd_str += k.format(l[mod])
            config_str += k.format(l[mod]) + '\n'
        cmd_str += f' --log-file {file_name}'
        cmd_str += f' &> {out_name}'
        if (mock):
            print(cmd_str)
        else:
            with open(config_idx_file, 'a+', encoding='utf8') as config_file:
                print(f'\n {old_comb_idx}: ', file=config_file)
                print(config_str, file=config_file)
            os.system(cmd_str)

def run_parse_log(file_dir, pick_map):
    import numpy as np
    for (k,v) in pick_map.items():
        print(k, '\t', end="")
    print('')

    config_pattern = r'config:(.+)=(.+)\n'
    result_pattern = r'test_result:(.+)=(.+)\n'
    files = os.listdir(file_dir)
    for file_name in files:
        if (os.path.isdir(file_name)):
            continue
        with open(file_dir + "/" + file_name, 'r', encoding='utf8') as log_file:
            for line in log_file:
                config = re.match(config_pattern, line)
                result = re.match(result_pattern, line)
                if config:
                    key = config.group(1)
                    value = config.group(2)
                    if (key in pick_map.keys()):
                        pick_map[key] = value
                if result:
                    key = result.group(1)
                    value = result.group(2)
                    if (key in pick_map.keys()):
                        pick_map[key].append(float(value))
            for (k,v) in pick_map.items():
                res = v
                if isinstance(v, list):
                    res = '{:.2f}'.format(np.mean(v[0:]))
                print(res, '\t', end="")
                if isinstance(v, list):
                    v.clear()
            print('')


if __name__ == '__main__':
    # run_app()
    pick_map = { "omp_num_threads" : "", "num_sample_worker" : "", "num_train_worker" : "",
        "sample_time" : [], "copy_time" : [],
        "convert_time" : [], "train_time" : [], "epoch_time" : []}
    run_parse_log('/home/tangdahai.tdh/dahai-logs/2021-09-22_10-30-17/log/', pick_map)
    '''
    '''
