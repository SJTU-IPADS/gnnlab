import os, sys

from numpy import NaN
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['cache_policy_short', 'cache_percentage']
selected_col += ['dataset_short', 'sample_type_short']
# selected_col += ['train_process_time', 'hit_percent', 'epoch_time:train_total', 'epoch_time:copy_time']


# selected_col += ['Step L1 miss nbytes', 'Step L3 cache extract_miss', 'Step L3 cache copy_miss']
selected_col += ['Step L1 miss nbytes MB']
selected_col += ['Step L3 cache extract_miss_bandwidth', 'Step L3 cache copy_miss_bandwidth', 'Step L3 cache combine_miss_bandwidth']

# cfg_list_collector.select('cache_policy', [
#     CachePolicy.cache_by_degree,
# ])

def div_nan(a,b):
  if b == 0:
    return NaN
  return a/b

if __name__ == '__main__':
  with open(f'data.dat', 'w') as f:
    bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
    for bench in bench_list:
      bench.vals['Step L3 cache extract_miss_bandwidth'] = '{:.2f} GB/s'.format(div_nan(bench.vals['Step L1 miss nbytes'], bench.vals['Step L3 cache extract_miss']) / 1024/1024/1024)
      bench.vals['Step L3 cache copy_miss_bandwidth']    = '{:.2f} GB/s'.format(div_nan(bench.vals['Step L1 miss nbytes'], bench.vals['Step L3 cache copy_miss']) / 1024/1024/1024)
      bench.vals['Step L3 cache combine_miss_bandwidth'] = '{:.2f} GB/s'.format(div_nan(bench.vals['Step L1 miss nbytes'], bench.vals['Step L3 cache combine_miss']) / 1024/1024/1024)
      bench.vals['Step L1 miss nbytes MB'] = '{:.2f} MB'.format(bench.vals['Step L1 miss nbytes']/1024/1024)
    BenchInstance.print_dat(bench_list, f,selected_col)