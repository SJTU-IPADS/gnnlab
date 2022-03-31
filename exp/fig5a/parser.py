import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['cache_policy', 'cache_percentage']
selected_col += ['dataset_short', 'sample_type', 'app']
selected_col += ['hit_percent', 'optimal_hit_percent', 'batch_miss_nbytes', 'batch_feat_nbytes']

if __name__ == '__main__':
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat([BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list], f,selected_col)

