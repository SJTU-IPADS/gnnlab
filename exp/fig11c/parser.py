import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector
import pandas

selected_col = ['cache_policy', 'cache_percentage']
selected_col += ['dataset_short', 'sample_type', 'app']
selected_col += ['hit_percent', 'optimal_hit_percent', 'batch_feat_nbytes', 'batch_miss_nbytes']

if __name__ == '__main__':
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat([BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list], f,selected_col)

  with open(f'data.dat', 'r') as f:
    table = pandas.read_csv(f, sep='\t')
    # by fixing cache size to 5GB, we may calculate performance under different dimension by varying cache rate
    # 54228 / 128 * new_dim * cache_rate/100 = 5000
    # thus we have: cache_rate = 5000 * 128 / cache_rate / 54228 * 100
    table['dim'] = 5120 * 128 / table['cache_percentage'] / 54228 * 100
    table['new_batch_feat_GB'] = table['batch_feat_nbytes'] / 128 * table['dim'] / 1024/1024/1024
    table['new_batch_miss_GB'] = (100 - table['hit_percent'])/100 * table['new_batch_feat_GB']

  with open(f'data.dat', 'w') as f:
    table.to_csv(f, sep='\t', index=None)