import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector
import pandas

selected_col = ['cache_policy', 'cache_percentage']
selected_col += ['dataset_short', 'sample_type', 'app']
selected_col += [
  'init:load_dataset:mmap', # disk to dram
  'init:copy', # dram to gpu
  # 'init:other',
  'init:load_dataset:copy', # dram to gpu: copy dataset to gpu
  # 'init:load_dataset:copy:sampler', # dram to gpu: copy dataset to gpu
  # 'init:load_dataset:copy:trainer', # should be zero
  'init:cache', # dram to gpu: copy cache to gpu
  'init:presample', # presample
  # 'init:dist_queue',
  # 'init:internal',
]

if __name__ == '__main__':
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat([BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list], f,selected_col)
  with open('data.dat', 'r') as f:
    table = pandas.read_csv(f, sep='\t')
    table = table.T
    table = table.rename({
      'init:load_dataset:mmap' : 'Disk to DRAM',
      'init:copy' : 'DRAM to GPU-mem',
      'init:load_dataset:copy' : 'Load graph topology',
      'init:cache' : 'Load feature cache',
      'init:presample' : 'Pre-sampling for PreSC#1'}, axis=0)
  with open('table6.dat', 'w') as f:
    print(table, file=f)