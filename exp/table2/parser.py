import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector
import pandas

selected_col = []
selected_col += ['dataset_short', 'sample_type', 'app']
selected_col += ['node_access:epoch_similarity']

if __name__ == '__main__':
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat([BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list], f,selected_col)

  with open(f'data.dat', 'r') as f, open('table2.dat', 'w') as table2:
    a = pandas.read_csv(f, sep="\t")
    a = a.pivot_table(values=['node_access:epoch_similarity'], columns=['dataset_short'], index=['sample_type', 'app'])
    a = a[[('node_access:epoch_similarity', 'PR'), ('node_access:epoch_similarity', 'TW'), ('node_access:epoch_similarity', 'PA'), ('node_access:epoch_similarity', 'UK')]]
    print(a, file=table2)