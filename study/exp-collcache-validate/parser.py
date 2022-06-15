import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['cache_policy_short', 'cache_percentage']
selected_col += ['dataset_short', 'sample_type_short', 'pipeline']
selected_col += ['train_process_time', 'epoch_time:train_total', 'epoch_time:copy_time']
selected_col += ['Step L1 recv']
selected_col += ['Step L2 cache feat copy']
selected_col += ['Step L1 train total']
selected_col += ['Wght.L','Wght.R','Wght.C', 'TA']
selected_col += ['Thpt.L','Thpt.R','Thpt.C']
selected_col += ['Time.L','Time.R','Time.C']
selected_col += ['SizeGB.L','SizeGB.R','SizeGB.C']
selected_col += ['logdir']

# cfg_list_collector.override_T('logdir', ['run-logs-legacy', 'run-logs-coll']).override('cache_impl', ['coll'])
# cfg_list_collector.part_override('logdir', ['run-logs-legacy'], 'cache_impl', ['legacy'])
T_local = 1
T_remote = 2.08
T_cpu = 45.45

cfg_list_collector = (cfg_list_collector.copy()
  # .select('dataset', [Dataset.twitter, Dataset.uk_2006_05])
  # .select('cache_policy', [CachePolicy.coll_cache_10])
  .select('pipeline', [False])
  # .override_T('logdir', [
  #   # 'run-logs-backup-pcvyatta',
  #   # 'run-logs-backup',
  #   'run-logs',
  # ])
)

def div_nan(a,b):
  if b == 0:
    return NaN
  return a/b

if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  for inst in bench_list:
    inst : BenchInstance
    try:
      inst.vals['Step L1 train total'] = inst.vals['Step L1 convert time'] + inst.vals['Step L1 train']
      inst.vals['Size.R'] = inst.vals['Step L1 remote nbytes']
      inst.vals['Size.C'] = inst.vals['Step L1 miss nbytes']
      inst.vals['Size.L'] = inst.vals['Step L1 feature nbytes'] - inst.vals['Size.C'] - inst.vals['Size.R']

      inst.vals['SizeGB.R'] = inst.vals['Size.R'] / 1024 / 1024 / 1024
      inst.vals['SizeGB.C'] = inst.vals['Size.C'] / 1024 / 1024 / 1024
      inst.vals['SizeGB.L'] = inst.vals['Size.L'] / 1024 / 1024 / 1024

      inst.vals['Time.R'] = inst.vals['Step L3 cache combine remote']
      inst.vals['Time.C'] = inst.vals['Step L3 cache combine_miss']
      inst.vals['Time.L'] = inst.vals['Step L3 cache combine cache']

      inst.vals['Thpt.R'] = div_nan(inst.vals['Size.R'], inst.vals['Step L3 cache combine remote']) / 1024 / 1024 / 1024
      inst.vals['Thpt.C'] = div_nan(inst.vals['Size.C'], inst.vals['Step L3 cache combine_miss']) / 1024 / 1024 / 1024
      inst.vals['Thpt.L'] = div_nan(inst.vals['Size.L'], inst.vals['Step L3 cache combine cache']) / 1024 / 1024 / 1024

      inst.vals['Wght.R'] = inst.vals['Step L1 remote nbytes'] / inst.vals['Step L1 feature nbytes'] * 100
      inst.vals['Wght.C'] = inst.vals['Step L1 miss nbytes'] / inst.vals['Step L1 feature nbytes'] * 100
      inst.vals['Wght.L'] = 100 - inst.vals['Wght.R'] - inst.vals['Wght.C']
      inst.vals['TA'] = inst.vals['Wght.L'] * T_local + inst.vals['Wght.R'] * T_remote + inst.vals['Wght.C'] * T_cpu
    except Exception as e:
      print("Error when " + inst.cfg.get_log_fname() + '.log')
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)