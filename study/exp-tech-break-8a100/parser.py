import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['policy_impl', 'cache_percentage', 'batch_size']
# selected_col += ['unsupervised']
selected_col += ['dataset_short', 'pipeline']
selected_col += ['Step(average) L1 sample']
selected_col += ['Step(average) L1 recv']
selected_col += ['Step(average) L2 feat copy']
selected_col += ['Step(average) L1 train total']
selected_col += ['Time.L','Time.R','Time.C']
selected_col += ['Wght.L','Wght.R','Wght.C']
selected_col += ['optimal_local_rate','optimal_remote_rate','optimal_cpu_rate']
selected_col += ['Thpt.L','Thpt.R','Thpt.C']
selected_col += ['SizeGB.L','SizeGB.R','SizeGB.C']
selected_col += ['coll_cache:local_cache_rate']
selected_col += ['coll_cache:remote_cache_rate']
selected_col += ['coll_cache:global_cache_rate']
selected_col += ['train_process_time', 'epoch_time:train_total', 'epoch_time:copy_time']
selected_col += ['coll_cache:z']


cfg_list_collector = (cfg_list_collector.copy()
  # .select('dataset', [Dataset.twitter, Dataset.uk_2006_05])
  # .select('cache_policy', [CachePolicy.coll_cache_10])
  .select('pipeline', [False])
  # .override_T('logdir', [
  #   # 'run-logs-backup-pcvyatta',
  #   # 'run-logs-backup',
  #   'run-logs-backup-12-05',
  # ])
)

def div_nan(a,b):
  if b == 0:
    return math.nan
  return a/b

def max_nan(a,b):
  if math.isnan(a):
    return b
  elif math.isnan(b):
    return a
  else:
    return max(a,b)

def handle_nan(a, default=0):
  if math.isnan(a):
    return default
  return a
def zero_nan(a):
  return handle_nan(a, 0)

def short_app_name(inst: BenchInstance):
  suffix = "_unsup" if inst.get_val('unsupervised') else "_sup"
  inst.vals['short_app'] = inst.get_val('sample_type_short') + suffix

def full_policy_name(inst: BenchInstance):
  inst.vals['policy_impl'] = inst.get_val('coll_cache_concurrent_link') + inst.get_val('cache_policy_short')

if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  for inst in bench_list:
    inst : BenchInstance
    short_app_name(inst)
    full_policy_name(inst)
    try:
      inst.vals['Step(average) L1 train total'] = inst.get_val('Step(average) L1 convert time') + inst.get_val('Step(average) L1 train')
      # when cache rate = 0, extract time has different log name...
      inst.vals['Step(average) L2 feat copy'] = max_nan(inst.get_val('Step(average) L2 cache feat copy'), inst.get_val('Step(average) L2 extract'))

      # per-step feature nbytes (Remote, Cpu, Local)
      inst.vals['Size.A'] = inst.get_val('Step(average) L1 feature nbytes')
      inst.vals['Size.R'] = handle_nan(inst.get_val('Step(average) L1 remote nbytes'), 0)
      inst.vals['Size.C'] = handle_nan(inst.get_val('Step(average) L1 miss nbytes'), inst.vals['Size.A'])
      inst.vals['Size.L'] = inst.get_val('Size.A') - inst.get_val('Size.C') - inst.get_val('Size.R')

      inst.vals['SizeGB.R'] = inst.get_val('Size.R') / 1024 / 1024 / 1024
      inst.vals['SizeGB.C'] = inst.get_val('Size.C') / 1024 / 1024 / 1024
      inst.vals['SizeGB.L'] = inst.get_val('Size.L') / 1024 / 1024 / 1024

      # per-step extraction time
      inst.vals['Time.R'] = handle_nan(inst.get_val('Step(average) L3 cache combine remote'))
      inst.vals['Time.C'] = handle_nan(inst.get_val('Step(average) L3 cache combine_miss'), inst.get_val('Step(average) L2 extract'))
      inst.vals['Time.L'] = handle_nan(inst.get_val('Step(average) L3 cache combine cache'))

      # per-step extraction throughput (GB/s)
      inst.vals['Thpt.R'] = div_nan(inst.get_val('Size.R'), inst.get_val('Time.R')) / 1024 / 1024 / 1024
      inst.vals['Thpt.C'] = div_nan(inst.get_val('Size.C'), inst.get_val('Time.C')) / 1024 / 1024 / 1024
      inst.vals['Thpt.L'] = div_nan(inst.get_val('Size.L'), inst.get_val('Time.L')) / 1024 / 1024 / 1024

      # per-step extraction portion from different source
      inst.vals['Wght.R'] = inst.get_val('Size.R') / inst.get_val('Size.A') * 100
      inst.vals['Wght.C'] = inst.get_val('Size.C') / inst.get_val('Size.A') * 100
      inst.vals['Wght.L'] = 100 - inst.get_val('Wght.R') - inst.get_val('Wght.C')
    except Exception as e:
      print("Error when " + inst.cfg.get_log_fname() + '.log')
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)