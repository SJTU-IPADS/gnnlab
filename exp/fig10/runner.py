import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

def copy_optimal(cfg: RunConfig):
  os.system(f"rm -f \"{cfg.get_log_fname()}_optimal_cache_hit.txt\"")
  os.system(f"mv node_access_optimal_cache_hit* \"{cfg.get_log_fname()}_optimal_cache_hit.txt\"")
  os.system(f"rm -f node_access_optimal_cache_*")

cur_common_base = (ConfigList()
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('pipeline', [False])
  .override('epoch', [3])
  .override('logdir', ['run-logs',])
  .override('cache_policy', [CachePolicy.cache_by_degree])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('multi_gpu', [True])
  .override('dataset', [
    Dataset.products,   
    Dataset.twitter,    
    Dataset.papers100M, 
    Dataset.uk_2006_05, 
  ]))

cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))

cur_common_base = cfg_list_collector.copy()

cfg_list_collector.override('cache_policy', [
  CachePolicy.cache_by_random,
  CachePolicy.cache_by_degree,
  CachePolicy.cache_by_presample_1
]).override('cache_percent', percent_gen(10, 10, 1))

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)
  cur_common_base.override('arch', [Arch.arch3]).override('multi_gpu', [False]).override('report_optimal', [1]).override('cache_percent', [0])
  run_in_list(cur_common_base.conf_list, do_mock, durable_log, copy_optimal)
