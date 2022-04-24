import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

def copy_optimal(cfg: RunConfig):
  os.system(f"rm -f \"{cfg.get_log_fname()}_optimal_cache_hit.txt\"")
  os.system(f"mv node_access_optimal_cache_hit* \"{cfg.get_log_fname()}_optimal_cache_hit.txt\"")
  os.system(f"rm -f node_access_optimal_cache_*")
  os.system(f"rm -f node_access_freq*")

cur_common_base = (ConfigList()
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('pipeline', [False])
  .override('logdir', ['run-logs',])
  .override('cache_policy', [CachePolicy.cache_by_degree])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('cache_percent', percent_gen(10, 10, 1))
  .override('num_hidden', [2])
  .override('multi_gpu', [True]))

cur_common_base.override('cache_policy', [
  # CachePolicy.cache_by_random,
  CachePolicy.cache_by_degree,
  CachePolicy.cache_by_presample_1
])

cfg_list_collector = ConfigList.Empty()

cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-10']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=10']).override('train_set_percent',[10]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-8']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=8']).override('train_set_percent',[8]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-6']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=6']).override('train_set_percent',[6]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-3']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=3']).override('train_set_percent',[3]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-2']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=2']).override('train_set_percent',[2]))

cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-1']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=1']).override('train_set_percent',[1]))

cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-0.3']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0.3']).override('train_set_percent',[0.3]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-0.1']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0.1']).override('train_set_percent',[0.1]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-0.03']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0.03']).override('train_set_percent',[0.03]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-0.01']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0.01']).override('train_set_percent',[0.01]))
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/percent-0.003']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0.003']).override('train_set_percent',[0.003]))

cur_common_base = cfg_list_collector.copy()
cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))

cfg_list_collector.override('dataset', [
    Dataset.products,   
    Dataset.twitter,    
    Dataset.papers100M, 
    Dataset.uk_2006_05, 
  ])

cur_common_base = cfg_list_collector.copy().select('cache_policy', [CachePolicy.cache_by_degree])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  # run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)
  cur_common_base.override('arch', [Arch.arch3]).override('multi_gpu', [False]).override('report_optimal', [1]).override('cache_percent', [0]).override('epoch', [20])
  run_in_list(cur_common_base.conf_list, do_mock, durable_log, copy_optimal)
