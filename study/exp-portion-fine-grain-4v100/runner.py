import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, SampleType, percent_gen, reverse_percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('root_path', ['/nvme/graph-learning-copy/samgraph/'])
  # .hyper_override(
  #   ['unsupervised', 'max_num_step'], [
  #   [True,           1000],
  #   [False,          10000000],
  # ])
  # .override('batch_size', [
  #   1000,
  #   # 2000,
  #   # 4000,
  #   # 8000,
  # ])
  .override('amp', [True])
  .override('copy_job', [1])
  .override('sample_job', [1])
  # .override('epoch', [10])
  .override('epoch', [3])
  .override('num_sampler', [1])
  .override('num_trainer', [3])
  .override('logdir', [
    # 'run-logs-pipe',
    'run-logs',
  ])
  .override('profile_level', [3])
  # .override('log_level', ['error'])
  .override('log_level', ['warn'])
  .override('multi_gpu', [True])
  .override('pipeline', [
    # True,
    False,
  ]))

cfg_list_collector = ConfigList.Empty()


cur_common_base = (cur_common_base.copy().override('app', [App.graphsage       ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
cur_common_base.override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}'])
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen( 4, 20, 4)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 2, 12, 2) + percent_gen(16, 20, 4)).override('num_feat_dim_hack', [256]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))     # (  2, 10,  2)

cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 2, 18, 2)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 4, 16, 4)).override('num_feat_dim_hack', [256]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))     # (   2,  10, 2)


cfg_list_collector.hyper_override(
  ['cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part_2, "DIRECT", ""],
    [CachePolicy.clique_part_2, "", "MPS"],
    # [CachePolicy.clique_part_2, "", "MPSForLandC"],
    [CachePolicy.rep_2, "DIRECT", ""],
    [CachePolicy.rep_2, "", "MPS"],
    # [CachePolicy.rep_2, "", "MPSForLandC"],
    [CachePolicy.coll_cache_asymm_link_2, "", "MPS"],
    [CachePolicy.coll_cache_asymm_link_2, "", "MPSPhase"],
  ])

# cfg_list_collector.select('cache_policy', [CachePolicy.coll_cache_asymm_link_2]).select('cache_percent', [12/100])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)
  # run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)
