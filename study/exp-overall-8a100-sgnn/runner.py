import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, SampleType, percent_gen, reverse_percent_gen

do_mock = False
durable_log = True
fail_only = False

cur_common_base = (ConfigList()
  .override('root_path', ['/nvme/songxiaoniu/graph-learning/samgraph/'])
  .override('amp', [True])
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('epoch', [3])
  .override('empty_feat', [25])
  .override('num_worker', [8])
  .override('logdir', [
    'run-logs',
  ])
  .override('profile_level', [3])
  .override('log_level', ['warn'])
  .override('multi_gpu_sgnn', [True])
  .override('pipeline', [
    # True,
    False,
  ]))

cfg_list_collector = ConfigList.Empty()

'''
GraphSage
'''
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('app', [App.graphsage       ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen( 5, 25, 5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 5, 25, 5)).override('num_feat_dim_hack', [256]))

# # 1.3 unsup, mag 240 requires different batch
# # cur_common_base = (cur_common_base.copy().override('batch_size', [4000]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 6, 6, 2)))
# cur_common_base = (cur_common_base.copy().)
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 1, 2, 1) + percent_gen( 4, 10, 2)).override('batch_size', [1000, 2000]))

cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen( 1,  2, 1) + percent_gen( 4, 12, 2) + percent_gen(16,20,4) + percent_gen(25, 25, 5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 5, 25, 5)).override('num_feat_dim_hack', [256]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 1,  2, 1) + percent_gen( 4, 10, 2)).override('batch_size', [8000, 4000, 2000]))

cfg_list_collector.hyper_override(
  ['cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part_2, "DIRECT", ""],
    [CachePolicy.clique_part_2, "", "MPSPhase"],
    [CachePolicy.rep_2, "DIRECT", ""],
    [CachePolicy.rep_2, "", "MPSPhase"],
    [CachePolicy.coll_cache_asymm_link_2, "", "MPSPhase"],
  ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log, fail_only=fail_only)
