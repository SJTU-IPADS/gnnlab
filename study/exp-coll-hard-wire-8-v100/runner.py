import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, SampleType, percent_gen, reverse_percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('copy_job', [1])
  .override('sample_job', [1])
  # .override('epoch', [10])
  .override('epoch', [3])
  .override('num_sampler', [2])
  .override('num_trainer', [6])
  .override('logdir', [
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

'''
GCN-only
'''
cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('batch_size', [4000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={150*1024*1024}']))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 20, 100, 10)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  60,  5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  40,  5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  25,  5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  30,  5)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2)).override('num_feat_dim_hack', [256]))

# 1.2 unsup, small batch
cur_common_base = (cur_common_base.copy().override('batch_size', [2000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}']))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 20, 100, 10)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  55,  10) + [0.60]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  45,  10) + []))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  35,  10) + []))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  25,  10) + [0.30]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  15,  10) + [0.20]).override('num_feat_dim_hack', [256]))

# 1.3 unsup, mag 240 requires different batch
cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [] + percent_gen( 1, 6, 1)))

# 2.1 sup
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}']))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 20, 100, 10)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  65, 10) + []))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  45, 10) + [0.50]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  35, 10) + [0.40]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  25, 10) + [0.30]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0] + percent_gen( 1, 3, 1) + percent_gen( 4, 10, 2) + percent_gen( 15,  15, 10) + [0.20]).override('num_feat_dim_hack', [256]))

# 2.2 sup, mag 240 requires different batch
cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0] + percent_gen( 1, 6, 1)))
cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0] + percent_gen( 1, 6, 1)))


'''
Untested application
'''
# cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))

# cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1))) # 8000+60, up?
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))

cfg_list_collector.override_T('cache_policy', [
  CachePolicy.clique_part_2,
  CachePolicy.rep_2,
  CachePolicy.coll_cache_asymm_link_2,
  ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)