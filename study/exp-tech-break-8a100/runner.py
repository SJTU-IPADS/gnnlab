import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, SampleType, percent_gen, reverse_percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('root_path', ['/nvme/graph-learning-copy/samgraph/'])
  .override('amp', [True])
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

# '''
# GCN-only
# '''
# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
# # 1.1 unsup, large batch
# cur_common_base = (cur_common_base.copy().override('batch_size', [4000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={150*1024*1024}']))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 10,  50,  10) + percent_gen( 90,  100,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 10,  60,  10))) # percent_gen( 45,   60,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 10,  40,  10))) # percent_gen( 25,   40,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 10,  25,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 10,  30,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01, 0.02, 0.04, 0.05]).override('num_feat_dim_hack', [256]))

# # 1.2 unsup, small batch
# cur_common_base = (cur_common_base.copy().override('batch_size', [2000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}']))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 10,  50,  10) + percent_gen( 70, 100, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 10,  60,  10) + [0.65])) # percent_gen( 50,  65,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 10,  50,  10))) # percent_gen( 35,  50,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 10,  40,  10))) # percent_gen( 25,  40,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 10,  30,  10) + [0.35])) # percent_gen( 20,  35,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 10,  25,  5)).override('num_feat_dim_hack', [256]))

# # 1.3 unsup, mag 240 requires different batch
# cur_common_base = (cur_common_base.copy().override('batch_size', [500]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))
# cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))

# # 2.1 sup
# cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
# cur_common_base = (cur_common_base.copy().override('batch_size', [8000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}']))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 10,  50,  10) + percent_gen( 90,  100,  5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 10,  60,  10) + [0.65])) # percent_gen( 50,  65, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 10,  50,  10))) # percent_gen( 35,  50, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 10,  40,  10) + [0.45])) # percent_gen( 30,  45, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 10,  30,  10) + [0.35])) # percent_gen( 20,  35, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 10,  25, 5)).override('num_feat_dim_hack', [256]))

# # 2.2 sup, mag 240 requires different batch
# cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))
# cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 6, 2)))


'''
GraphSage
'''
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}']))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(5, 25, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(5, 25, 5)).override('num_feat_dim_hack', [256]))
# # 1.2 unsup, small batch
# cur_common_base = (cur_common_base.copy().override('batch_size', [4000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={150*1024*1024}']))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen( 5, 25, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 5, 25, 5)).override('num_feat_dim_hack', [256]))

# # # 1.3 unsup, mag 240 requires different batch
# # # cur_common_base = (cur_common_base.copy().override('batch_size', [4000]))
# # # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 6, 6, 2)))
# # cur_common_base = (cur_common_base.copy().)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 10, 2)).override('batch_size', [1000, 2000]))


cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]).override('custom_env', [f'SAMGRAPH_MQ_SIZE={55*1024*1024}']))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20] +  percent_gen(25, 25, 5)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(5, 25, 5)).override('num_feat_dim_hack', [256]))

# # cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 6, 6, 2)))
# # cur_common_base = (cur_common_base.copy().override('batch_size', [4000]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 6, 6, 2)))
# # cur_common_base = (cur_common_base.copy())
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 2, 10, 2)).override('batch_size', [8000, 4000, 2000]))
'''
Untested application
'''

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

# cfg_list_collector.override_T('cache_policy', [
#   CachePolicy.clique_part_2,
#   CachePolicy.rep_2,
#   CachePolicy.coll_cache_asymm_link_2,
#   # CachePolicy.clique_part_by_degree_2,
#   ])

cfg_list_collector.hyper_override(
  ['cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part_2, "DIRECT", ""],
    [CachePolicy.clique_part_2, "", "MPS"],
    [CachePolicy.rep_2, "DIRECT", ""],
    [CachePolicy.rep_2, "", "MPS"],
    [CachePolicy.coll_cache_asymm_link_2, "", "MPS"],
    [CachePolicy.coll_cache_asymm_link_2, "", "MPSPhase"],
  ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)