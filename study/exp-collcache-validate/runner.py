import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('epoch', [10])
  .override('num_sampler', [2])
  .override('num_trainer', [6])
  .override('logdir', [
    # 'run-logs-pipe',
    'run-logs',
  ])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('multi_gpu', [True])
  .override('cache_policy', [
    # CachePolicy.cache_by_random,
    # CachePolicy.cache_by_degree,
    # CachePolicy.cache_by_presample_1,
    # CachePolicy.coll_cache,
    CachePolicy.cache_by_presample_10,
    CachePolicy.coll_cache_10,
  ])
  .override('pipeline', [
    # True,
    False,
  ]))

cfg_list_collector = ConfigList.Empty()

# 16GB
# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(22, 22, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(20, 20, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(20, 20, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(11, 11, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(11, 11, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 3,  3, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [4]).override('num_trainer', [4]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(31, 31, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(24, 24, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(24, 24, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(16, 16, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(16, 16, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 3,  3, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(25, 25, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(21, 21, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(21, 21, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen( 9,  9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen( 9,  9, 1)).override('num_feat_dim_hack', [256]).override('num_sampler', [1]).override('num_trainer', [7]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 3,  3, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(24, 24, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(21, 21, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(21, 21, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(11, 11, 1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(11, 11, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen( 3,  3, 1)).override('num_sampler', [1]).override('num_trainer', [7]))


# 40GB
cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen( 100, 100, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(  76,  76, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(  58,  58, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(  46,  46, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(  40,  40, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(  25,  25, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(   9,   9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen( 100, 100, 1)).override('num_sampler', [4]).override('num_trainer', [4]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(  82,  82, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(  60,  60, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(  60,  60, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(  44,  44, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(  48,  48, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(   9,   9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen( 100, 100, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(  76,  76, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(  58,  58, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(  50,  50, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(  40,  40, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(  34,  34, 1)).override('num_feat_dim_hack', [256]).override('num_sampler', [1]).override('num_trainer', [7]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(   9,   9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen( 100, 100, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(  80,  80, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(  60,  60, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(  54,  54, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(  42,  42, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(  44,  44, 1)).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(   9,   9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))



# # full 80GB
# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(100,100,1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(96,96,1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(88,88,1)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [4]).override('num_trainer', [4]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(100,100,1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(100,100,2)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(100,100,1)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(94,94,1)).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(100,100,1)).override('num_feat_dim_hack', [256]).override('num_sampler', [1]).override('num_trainer', [7]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', percent_gen(100,100,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', percent_gen(100,100,1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', percent_gen(96,96,1)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', percent_gen(100,100,1)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', percent_gen(100,100,1)).override('num_sampler', [1]).override('num_trainer', [7]))


# cfg_list_collector.override('cache_policy', [
#   # CachePolicy.cache_by_random,
#   # CachePolicy.cache_by_degree,
#   # CachePolicy.cache_by_presample_1,
# ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)
