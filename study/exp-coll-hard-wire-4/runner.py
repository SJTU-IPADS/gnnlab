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

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
# cur_common_base.override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}'])
# cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 16, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 8, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 12, 2)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', []).override('num_feat_dim_hack', [256]))
# # # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (  2, 10,  2)
# cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 20, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 6, 2)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (  2, 10,  2)

# cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 20, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 10, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 14, 2)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (   2,  10, 2)

# cur_common_base = (cur_common_base.copy().override('batch_size', [4000]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 8, 2)).override('num_feat_dim_hack', [256]))


cur_common_base = (cur_common_base.copy().override('app', [App.graphsage       ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
cur_common_base.override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}'])
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 24, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 22, 2)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 20, 2)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 14, 4) + [0.15]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 14, 4) + [0.15]).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (  2, 10,  2)
# cur_common_base = (cur_common_base.copy().override('batch_size', [1000]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 22, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 22, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 16, 2)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 18, 2)).override('num_feat_dim_hack', [256]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (  2, 10,  2)

cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 22, 2)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 20, 4)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 14, 4) + [0.15]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + [0.05] + percent_gen( 6, 14, 4) + [0.15]).override('num_feat_dim_hack', [256]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.01] + percent_gen( 1, 1, 1)))     # (   2,  10, 2)

# cur_common_base = (cur_common_base.copy().override('batch_size', [4000]))
# cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# # cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', [0.01] + percent_gen( 5, 25, 10)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.01] + percent_gen( 5, 15, 10) + percent_gen(16, 18, 2)).override('num_feat_dim_hack', [256]))


# cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # ( 100, 100, 1)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  72,  78, 2) x 1000 x2000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  54,  60, 2) x1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  48,  54, 2)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  36,  42, 2) x1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256])) # (  34,  40, 2) x1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))     # (   2,  10, 2)

# cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # ( 100, 100, 1)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1))) # 8000+60, up?                       # (  58,  64, 2) x1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  40,  46, 2) x all
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  12,  18, 2)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  24,  30, 2) x all
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256])) # (   1,   4, 1)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))     # (   2,  10, 2)

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,         ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # ( 100, 100, 1)
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,          ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  58,  64, 2) x all
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  48,  54, 2) x 1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  38,  44, 2) x 1000
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))                                      # (  28,  34, 2) x 1 2 4
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)).override('num_feat_dim_hack', [256])) # (  18,  24, 2) x 1 2 4
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', reverse_percent_gen( 1, 1, 1)))     # (   2,  10, 2) x 1 2

cfg_list_collector.override_T('cache_policy', [
  CachePolicy.clique_part_2,
  CachePolicy.rep_2,
  CachePolicy.coll_cache_asymm_link_2,
  # CachePolicy.clique_part_by_degree_2,
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
