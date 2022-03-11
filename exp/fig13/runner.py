import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('pipeline', [True])
  .override('epoch', [3])
  .override('num_sampler', [2])
  .override('num_trainer', [6])
  .override('logdir', ['run-logs',])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('multi_gpu', [True])
  .override('cache_policy', [
    # CachePolicy.cache_by_random,
    CachePolicy.cache_by_degree,
    # CachePolicy.cache_by_presample_1,
  ]))

cfg_list_collector = ConfigList.Empty()
# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(21, 21, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(19, 19, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(10, 10, 1)))

# cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(29, 29, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(24, 24, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(15, 15, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(23, 23, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen( 7,  7, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

# cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(24, 24, 1)))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)))

cur_common_base = (cur_common_base.copy().override('cache_policy', [CachePolicy.cache_by_presample_1]))
cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(22, 22, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(20, 20, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(11, 11, 1)))

cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(31, 31, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(16, 16, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(25, 25, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen( 9,  9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)))

cur_common_base = (cur_common_base.copy().override('cache_policy', [CachePolicy.cache_by_degree]))
cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(22, 22, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(20, 20, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(11, 11, 1)))

cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(31, 31, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(16, 16, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(25, 25, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen( 9,  9, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)))

cur_common_base = (cur_common_base.copy().override('cache_policy', [CachePolicy.cache_by_random]))
cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(21, 21, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(18, 18, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(10, 10, 1)))

cur_common_base = (cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(29, 29, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen(15, 15, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(20, 20, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(20, 20, 1)).override('num_sampler', [1]).override('num_trainer', [7]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05, ]).override('cache_percent', percent_gen( 7,  7, 1)).override('num_sampler', [1]).override('num_trainer', [7]))

cur_common_base = (cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,    ]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M, ]).override('cache_percent', percent_gen(21, 21, 1)))

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)
