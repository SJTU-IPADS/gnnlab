import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('app', [App.gcn,])
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('cache_percent', [0])
  .override('epoch', [3])
  .override('pipeline', [False,])
  .override('logdir', ['run-logs',])
  .override('cache_policy', [CachePolicy.cache_by_presample_1,])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('multi_gpu', [True]))

cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.products,]).override('cache_percent', percent_gen(100, 100, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.twitter,]).override('cache_percent', percent_gen(24, 24, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M,]).override('cache_percent', percent_gen(20, 20, 1)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.uk_2006_05,]).override('cache_percent', percent_gen(13, 13, 1)))

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(cfg_list_collector.conf_list, do_mock, durable_log)

