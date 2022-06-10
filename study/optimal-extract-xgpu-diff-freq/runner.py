import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, run_in_list, SampleType, percent_gen

do_mock = False
durable_log = True

def copy_optimal(cfg: RunConfig):
  os.system(f"rm -f \"{cfg.get_log_fname()}*\"")
  os.system(f"mv node_access_optimal_cache_freq_bin* \"{cfg.get_log_fname()}_optimal_cache_freq_bin.bin\"")
  os.system(f"mv node_access_optimal_cache_bin*      \"{cfg.get_log_fname()}_optimal_cache_bin.bin\"")
  os.system(f"mv node_access_optimal_cache_hit*      \"{cfg.get_log_fname()}_optimal_cache_hit.txt\"")
  os.system(f"mv node_access_frequency*              \"{cfg.get_log_fname()}_frequency_histogram.txt\"")

cur_common_base = (ConfigList()
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('pipeline', [False])
  .override('epoch', [10])
  .override('logdir', ['run-logs',])
  .override('profile_level', [3])
  .override('log_level', ['error'])
  .override('arch', [Arch.arch3])
  .override('multi_gpu', [False])
  .override('report_optimal', [1])
  .override('cache_percent', [0])
  .override('num_hidden', [2])
  .override('cache_policy', [CachePolicy.cache_by_presample_1])
)

cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part1-0']).override('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=0/1']).override('part_num', [1]).override('part_idx', [0]))

# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part2-0']).override('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=0/2']).override('part_num', [2]).override('part_idx', [0]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part2-1']).override('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=1/2']).override('part_num', [2]).override('part_idx', [1]))

# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part4-0']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0/4']).override('part_num', [4]).override('part_idx', [0]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part4-1']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=1/4']).override('part_num', [4]).override('part_idx', [1]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part4-2']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=2/4']).override('part_num', [4]).override('part_idx', [2]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part4-3']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=3/4']).override('part_num', [4]).override('part_idx', [3]))

# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-0']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=0/8']).override('part_num', [8]).override('part_idx', [0]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-1']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=1/8']).override('part_num', [8]).override('part_idx', [1]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-2']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=2/8']).override('part_num', [8]).override('part_idx', [2]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-3']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=3/8']).override('part_num', [8]).override('part_idx', [3]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-4']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=4/8']).override('part_num', [8]).override('part_idx', [4]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-5']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=5/8']).override('part_num', [8]).override('part_idx', [5]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-6']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=6/8']).override('part_num', [8]).override('part_idx', [6]))
# cfg_list_collector.concat(cur_common_base.copy().override('logdir', ['run-logs/part8-7']).override('custom_env', ['export SAMGRAPH_FAKE_FEAT_DIM=1; export SAMGRAPH_TRAIN_SET_PART=7/8']).override('part_num', [8]).override('part_idx', [7]))

cur_common_base = cfg_list_collector
cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.graphsage ]).override('sample_type', [SampleType.kKHop2]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.pinsage   ]).override('sample_type', [SampleType.kRandomWalk]))
cfg_list_collector.concat(cur_common_base.copy().override('app', [App.gcn       ]).override('sample_type', [SampleType.kWeightedKHopPrefix]))

cfg_list_collector.override('dataset', [
  Dataset.twitter,
  Dataset.papers100M,
  Dataset.uk_2006_05,
])


if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(cfg_list_collector.conf_list, do_mock, durable_log, copy_optimal)
