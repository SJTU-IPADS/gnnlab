from runner_helper import RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, run_in_list
import os

def tmp_call_back(cfg: RunConfig):
  os.system(f"grep -A 4 'average' \"{cfg.get_log_fname()}.log\"")

if __name__ == '__main__':
  from sys import argv
  do_mock = False
  durable_log = True
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False

  run_in_list(ConfigList()
    .select('app', [
      App.gcn,
      # App.graphsage,
      # App.pinsage,
    ]).select('dataset', [
      # Dataset.reddit,
      # Dataset.products,
      Dataset.papers100M,
      # Dataset.friendster,
    ]).select('cache_policy', [
      CachePolicy.no_cache,
      # CachePolicy.cache_by_degree,
      # CachePolicy.cache_by_heuristic,
      # CachePolicy.dynamic_cache,
    ]).select('pipeline', [
      False,
      # True,
    ])
    # .override_arch(Arch.arch0)
    .override('logdir', [TMP_LOG_DIR])
    .override('cache_policy', [
      # CachePolicy.cache_by_degree,
      CachePolicy.cache_by_heuristic,
      # CachePolicy.cache_by_presample,
      # CachePolicy.cache_by_degree_hop,
      # CachePolicy.cache_by_presample_static,
      # CachePolicy.cache_by_fake_optimal,
    ])
    .override('batch_size',[
      # 1000,
      8000,
    ])
    .override('cache_percent', [
      # 0.0,
      0.01,0.02,0.03,0.04,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,
      # 0.55, 0.60,
      # 1,
    ])
    .conf_list
    ,do_mock
    ,durable_log
    # , tmp_call_back
    )

