import os
import datetime
from enum import Enum

LOG_DIR='run-logs/logs_samgraph_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TMP_LOG_FILE = 'run-logs/tmp.log'

root_path="/graph-learning/samgraph/"

DO_MOCK=False
TMP_LOG=True
DURABLE_LOG=True

class CachePolicy(Enum):
  cache_by_degree = 0
  cache_by_heuristic = 1
  dynamic_cache = 2
  no_cache = 3

class Arch(Enum):
  arch0 = 0
  arch1 = 1
  arch2 = 2
  arch3 = 3
  arch4 = 4


class App(Enum):
  gcn = 0
  graphsage = 1
  pinsage = 2


class Dataset(Enum):
  reddit = 0
  products = 1
  papers100M = 2
  friendster = 3
  def __str__(self):
    if self is Dataset.friendster:
      return 'com-friendster'
    return self.name

class RunConfig:
  def __init__(self, app:App, dataset:Dataset, 
               cache_policy:CachePolicy=CachePolicy.no_cache, cache_percent:float=0.0, 
               pipeline:bool=False,
               max_sampling_jobs=10,
               max_copying_jobs=2,
               arch:Arch=Arch.arch4,
               epoch:int=3, logdir:str=LOG_DIR):
    self.app           = app
    self.dataset       = dataset
    self.cache_policy  = cache_policy
    self.cache_percent = cache_percent
    self.pipeline      = pipeline
    self.sample_job    = max_sampling_jobs
    self.copy_job      = max_copying_jobs
    self.arch          = arch
    self.epoch         = epoch
    self.logdir        = logdir

  def cache_log_name(self):
    if self.cache_policy is CachePolicy.no_cache:
      return []
    return ["cache"]

  def pipe_log_name(self):
    if self.pipeline:
      return ["pipeline"]
    return []

  def form_cmd(self):
    cmd_line = ''
    cmd_line += 'export SAMGRAPH_PROFILE_LEVEL=1; '
    cmd_line += 'export SAMGRAPH_LOG_LEVEL=warn; '
    cmd_line += f'python samgraph/train_{self.app.name}.py --arch {self.arch.name}'
    cmd_line += f' --max-sampling-jobs {self.sample_job}'
    cmd_line += f' --max-copying-jobs {self.copy_job}'
    if self.pipeline:
      cmd_line += ' --pipeline'

    if self.cache_policy is not CachePolicy.no_cache:
      cmd_line += f' --cache-policy {self.cache_policy.value} --cache-percentage {self.cache_percent}'
    else:
      cmd_line += f' --cache-percentage 0.0'

    
    cmd_line += f' --dataset-path {root_path}{str(self.dataset)}'
    cmd_line += f' --num-epoch {self.epoch}'
    if TMP_LOG:
      cmd_line += f' > \"{TMP_LOG_FILE}\" 2>&1'
    elif DURABLE_LOG:
      cmd_line += f' > \"{LOG_DIR}/'
      cmd_line += '_'.join(['samgraph']+self.cache_log_name() + self.pipe_log_name()+[self.app.name, str(self.dataset)]) 
      cmd_line += '.log\" 2>&1'
    return cmd_line

  def beauty(self):
    msg = ' '.join(['Running '] + self.cache_log_name() + self.pipe_log_name() + [self.app.name, self.dataset.name])
    return msg
    
  def run(self, mock=False):
    if mock:
      print(self.form_cmd())
    else:
      print(self.beauty())
      os.system('mkdir -p {}'.format(LOG_DIR))
      os.system(self.form_cmd())
    pass

def run_in_list(conf_list : list):
  for conf in conf_list:
    conf : RunConfig
    conf.run(DO_MOCK)

class ConfigList:
  def __init__(self):
    self.conf_list = [
      RunConfig(App.gcn,       Dataset.reddit     ),
      RunConfig(App.gcn,       Dataset.products   ),
      RunConfig(App.gcn,       Dataset.papers100M ),
      RunConfig(App.gcn,       Dataset.friendster ),

      RunConfig(App.graphsage, Dataset.reddit     ),
      RunConfig(App.graphsage, Dataset.products   ),
      RunConfig(App.graphsage, Dataset.papers100M ),
      RunConfig(App.graphsage, Dataset.friendster ),

      RunConfig(App.pinsage,   Dataset.reddit     ),
      RunConfig(App.pinsage,   Dataset.products   ),
      RunConfig(App.pinsage,   Dataset.papers100M ),
      RunConfig(App.pinsage,   Dataset.friendster ),

      RunConfig(App.gcn,       Dataset.reddit,      pipeline=True),
      RunConfig(App.gcn,       Dataset.products,    pipeline=True),
      RunConfig(App.gcn,       Dataset.papers100M,  pipeline=True),
      RunConfig(App.gcn,       Dataset.friendster,  pipeline=True),

      RunConfig(App.graphsage, Dataset.reddit,      pipeline=True),
      RunConfig(App.graphsage, Dataset.products,    pipeline=True),
      RunConfig(App.graphsage, Dataset.papers100M,  pipeline=True),
      RunConfig(App.graphsage, Dataset.friendster,  pipeline=True),

      RunConfig(App.pinsage,   Dataset.reddit,      pipeline=True),
      RunConfig(App.pinsage,   Dataset.products,    pipeline=True),
      RunConfig(App.pinsage,   Dataset.papers100M,  pipeline=True),
      RunConfig(App.pinsage,   Dataset.friendster,  pipeline=True),

      RunConfig(App.gcn,       Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.gcn,       Dataset.products,   cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.gcn,       Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic,  cache_percent=0.25),
      RunConfig(App.gcn,       Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.25),

      RunConfig(App.graphsage, Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.graphsage, Dataset.products,   cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.graphsage, Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic,  cache_percent=0.25),
      RunConfig(App.graphsage, Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.25),

      RunConfig(App.pinsage,   Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.pinsage,   Dataset.products,   cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.98),
      RunConfig(App.pinsage,   Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic,  cache_percent=0.25),
      RunConfig(App.pinsage,   Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,     cache_percent=0.25),

      RunConfig(App.gcn,       Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.gcn,       Dataset.products,   cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.gcn,       Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.gcn,       Dataset.friendster, cache_policy=CachePolicy.dynamic_cache),

      RunConfig(App.graphsage, Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.graphsage, Dataset.products,   cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.graphsage, Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.graphsage, Dataset.friendster, cache_policy=CachePolicy.dynamic_cache),

      RunConfig(App.pinsage,   Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.pinsage,   Dataset.products,   cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.pinsage,   Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache),
      RunConfig(App.pinsage,   Dataset.friendster, cache_policy=CachePolicy.dynamic_cache),

      RunConfig(App.gcn,       Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True, max_copying_jobs=2),
      RunConfig(App.gcn,       Dataset.products,   cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True, max_copying_jobs=2),
      RunConfig(App.gcn,       Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic, cache_percent=0.25, pipeline=True), # 2
      RunConfig(App.gcn,       Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,    cache_percent=0.25, pipeline=True), # 2

      RunConfig(App.graphsage, Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True), # 2
      RunConfig(App.graphsage, Dataset.products,   cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True), # 2
      RunConfig(App.graphsage, Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic, cache_percent=0.4,  pipeline=True), # 2
      RunConfig(App.graphsage, Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,    cache_percent=0.4,  pipeline=True, max_copying_jobs=2),

      RunConfig(App.pinsage,   Dataset.reddit,     cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True),
      RunConfig(App.pinsage,   Dataset.products,   cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True),
      RunConfig(App.pinsage,   Dataset.papers100M, cache_policy=CachePolicy.cache_by_heuristic, cache_percent=0.4,  pipeline=True),
      RunConfig(App.pinsage,   Dataset.friendster, cache_policy=CachePolicy.cache_by_degree,    cache_percent=0.4,  pipeline=True),

      RunConfig(App.gcn,       Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.gcn,       Dataset.products,   cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.gcn,       Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.gcn,       Dataset.friendster, cache_policy=CachePolicy.dynamic_cache, pipeline=True),

      RunConfig(App.graphsage, Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.graphsage, Dataset.products,   cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.graphsage, Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.graphsage, Dataset.friendster, cache_policy=CachePolicy.dynamic_cache, pipeline=True),

      RunConfig(App.pinsage,   Dataset.reddit,     cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.pinsage,   Dataset.products,   cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.pinsage,   Dataset.papers100M, cache_policy=CachePolicy.dynamic_cache, pipeline=True),
      RunConfig(App.pinsage,   Dataset.friendster, cache_policy=CachePolicy.dynamic_cache, pipeline=True),
    ]
  
  def select_app(self, app_indicator):
    newlist = []
    for cfg in self.conf_list:
      if cfg.app in app_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self
  def select_dataset(self, ds_indicator):
    newlist = []
    for cfg in self.conf_list:
      if cfg.dataset in ds_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self
  def select_cache(self, cache_indicator):
    newlist = []
    for cfg in self.conf_list:
      if cfg.cache_policy in cache_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self
  def select_pipe(self, pipe_indicator):
    newlist = []
    for cfg in self.conf_list:
      if cfg.pipeline in pipe_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

run_in_list(ConfigList()
  .select_app([
    App.gcn,
    # App.graphsage,
    # App.pinsage,
  ]).select_dataset([
    Dataset.reddit,
    # Dataset.products,
    # Dataset.papers100M,
    # Dataset.friendster,
  ]).select_cache([
    # CachePolicy.no_cache,
    # CachePolicy.cache_by_degree,
    # CachePolicy.cache_by_heuristic,
    CachePolicy.dynamic_cache,
  ]).select_pipe([
    # False,
    True,
  ]).conf_list)

