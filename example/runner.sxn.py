import os
import datetime
from enum import Enum
import copy

LOG_DIR='run-logs/logs_samgraph_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TMP_LOG_DIR='run-logs/tmp_log_dir'

root_path="/graph-learning/samgraph/"

DO_MOCK=False
DURABLE_LOG=True

class CachePolicy(Enum):
  cache_by_degree = 0
  cache_by_heuristic = 1
  cache_by_presample = 2
  cache_by_degree_hop = 3
  cache_by_presample_static = 4
  cache_by_fake_optimal = 5
  dynamic_cache = 6

  cache_by_presample_1 = 11
  cache_by_presample_2 = 12
  cache_by_presample_3 = 13
  cache_by_presample_max = 14
  no_cache = 20
  def get_samgraph_policy_value(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return CachePolicy.cache_by_presample.value
    return self.value
  def get_presample_epoch(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return self.value - CachePolicy.cache_by_presample_1.value + 1
    return 1
  def get_log_fname(self):
    if self is CachePolicy.cache_by_presample:
      return CachePolicy.cache_by_presample_1.name
    return self.name

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
               arch:Arch=Arch.arch3,
               epoch:int=3,
               batch_size:int=8000,
               logdir:str=LOG_DIR):
    self.app           = app
    self.dataset       = dataset
    self.cache_policy  = cache_policy
    self.cache_percent = cache_percent
    self.pipeline      = pipeline
    self.sample_job    = max_sampling_jobs
    self.copy_job      = max_copying_jobs
    self.arch          = arch
    self.epoch         = epoch
    self.batch_size    = batch_size
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
    cmd_line += f'export SAMGRAPH_PRESAMPLE_EPOCH={self.cache_policy.get_presample_epoch()}; '
    cmd_line += 'export SAMGRAPH_LOG_NODE_ACCESS=0; '
    cmd_line += 'export SAMGRAPH_LOG_NODE_ACCESS_SIMPLE=0; '
    cmd_line += 'export SAMGRAPH_PROFILE_LEVEL=1; '
    cmd_line += 'export SAMGRAPH_BARRIER_EPOCH=1; '
    cmd_line += 'export SAMGRAPH_LOG_LEVEL=warn; '
    cmd_line += 'export SAMGRAPH_DUMP_TRACE=0; '
    cmd_line += f'python samgraph/train_{self.app.name}.py --arch {self.arch.name}'
    cmd_line += f' --max-sampling-jobs {self.sample_job}'
    cmd_line += f' --max-copying-jobs {self.copy_job}'
    if self.pipeline:
      cmd_line += ' --pipeline'

    if self.cache_policy is not CachePolicy.no_cache:
      cmd_line += f' --cache-policy {self.cache_policy.get_samgraph_policy_value()} --cache-percentage {self.cache_percent}'
    else:
      cmd_line += f' --cache-percentage 0.0'

    
    cmd_line += f' --dataset-path {root_path}{str(self.dataset)}'
    cmd_line += f' --num-epoch {self.epoch}'
    cmd_line += f' --batch-size {self.batch_size}'

    if DURABLE_LOG:
      std_out_log = self.get_log_fname() + '.log'
      std_err_log = self.get_log_fname() + '.err.log'
      cmd_line += f' > \"{std_out_log}\"'
      cmd_line += f' 2> \"{std_err_log}\"'
      cmd_line += ';'
      cmd_line += ' sed -i \"/^Using.*/d\" \"' + std_err_log + "\"" 
    return cmd_line
  
  def get_log_fname(self):
    std_out_log = f'{self.logdir}/'
    std_out_log += '_'.join(
      ['samgraph']+self.cache_log_name() + self.pipe_log_name() +
      [self.app.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache_rate_{int(self.cache_percent*100):0>3}', f'batch_size_{self.batch_size}']) 
    return std_out_log

  def beauty(self):
    msg = ' '.join(
      ['Running '] + self.cache_log_name() + self.pipe_log_name() + 
      [self.app.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache rate:{int(self.cache_percent*100):0>3}%', f'batch size:{self.batch_size}', ])
    return msg
    
  def run(self, mock=False, callback = None):
    if mock:
      print(self.form_cmd())
    else:
      print(self.beauty())
      os.system('mkdir -p {}'.format(self.logdir))
      os.system(self.form_cmd())
      if callback != None:
        callback(self)
    pass

def run_in_list(conf_list : list, callback = None):
  for conf in conf_list:
    conf : RunConfig
    conf.run(DO_MOCK, callback)

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

  def select(self, key, val_indicator):
    '''
    filter config list by key and list of value
    available key: app, dataset, cache_policy, pipeline
    '''
    newlist = []
    for cfg in self.conf_list:
      if getattr(cfg, key) in val_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def override_arch(self, arch):
    '''
    override all arch in config list by arch
    '''
    for cfg in self.conf_list:
      cfg.arch = arch
    return self

  def override(self, key, val_list):
    '''
    override config list by key and value.
    if len(val_list)>1, then config list is extended, example:
       [cfg1(batch_size=4000)].override('batch_size',[1000,8000]) 
    => [cfg1(batch_size=1000),cfg1(batch_size=8000)]
    available key: arch, logdir, cache_percent, cache_policy, batch_size
    '''
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for val in val_list:
      new_list = copy.deepcopy(orig_list)
      for cfg in new_list:
        setattr(cfg, key, val)
      self.conf_list += new_list
    return self

def tmp_call_back(cfg: RunConfig):
  os.system(f"grep -A 4 'average' \"{cfg.get_log_fname()}.log\"")

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      DO_MOCK = True
    elif arg == '-i' or arg == '--interactive':
      DURABLE_LOG = False

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
    # , tmp_call_back
    )

