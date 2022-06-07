"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os
import datetime
from enum import Enum
import copy

LOG_DIR='run-logs/logs_samgraph_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TMP_LOG_DIR='run-logs/tmp_log_dir'

def percent_gen(lb, ub, gap=1):
  ret = []
  i = lb
  while i <= ub:
    ret.append(i/100)
    i += gap
  return ret

class System(Enum):
  samgraph = 0
  dgl = 1

class CachePolicy(Enum):
  cache_by_degree = 0
  cache_by_heuristic = 1
  cache_by_presample = 2
  cache_by_degree_hop = 3
  cache_by_presample_static = 4
  cache_by_fake_optimal = 5
  dynamic_cache = 6
  cache_by_random = 7
  coll_cache = 8

  cache_by_presample_1 = 11
  cache_by_presample_2 = 12
  cache_by_presample_3 = 13
  cache_by_presample_4 = 14
  cache_by_presample_5 = 15
  cache_by_presample_6 = 16
  cache_by_presample_7 = 17
  cache_by_presample_8 = 18
  cache_by_presample_9 = 19
  cache_by_presample_10 = 20
  cache_by_presample_max = 21

  coll_cache_1 = 31
  coll_cache_2 = 32
  coll_cache_3 = 33
  coll_cache_4 = 34
  coll_cache_5 = 35
  coll_cache_6 = 36
  coll_cache_7 = 37
  coll_cache_8 = 38
  coll_cache_9 = 39
  coll_cache_10 = 40
  coll_cache_max = 41

  no_cache = 50
  def get_samgraph_policy_param_name(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return "pre_sample"
    if self.value in range(CachePolicy.coll_cache_1.value, CachePolicy.coll_cache_max.value):
      return "coll_cache"
    name_list = [
      'degree',
      'heuristic',
      'pre_sample',
      'degree_hop',
      'presample_static',
      'fake_optimal',
      'dynamic_cache',
      'random',
      'coll_cache',
    ]
    return name_list[self.value]
  def get_presample_epoch(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return self.value - CachePolicy.cache_by_presample_1.value + 1
    if self.value in range(CachePolicy.coll_cache_1.value, CachePolicy.coll_cache_max.value):
      return self.value - CachePolicy.coll_cache_1.value + 1
    return 1
  def get_log_fname(self):
    if self is CachePolicy.cache_by_presample:
      return CachePolicy.cache_by_presample_1.name
    if self is CachePolicy.coll_cache:
      return CachePolicy.coll_cache_1.name
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

class SampleType(Enum):
  kKHop0 = 0
  kKHop1 = 1
  kWeightedKHop = 2 # should be deprecated
  kRandomWalk = 3
  kWeightedKHopPrefix = 4
  kKHop2 = 5
  kWeightedKHopHashDedup = 6
  kSaint = 7

  kDefaultForApp = 10
  def __str__(self):
    name_list = [
      "khop0",
      "khop1",
      "weighted_khop",
      "random_walk",
      "weighted_khop_prefix",
      "khop2",
      "weighted_khop_hash_dedup",
      "saint"
    ]
    return name_list[self.value]

class Dataset(Enum):
  reddit = 0
  products = 1
  papers100M = 2
  friendster = 3
  uk_2006_05 = 5
  twitter = 6

  papers100M_undir = 7
  mag240m_homo = 8

  def __str__(self):
    if self is Dataset.friendster:
      return 'com-friendster'
    elif self is Dataset.uk_2006_05:
      return 'uk-2006-05'
    elif self is Dataset.papers100M_undir:
      return 'papers100M-undir'
    elif self is Dataset.mag240m_homo:
      return 'mag240m-homo'
    return self.name
  def FeatGB(self):
    return [0.522,0.912,52.96,34.22, None ,74.14,39.72, 52.96,349.27][self.value]

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
    self.sample_type   = SampleType.kDefaultForApp
    self.no_torch = False
    self.report_optimal = 0
    self.profile_level = 1
    self.multi_gpu = False
    self.multi_gpu_sgnn = False
    self.empty_feat = 0
    self.log_level = 'warn'
    self.custom_arg = ''
    self.custom_env = ''
    self.async_train = False
    self.num_sampler = 1
    self.num_trainer = 1
    self.num_feat_dim_hack = None
    self.num_hidden = None
    self.lr = None
    self.cuda_launch_blocking = 0
    self.dump_trace = 0
    self.system = System.samgraph
    self.dgl_gpu_sampling = True
    self.nv_prof = False
    self.mps_mode = None
    self.root_path="/graph-learning/samgraph/"

  def cache_log_name(self):
    if self.cache_policy is CachePolicy.no_cache:
      return []
    return ["cache"]

  def pipe_log_name(self):
    if self.pipeline:
      return ["pipeline"]
    return []
  def preprocess_sample_type(self):
    if self.sample_type is SampleType.kDefaultForApp:
      if self.app is App.pinsage:
        self.sample_type = SampleType.kRandomWalk
      else:
        self.sample_type = SampleType.kKHop2
    else:
      return

  def form_cmd(self, durable_log=True):
    if self.system is System.samgraph:
      return self.form_samgraph_cmd(durable_log)
    elif self.system is System.dgl:
      return self.form_dgl_cmd(durable_log)
    else:
      assert(False)

  def form_dgl_cmd(self, durable_log=True):
    assert(self.system is System.dgl)
    cmd_line = ''
    cmd_line += f'export CUDA_LAUNCH_BLOCKING={self.cuda_launch_blocking}; '
    cmd_line += f'{self.custom_env} ;'
    if self.multi_gpu:
      if self.async_train:
        cmd_line += f'python dgl/multi_gpu/async/train_{self.app.name}.py'
      else:
        cmd_line += f'python dgl/multi_gpu/train_{self.app.name}.py'
    else:
      cmd_line += f'python dgl/train_{self.app.name}.py'

    if self.dgl_gpu_sampling:
      cmd_line += ' --use-gpu-sampling'
    else:
      cmd_line += ' --no-use-gpu-sampling'

    if self.multi_gpu:
      # cmd_line += f' --num-sample-worker {self.num_sampler} '
      cmd_line += ' --devices ' + ' '.join([str(dev_id) for dev_id in range(self.num_trainer)])

    if self.pipeline:
      cmd_line += ' --pipelining'
    else:
      cmd_line += ' --no-pipelining'

    cmd_line += f' --root-path "{self.root_path}"'
    cmd_line += f' --dataset {str(self.dataset)}'
    cmd_line += f' --num-epoch {self.epoch}'
    cmd_line += f' --batch-size {self.batch_size}'
    if self.lr is not None:
      cmd_line += f' --lr {self.lr}'

    cmd_line += f' {self.custom_arg} '

    if durable_log:
      std_out_log = self.get_log_fname() + '.log'
      std_err_log = self.get_log_fname() + '.err.log'
      cmd_line += f' > \"{std_out_log}\"'
      cmd_line += f' 2> \"{std_err_log}\"'
      cmd_line += ';'
      cmd_line += ' sed -i \"/^Using.*/d\" \"' + std_err_log + "\"" 
    return cmd_line

  def form_samgraph_cmd(self, durable_log=True):
    assert(self.system is System.samgraph)
    self.preprocess_sample_type()
    cmd_line = ''
    cmd_line += f'CUDA_LAUNCH_BLOCKING={self.cuda_launch_blocking} '
    cmd_line += 'SAMGRAPH_LOG_NODE_ACCESS=0 '
    cmd_line += f'SAMGRAPH_LOG_NODE_ACCESS_SIMPLE={self.report_optimal} '
    cmd_line += f'SAMGRAPH_DUMP_TRACE={self.dump_trace} '
    if self.num_feat_dim_hack != None:
      cmd_line += f'SAMGRAPH_FAKE_FEAT_DIM={self.num_feat_dim_hack} '
    if self.custom_env != '':
      cmd_line += f'{self.custom_env} '
    if self.multi_gpu:
      if self.async_train:
        cmd_line += f'python ../../example/samgraph/multi_gpu/async/train_{self.app.name}.py'
      elif self.mps_mode != None:
        cmd_line += f'python ../../example/samgraph/multi_gpu/mps/train_{self.app.name}.py'
      else:
        cmd_line += f'python ../../example/samgraph/multi_gpu/train_{self.app.name}.py'
    elif self.multi_gpu_sgnn:
      cmd_line += f'python ../../example/samgraph/sgnn/train_{self.app.name}.py'
    else:
      cmd_line += f'python ../../example/samgraph/train_{self.app.name}.py --arch {self.arch.name}'

    cmd_line += f' --sample-type {str(self.sample_type)}'
    cmd_line += f' --max-sampling-jobs {self.sample_job}'
    cmd_line += f' --max-copying-jobs {self.copy_job}'

    if self.multi_gpu:
      cmd_line += f' --num-sample-worker {self.num_sampler} '
      cmd_line += f' --num-train-worker {self.num_trainer} '
      if self.mps_mode != None:
        cmd_line += f' --mps-mode {self.mps_mode} '

    if self.pipeline:
      cmd_line += ' --pipeline'
    else:
      cmd_line += ' --no-pipeline'

    if self.cache_policy is not CachePolicy.no_cache:
      cmd_line += f' --cache-policy {self.cache_policy.get_samgraph_policy_param_name()} --cache-percentage {self.cache_percent}'
    else:
      cmd_line += f' --cache-percentage 0.0'
    
    cmd_line += f' --presample-epoch {self.cache_policy.get_presample_epoch()}'
    cmd_line += f' --barriered-epoch 1'

    cmd_line += f' --root-path "{self.root_path}"'
    cmd_line += f' --dataset {str(self.dataset)}'
    cmd_line += f' --num-epoch {self.epoch}'
    cmd_line += f' --batch-size {self.batch_size}'
    if self.num_hidden is not None:
      cmd_line += f' --num-hidden {self.num_hidden}'
    if self.lr is not None:
      cmd_line += f' --lr {self.lr}'

    cmd_line += f' --profile-level {self.profile_level}'
    cmd_line += f' --log-level {self.log_level}'
    cmd_line += f' --empty-feat {self.empty_feat}'
    cmd_line += f' {self.custom_arg} '

    if durable_log:
      std_out_log = self.get_log_fname() + '.log'
      std_err_log = self.get_log_fname() + '.err.log'
      cmd_line += f' > \"{std_out_log}\"'
      cmd_line += f' 2> \"{std_err_log}\"'
      cmd_line += ';'
    return cmd_line
  
  def get_log_fname(self):
    self.preprocess_sample_type()
    std_out_log = f'{self.logdir}/'
    if self.report_optimal == 1:
      std_out_log += "report_optimal_"
    std_out_log += '_'.join(
      [self.system.name]+self.cache_log_name() + self.pipe_log_name() +
      [self.app.name, self.sample_type.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache_rate_{int(self.cache_percent*100):0>3}', f'batch_size_{self.batch_size}']) 
    return std_out_log

  def beauty(self):
    self.preprocess_sample_type()
    msg = ' '.join(
      ['Running', self.system.name] + self.cache_log_name() + self.pipe_log_name() + 
      [self.app.name, self.sample_type.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache rate:{int(self.cache_percent*100):0>3}%', f'batch size:{self.batch_size}', ])
    return msg
    
  def run(self, mock=False, durable_log=True, callback = None):
    if mock:
      print(self.form_cmd(durable_log))
    else:
      print(self.beauty())
      if durable_log:
        os.system('mkdir -p {}'.format(self.logdir))
      os.system(self.form_cmd(durable_log))
      if callback != None:
        callback(self)
    pass

def run_in_list(conf_list : list, mock=False, durable_log=True, callback = None):
  for conf in conf_list:
    conf : RunConfig
    conf.run(mock, durable_log, callback)

class ConfigList:
  def __init__(self):
    self.conf_list = [
      RunConfig(App.gcn,       Dataset.products,     cache_policy=CachePolicy.cache_by_degree,    cache_percent=1.0,  pipeline=True, max_copying_jobs=2),
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
  def concat(self, another_list):
    self.conf_list += copy.deepcopy(another_list.conf_list)
    return self
  def copy(self):
    return copy.deepcopy(self)
  @staticmethod
  def Empty():
    ret = ConfigList()
    ret.conf_list = []
    return ret