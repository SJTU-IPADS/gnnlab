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
import math

LOG_DIR='run-logs/logs_samgraph_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TMP_LOG_DIR='run-logs/tmp_log_dir'

def percent_gen(lb, ub, gap=1):
  ret = []
  i = lb
  while i <= ub:
    ret.append(i/100)
    i += gap
  return ret

def reverse_percent_gen(lb, ub, gap=1):
  ret = percent_gen(lb, ub, gap)
  return list(reversed(ret))

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
  coll_intuitive = 9
  partition = 10

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

  coll_intuitive_1 = 51
  coll_intuitive_2 = 52
  coll_intuitive_3 = 53
  coll_intuitive_4 = 54
  coll_intuitive_5 = 55
  coll_intuitive_6 = 56
  coll_intuitive_7 = 57
  coll_intuitive_8 = 58
  coll_intuitive_9 = 59
  coll_intuitive_10 = 60
  coll_intuitive_max = 61

  partition_1 = 71
  partition_2 = 72
  partition_3 = 73
  partition_4 = 74
  partition_5 = 75
  partition_6 = 76
  partition_7 = 77
  partition_8 = 78
  partition_9 = 79
  partition_10 = 80
  partition_max = 81

  part_rep_1 = 91
  part_rep_2 = 92
  part_rep_10 = 100
  part_rep_max = 101

  rep_1 = 111
  rep_2 = 112
  rep_4 = 114
  rep_10 = 120
  rep_max = 121

  coll_cache_asymm_link_1 = 131
  coll_cache_asymm_link_2 = 132
  coll_cache_asymm_link_4 = 134
  coll_cache_asymm_link_10 = 140
  coll_cache_asymm_link_max = 141

  clique_part_1 = 151
  clique_part_2 = 152
  clique_part_4 = 154
  clique_part_10 = 160
  clique_part_max = 161

  clique_part_by_degree_1 = 171
  clique_part_by_degree_2 = 172
  clique_part_by_degree_10 = 180
  clique_part_by_degree_max = 181

  no_cache = 200
  def get_samgraph_policy_param_name(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return "pre_sample"
    if self.value in range(CachePolicy.coll_cache_1.value, CachePolicy.coll_cache_max.value):
      return "coll_cache"
    if self.value in range(CachePolicy.coll_intuitive_1.value, CachePolicy.coll_intuitive_max.value):
      return "coll_intuitive"
    if self.value in range(CachePolicy.partition_1.value, CachePolicy.partition_max.value):
      return "partition"
    if self.value in range(CachePolicy.part_rep_1.value, CachePolicy.part_rep_max.value):
      return "part_rep"
    if self.value in range(CachePolicy.rep_1.value, CachePolicy.rep_max.value):
      return "rep"
    if self.value in range(CachePolicy.coll_cache_asymm_link_1.value, CachePolicy.coll_cache_asymm_link_max.value):
      return "coll_cache_asymm_link"
    if self.value in range(CachePolicy.clique_part_1.value, CachePolicy.clique_part_max.value):
      return "clique_part"
    if self.value in range(CachePolicy.clique_part_by_degree_1.value, CachePolicy.clique_part_by_degree_max.value):
      return "clique_part_by_degree"
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
      'coll_intuitive',
      'partition',
    ]
    return name_list[self.value]
  def get_presample_epoch(self):
    if self.value in range(CachePolicy.cache_by_presample_1.value, CachePolicy.cache_by_presample_max.value):
      return self.value - CachePolicy.cache_by_presample_1.value + 1
    if self.value in range(CachePolicy.coll_cache_1.value, CachePolicy.coll_cache_max.value):
      return self.value - CachePolicy.coll_cache_1.value + 1
    if self.value in range(CachePolicy.coll_intuitive_1.value, CachePolicy.coll_intuitive_max.value):
      return self.value - CachePolicy.coll_intuitive_1.value + 1
    if self.value in range(CachePolicy.partition_1.value, CachePolicy.partition_max.value):
      return self.value - CachePolicy.partition_1.value + 1
    if self.value in range(CachePolicy.part_rep_1.value, CachePolicy.part_rep_max.value):
      return self.value - CachePolicy.part_rep_1.value + 1
    if self.value in range(CachePolicy.rep_1.value, CachePolicy.rep_max.value):
      return self.value - CachePolicy.rep_1.value + 1
    if self.value in range(CachePolicy.coll_cache_asymm_link_1.value, CachePolicy.coll_cache_asymm_link_max.value):
      return self.value - CachePolicy.coll_cache_asymm_link_1.value + 1
    if self.value in range(CachePolicy.clique_part_1.value, CachePolicy.clique_part_max.value):
      return self.value - CachePolicy.clique_part_1.value + 1
    if self.value in range(CachePolicy.clique_part_by_degree_1.value, CachePolicy.clique_part_by_degree_max.value):
      return self.value - CachePolicy.clique_part_by_degree_1.value + 1
    return 1
  def get_log_fname(self):
    if self is CachePolicy.cache_by_presample:
      return CachePolicy.cache_by_presample_1.name
    if self is CachePolicy.coll_cache:
      return CachePolicy.coll_cache_1.name
    if self is CachePolicy.coll_intuitive:
      return CachePolicy.coll_intuitive_1.name
    if self is CachePolicy.partition:
      return CachePolicy.partition_1.name
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
  def TopoGB(self):
    return [math.nan, 0.4700, 6.4326, 13.7007, math.nan , 11.3358, 5.6252, 12.4394, 13.7785][self.value]

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
    self.unsupervised = False
    self.amp = False
    self.max_num_step = None
    self.coll_cache_no_group = ""
    self.coll_cache_concurrent_link = ""
    self.rolling = 0

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
    if self.multi_gpu:
      cmd_line += f'COLL_NUM_REPLICA={self.num_trainer} '
    elif self.multi_gpu_sgnn:
      cmd_line += f'COLL_NUM_REPLICA={self.num_worker} '
    cmd_line += f'CUDA_LAUNCH_BLOCKING={self.cuda_launch_blocking} '
    cmd_line += 'SAMGRAPH_LOG_NODE_ACCESS=0 '
    cmd_line += f'SAMGRAPH_LOG_NODE_ACCESS_SIMPLE={self.report_optimal} '
    cmd_line += f'SAMGRAPH_DUMP_TRACE={self.dump_trace} '
    if self.coll_cache_no_group != "":
      cmd_line += f'SAMGRAPH_COLL_CACHE_NO_GROUP={self.coll_cache_no_group} '
    if self.coll_cache_concurrent_link != "":
      cmd_line += f' SAMGRAPH_COLL_CACHE_CONCURRENT_LINK_IMPL={self.coll_cache_concurrent_link} SAMGRAPH_COLL_CACHE_CONCURRENT_LINK=1 '
    else:
      cmd_line += f' SAMGRAPH_COLL_CACHE_CONCURRENT_LINK=0 '
    if self.num_feat_dim_hack != None:
      cmd_line += f'SAMGRAPH_FAKE_FEAT_DIM={self.num_feat_dim_hack} '
    if self.custom_env != '':
      cmd_line += f'{self.custom_env} '
    if self.multi_gpu:
      if self.async_train:
        cmd_line += f'python ../../example/samgraph/multi_gpu/async/train_{self.app.name}.py'
      elif self.mps_mode != None:
        cmd_line += f'python ../../example/samgraph/multi_gpu/mps/train_{self.app.name}.py'
      elif self.unsupervised:
        cmd_line += f'python ../../example/samgraph/multi_gpu/unsupervised/train_{self.app.name}.py'
      else:
        cmd_line += f'python ../../example/samgraph/multi_gpu/train_{self.app.name}.py'
    elif self.multi_gpu_sgnn:
      if self.unsupervised:
        cmd_line += f'python ../../example/samgraph/sgnn/unsupervised/train_{self.app.name}.py'
      else:
        cmd_line += f'python ../../example/samgraph/sgnn/train_{self.app.name}.py'
    elif self.unsupervised:
      cmd_line += f'python ../../example/samgraph/unsupervised/train_{self.app.name}.py'
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
    elif self.multi_gpu_sgnn:
      cmd_line += f' --num-worker {self.num_worker} '

    if self.amp:
      cmd_line += ' --amp '
    if self.unsupervised:
      cmd_line += f' --unsupervised'
    if self.max_num_step:
      cmd_line += f' --max-num-step {self.max_num_step}'

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
    cmd_line += f' --rolling {self.rolling}'

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
    if self.unsupervised:
      std_out_log += "unsupervised_"
    if self.multi_gpu_sgnn:
      std_out_log += "sgnn_"
    std_out_log += '_'.join(
      [self.system.name]+self.cache_log_name() + self.pipe_log_name() +
      [self.app.name, self.sample_type.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache_rate_{round(self.cache_percent*100):0>3}', f'batch_size_{self.batch_size}'])
    if self.rolling != 0:
      std_out_log += f'_rolling_{self.rolling}'
    if self.coll_cache_no_group != "":
      std_out_log += f'_nogroup_{self.coll_cache_no_group}'
    if self.coll_cache_concurrent_link != "":
      std_out_log += f'_concurrent_impl_{self.coll_cache_concurrent_link}'
    return std_out_log

  def beauty(self):
    self.preprocess_sample_type()
    msg = ' '.join(
      ['Running', self.system.name] + self.cache_log_name() + self.pipe_log_name() + 
      (["unsupervised"] if self.unsupervised else []) +
      [self.app.name, self.sample_type.name, str(self.dataset), self.cache_policy.get_log_fname()] + 
      [f'cache rate:{round(self.cache_percent*100):0>3}%', f'batch size:{self.batch_size}', ])
    if self.coll_cache_no_group != "":
      msg += f' nogroup={self.coll_cache_no_group}'
    if self.coll_cache_concurrent_link != "":
      msg += f' concurrent_link={self.coll_cache_concurrent_link}'
    return datetime.datetime.now().strftime('[%H:%M:%S]') + msg

  def run(self, mock=False, durable_log=True, callback = None, fail_only=False):
    '''
    fail_only: only run previously failed job. fail status is recorded in json file
    '''
    previous_succeed = False
    if fail_only:
      try:
        with open(self.get_log_fname() + '.log', "r") as logf:
          first_line = logf.readline().strip()
        if first_line == "succeed=True":
          previous_succeed = True
      except Exception as e:
        pass
      if previous_succeed:
        if callback != None:
          callback(self)
        return 0

    if mock:
      print(self.form_cmd(durable_log))
    else:
      print(self.beauty())

      if durable_log:
        os.system('mkdir -p {}'.format(self.logdir))
      status = os.system(self.form_cmd(durable_log))
      if os.WEXITSTATUS(status) != 0:
        print("FAILED!")
        if durable_log:
          self.prepend_log_succeed(False)
        return 1
      else:
        if durable_log:
          self.prepend_log_succeed(True)
      if callback != None:
        callback(self)
    return 0
  def prepend_log_succeed(self, succeed_bool):
    with open(self.get_log_fname() + '.log', "r") as logf:
      log_content = logf.readlines()
    with open(self.get_log_fname() + '.log', "w") as logf:
      print(f"succeed={succeed_bool}", file=logf)
      print("".join(log_content), file=logf)

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

  def override_T(self, key, val_list):
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val in val_list:
        cfg = copy.deepcopy(cfg)
        setattr(cfg, key, val)
        self.conf_list.append(cfg)
    return self

  def part_override(self, filter_key, filter_val_list, override_key, override_val_list):
    newlist = []
    for cfg in self.conf_list:
      # print(cfg.cache_impl, cfg.logdir, filter_key, filter_val_list)
      if getattr(cfg, filter_key) in filter_val_list:
        # print(cfg.cache_impl, cfg.logdir)
        for val in override_val_list:
          # print(cfg.cache_impl, cfg.logdir)
          cfg = copy.deepcopy(cfg)
          setattr(cfg, override_key, val)
          newlist.append(cfg)
      else:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def hyper_override(self, key_array, val_matrix):
    if len(key_array) == 0 or len(val_matrix) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val_list in val_matrix:
        cfg = copy.deepcopy(cfg)
        for idx in range(len(key_array)):
          setattr(cfg, key_array[idx], val_list[idx])
        self.conf_list.append(cfg)
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
  @staticmethod
  def MakeList(conf):
    ret = ConfigList()
    if isinstance(conf, list):
      ret.conf_list = conf
    elif isinstance(conf, RunConfig):
      ret.conf_list = [conf]
    else:
      raise Exception("Please construct fron runconfig or list of it")
    return ret

  def run(self, mock=False, durable_log=True, callback = None, fail_only=False):
    for conf in self.conf_list:
      conf : RunConfig
      conf.run(mock, durable_log, callback, fail_only=fail_only)

  def run_stop_on_fail(self, mock=False, durable_log=True, callback = None, fail_only=False):
    last_conf = None
    last_ret = None
    for conf in self.conf_list:
      conf : RunConfig
      if last_conf != None and (
                      conf.unsupervised == last_conf.unsupervised and 
                      conf.app == last_conf.app and 
                      conf.sample_type == last_conf.sample_type and 
                      conf.batch_size == last_conf.batch_size and 
                      conf.dataset == last_conf.dataset):
        if conf.cache_percent == last_conf.cache_percent and last_ret != 0:
          continue
        if conf.cache_percent > last_conf.cache_percent and last_ret != 0 :
          continue
        if conf.cache_percent < last_conf.cache_percent and last_ret == 0 :
          continue
      ret = conf.run(mock, durable_log, callback, fail_only=fail_only)
      last_conf = conf
      last_ret = ret