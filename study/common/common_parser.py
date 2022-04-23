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

from numpy import NaN
from runner_helper import RunConfig, ConfigList, SampleType, CachePolicy, percent_gen, Dataset, App, Arch
import copy
import re
import sys
import glob
import math
import traceback

base_cfg_list = ConfigList()

policy_str = {
  0 : "degree",
  1 : "heuristic",
  2 : "presample",
  3 : "degree_hop",
  4 : "presample_static",
  5 : "fake_optimal",
  6 : "dynamic_cache",
  7 : "random",

  11: "presample_1",
  12: "presample_2",
  13: "presample_3",
  14: "presample_4",
  15: "presample_5",
  16: "presample_6",
  17: "presample_max",
  20: "no_cache",
}
policy_str_short = {
  0 : "Deg",
  1 : "heuristic",
  2 : "PreS",
  # 3 : "degree_hop",
  # 4 : "presample_static",
  # 5 : "fake_optimal",
  # 6 : "dynamic_cache",
  7 : "Rand",

  11: "PreS_1",
  12: "PreS_2",
  13: "PreS_3",
  14: "PreS_4",
  15: "PreS_5",
  16: "PreS_6",
  # 17: "presample_max",
  # 20: "no_cache",
}

size_unit_to_coefficient = {
  'GB':1024*1024*1024,
  'MB':1024*1024,
  'KB':1024,
  'Bytes':1,
}

def sample_method_str_short(cfg):
  if cfg.app is App.gcn:
    num_hop = 3
  elif cfg.app is App.graphsage:
    num_hop = 2
  if cfg.sample_type is SampleType.kRandomWalk:
    return 'RndWlk'
  elif cfg.sample_type is SampleType.kKHop2:
    return str(num_hop)+ 'HopRnd'
  elif cfg.sample_type is SampleType.kWeightedKHopPrefix:
    return str(num_hop)+ 'HopWgt'

def my_assert_eq(a, b):
  if a != b:
    print(a, "!=", b)
    assert(False)

def grep_from(fname, pattern, line_ctx=[0,0]):
  p = re.compile(pattern)
  with open(fname) as f:
    lines = f.readlines()
    ret_lines = []
    if len(lines) == 0:
      return ret_lines
    for i in range(len(lines)):
      if p.match(lines[i]):
        ret_lines += lines[max(0, i - line_ctx[0]):min(len(lines), i + line_ctx[1] + 1)]
    return ret_lines
def filter_from(line_list, pattern):
  ret_lines = []
  p = re.compile(pattern)
  for line in line_list:
    if p.match(line):
      ret_lines.append(line)
  return ret_lines
def exclude_from(line_list, pattern):
  ret_lines = []
  p = re.compile(pattern)
  for line in line_list:
    if not p.match(line):
      ret_lines.append(line)
  return ret_lines


default_meta_list = ['app', 'dataset', 'cache_policy', 'cache_percentage',
              'step_sample_time', 'step_copy_time', 'step_convert_time', 'step_train_time',
              'step_feature_KB', 'step_label_KB', 'step_id_KB', 'step_graph_KB', 'step_miss_KB',
              'batch_size', 'sampler_gpu', 'trainer_gpu', 'sample_type',
              ]
class BenchInstance:
  def __init__(self):
    pass
  def init_from_cfg(self, cfg):
    try:
      fname = cfg.get_log_fname() + '.log'
      self.vals = {}
      self.fname = fname
      self.prepare_config(cfg)
      self.prepare_init(cfg)
      self.prepare_epoch_eval(cfg)
      self.prepare_profiler_log(cfg)
      self.vals['optimal_hit_percent'] = self.get_optimal()
    except Exception:
      print("error when ", fname)
      print(traceback.format_exc())
      traceback.print_exc()
      sys.exit(1)
    return self

  def prepare_config(self, cfg):
    self.cfg = cfg
    fname = cfg.get_log_fname() + '.log'
    l = fname.split('/')[-1].split('.')[0].split('_')
    i = iter(l)
    config_str_list = grep_from(fname, "\(\'.*", [0, 0])
    if len(config_str_list) > 0:
      # old fashion:
      # ('key', 'val')
      for cur_str in config_str_list:
        cur_str : str
        if cur_str.startswith("('"):
          key = cur_str.split(',')[0][2:-1]
          val = cur_str.split(',')[1].strip()[:-1]
          if val[0] == "'":
            val = val[1:-1]
          self.vals[key] = val
    else:
      config_str_list = grep_from(fname, "^config:.*", [0, 0])
      for cur_str in config_str_list:
        cur_str : str
        key = cur_str.split('=')[0][7:]
        val = cur_str.split('=')[1].strip()
        self.vals[key] = val
    for k,v in cfg.__dict__.items():
      self.vals[k] = v
    self.vals['dataset'] = str(cfg.dataset)
    self.vals['cache_policy'] = policy_str[cfg.cache_policy.value]
    self.vals['cache_policy_short'] = policy_str_short[cfg.cache_policy.value]
    self.vals['sample_type_short'] = sample_method_str_short(cfg)
    self.vals['seq_num'] = getattr(cfg, 'seq_num', None)
    self.vals['cache_percentage'] = 100 * cfg.cache_percent
    self.vals['sample_type'] = cfg.sample_type.name
    self.vals['app'] = cfg.app.name
  
  def get_optimal(self):
    optimal_cfg = copy.deepcopy(self.cfg)
    optimal_cfg.cache_percent = 0
    optimal_cfg.report_optimal = 1
    optimal_cfg.multi_gpu = False
    optimal_cfg.cache_policy = CachePolicy.cache_by_degree
    optimal_file_list = glob.glob(optimal_cfg.get_log_fname() + '_optimal_cache_hit.txt')
    if len(optimal_file_list) == 0:
      return math.nan
    elif len(optimal_file_list) > 1:
      raise Exception("more than one optimal file to choose")
    with open(optimal_file_list[0]) as f:
      for line in f.readlines():
        percent = int(line.split("\t")[0])
        hit_rate = float(line.split("\t")[1])
        if abs(percent - self.vals['cache_percentage']) < 1e-4:
          return hit_rate*100
    raise Exception("can not find optimal cache hit rate")

  def prepare_init(self, cfg):
    self.vals['init:presample'] = math.nan
    self.vals['init:load_dataset:mmap'] = math.nan
    self.vals['init:load_dataset:copy'] = math.nan
    self.vals['init:dist_queue'] = math.nan
    self.vals['init:internal'] = math.nan
    self.vals['init:cache'] = math.nan
    self.vals['init:other'] = math.nan
    self.vals['init:copy'] = math.nan
    fname = cfg.get_log_fname() + '.log'
    init_str_list = grep_from(fname, "^test_result:init:.*", [0, 0])
    for line in init_str_list:
      m2 = re.match(r'test_result:(.+)=(.+)\n', line)
      if m2:
        key = m2.group(1)
        value = m2.group(2)
        self.vals[key] = float(value)
    if self.vals['init:presample'] is not math.nan:
      self.vals['init:load_dataset:copy'] = self.vals['init:load_dataset:copy:sampler'] + self.vals['init:load_dataset:copy:trainer']
      self.vals['init:dist_queue']        = self.vals['init:dist_queue:alloc+push']     + self.vals['init:dist_queue:pin:sampler']   + self.vals['init:dist_queue:pin:trainer']
      self.vals['init:internal']          = self.vals['init:internal:sampler']          + self.vals['init:internal:trainer']
      self.vals['init:cache']             = self.vals['init:cache:sampler']             + self.vals['init:cache:trainer']
      self.vals['init:other']             = self.vals['init:dist_queue']                + self.vals['init:internal']
      self.vals['init:copy'] = self.vals['init:load_dataset:copy'] + self.vals['init:cache']
  def prepare_epoch_eval(self, cfg):
    self.vals['epoch_time'] = math.nan
    fname = cfg.get_log_fname() + '.log'
    epoch_rst_list = exclude_from(grep_from(fname, "^test_result:.*", [0, 0]), "^test_result:init:.*")
    self.vals['pipeline_train_epoch_time'] = NaN
    self.vals['pipeline_train_epoch_time'] = NaN
    self.vals['epoch_time:sample_total']   = NaN
    self.vals['epoch_time:copy_time']      = NaN
    self.vals['epoch_time:train_total']    = NaN
    for line in epoch_rst_list:
      m2 = re.match(r'test_result:(.+)=(.+)\n', line)
      if m2:
        key = m2.group(1)
        value = m2.group(2)
        if key != 'cache_percentage':
          self.vals[key] = float(value)
    if 'cache_hit_rate' not in self.vals:
      self.vals['hit_percent'] = NaN
    if 'hit_percent' not in self.vals:
      self.vals['hit_percent'] = float(self.vals['cache_hit_rate'])*100
    if cfg.pipeline:
      self.vals['epoch_time'] = self.vals['pipeline_train_epoch_time']
      self.vals['train_process_time'] = self.vals['pipeline_train_epoch_time']
    else:
      if self.vals['epoch_time:sample_total'] is NaN and 'epoch_time:sample_time' in self.vals:
        self.vals['epoch_time:sample_total'] = self.vals['epoch_time:sample_time']
      self.vals['epoch_time'] = self.vals['epoch_time:sample_total'] + self.vals['epoch_time:copy_time'] + self.vals['epoch_time:train_total']
      self.vals['train_process_time'] = self.vals['epoch_time:copy_time'] + self.vals['epoch_time:train_total']

    self.vals['epoch_time'] = '{:.2f}'.format(self.vals['epoch_time'])
    self.vals['train_process_time'] = '{:.2f}'.format(self.vals['train_process_time'])

  @staticmethod
  def prepare_profiler_log_merge_groups(result_map_list, cfg):
    rst = {}
    for result_map in result_map_list:
      for key,val in result_map.items():
        if key in rst:
          val = max(rst[key], val)
        rst[key] = val
    # print(rst)
    return rst
  @staticmethod
  def prepare_profiler_log_one_group(line_list):
    result_map = {}
    if len(line_list) == 0:
      return None
    assert(line_list[0].startswith('    ['))
    global_prefix = line_list[0].strip()[1:5]
    for i in range(1, len(line_list)):
      line = line_list[i].strip()
      if line.find(':') != -1:
        prefix = line[:line.find(':')]
        line = line[len(prefix)+1:]
      else:
        prefix = line[:2]
        line = line[len(prefix):]
      item_list = line.split('|')
      item_list = [global_prefix + ' ' + prefix + ' ' + item.strip() for item in item_list]
      for item in item_list:
        m=re.match(r'([^\.]*) +([0-9\.]*)( (MB|Bytes|KB|GB))?', item)
        # print(item, m)
        key,val = m.group(1).strip(),float(m.group(2))
        if m.group(3):
          val *= size_unit_to_coefficient[m.group(4)]
        assert(key not in result_map)
        result_map[key] = val
    return result_map

  def prepare_profiler_log(self, cfg):
    line_list = grep_from(cfg.get_log_fname() + '.log', r'^(    \[Step|        L|    \[Init).*')
    line_list.append('    [END]')
    result_map_list = []
    cur_begin = -1
    for i in range(0, len(line_list)):
      line = line_list[i]
      if line.startswith('    ['):
        if cur_begin != -1:
          result_map_list.append(self.prepare_profiler_log_one_group(line_list[cur_begin:i]))
        cur_begin = i
    result_map = self.prepare_profiler_log_merge_groups(result_map_list, cfg)
    self.vals.update(result_map)
    # print(result_map_list)

  def to_formated_str(self):
    self.vals['cache_percentage'] = '{:.1f}'.format(self.vals['cache_percentage'])
    self.vals['hit_percent']      = '{:.3f}'.format(self.vals['hit_percent'])
    self.vals['optimal_hit_percent']      = '{:.3f}'.format(self.vals['optimal_hit_percent'])
    if 'step_sample_time' in self.vals:
      for key in ['step_sample_time','step_copy_time','step_train_time','step_convert_time','step_full_train_time']:
        self.vals[key] = '{:.4f}'.format(self.vals[key])
      self.vals['step_miss_KB']    = '{:.0f}'.format(self.vals['step_miss_KB'])
      self.vals['step_feature_KB'] = '{:.0f}'.format(self.vals['step_feature_KB'])
      self.vals['step_miss_MB']    = '{:.2f}'.format(self.vals['step_miss_MB'])
      self.vals['step_feature_MB'] = '{:.2f}'.format(self.vals['step_feature_MB'])
    self.vals['dataset_short'] = {'papers100M':'PA', 'products':'PR', 'uk-2006-05':'UK', 'twitter':'TW', 'papers100M_300':'PR_3', 'papers100M_600':'PR_6'}[self.vals['dataset']]
    for key in ['init:presample','init:load_dataset:mmap','init:load_dataset:copy','init:dist_queue','init:internal','init:cache','init:other','init:copy']:
      self.vals[key] = '{:.2f}'.format(self.vals[key])

  @staticmethod
  def print_dat(inst_list: list, outf, meta_list = default_meta_list, custom_col_title_list=None, sep='\t'):
    if custom_col_title_list is None:
      custom_col_title_list = meta_list
    print(sep.join(custom_col_title_list), file=outf)
    for inst in inst_list:
      try:
        inst.to_formated_str()
        print(sep.join([str(inst.vals[meta]) for meta in meta_list]), file=outf)
      except KeyError:
        print("error when ", inst.fname)
        # print(sys.exc_info())
        traceback.print_exc()
        print(sep.join([str(inst.vals[meta]) if meta in inst.vals else "None" for meta in meta_list]), file=outf)
        # sys.exit(1)
    pass

def assign_sequence_number(cfg_list : list, start=0):
  for i in range(len(cfg_list)):
    cfg_list[i].seq_num = i + start