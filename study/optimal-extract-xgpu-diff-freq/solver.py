import math, time
import sys, pulp, subprocess, numpy, os

from runner import cfg_list_collector, Dataset, App, SampleType
from runner_helper import ConfigList

(cfg_list_collector
  # .select('dataset', [
  #   Dataset.uk_2006_05,
  #   # Dataset.twitter,
  # ])
  # .select('app', [
  #   # App.gcn,
  #   App.pinsage,
  #   # App.graphsage,
  # ])
  # # .select('sample_type', [
  # #   SampleType.kKHop2,
  # #   SampleType.kRandomWalk,
  # #   SampleType.kWeightedKHopPrefix,
  # # ])
)

num_threads = 108
num_slot = 100
num_part = 2
num_stream = 1
stream_mapping = [0, 0]
coefficient = 1.05

def try_and_throw(exp):
  if not exp:
    raise Exception("")

def block_id_to_slot_id(block_id, stream_id, num_slot):
  return (block_id // (num_slot ** (num_stream - stream_id - 1))) % num_slot

# row is rank in part1, col is rank in part2
def get_density_array(cfg_list : list, method='exp', n_slot=20, cache_percent=0):
  cmd = f"./gen_density_matrix_fine_grain --coe {coefficient} -t {num_threads} -f " + " ".join([cfg.get_log_fname() + suffix for suffix in ["_optimal_cache_bin.bin", "_optimal_cache_freq_bin.bin"] for cfg in cfg_list]) + f" -m {method}" + f" -s {n_slot}" + f" -c {cache_percent}"
  print(cmd)
  subprocess.check_output(cmd, shell=True)
  # density_array = numpy.memmap('density.bin', mode='r', shape=((num_slot,)*len(cfg_list)))
  # freq_array = numpy.memmap('freq.bin', mode='r', shape=(num_slot,len(cfg_list)))
  density_array = numpy.memmap('density.bin', dtype='float64', mode='r')
  # freq_array = numpy.memmap('freq.bin', dtype='float64', mode='r', shape=(n_slot, num_part))
  block_freq_array = numpy.memmap('block_freq.bin', dtype='float64', mode='r', shape=(len(density_array), num_stream))
  return density_array, None, block_freq_array

class Problem:
  def __init__(self, cfg_list:list, mode = 'CONT'):
    '''
    layout: consider [i][j], 0<=i,j<100; i means top (i+1)% in first partition, j means top (j+1)% in second partition
    E1 & E2 means frequency of each i or j.
    '''
    # split each part's node ranking to 100 slots. with 2 part, we have 100x100 blocks.

    # this stores how many nodes this blocks have
    self.cfg_list = cfg_list
    self.num_block = num_slot ** num_stream
    self.density_array, _, self.block_freq_array = get_density_array(cfg_list, n_slot=num_slot)
    self.x_list_optimal = None
    self.mode = mode
    assert(mode in ['BIN', 'CONT'])
    pass

  def solve(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    print("building pulp problem")
    solver = pulp.getSolver('GUROBI_CMD', options=[("MIPgap", 0.005), ("Threads", num_threads), ("Method", 2)])
    # solver = MINDOPT(options={})
    # solver = pulp.getSolver('PULP_CBC_CMD')
    prob = pulp.LpProblem(name='NoName', sense=pulp.LpMinimize)

    ##### get super parameter
    density_array = self.density_array
    block_freq_array = self.block_freq_array

    # binary variables
    x_list = [[None for _ in range(num_part)] for _ in range(self.num_block)]
    c_list = [None  for _ in range(self.num_block)]

    if self.mode == 'BIN':
      param_kwargs = {"cat" : pulp.LpBinary}
    else:
      param_kwargs = {"lowBound":0,"upBound":1, "cat":pulp.LpContinuous}

    for block_id in range(self.num_block):
      for part_id in range(num_part):
        x_list[block_id][part_id] = pulp.LpVariable(f'x_block_{block_id:05d}_part_{part_id}', **param_kwargs, e=None)
      c_list[block_id] = pulp.LpVariable(f'c_block_{block_id:05d}', **param_kwargs, e=None)

    # final goal: max of each gpu's time
    z = pulp.LpVariable('z', cat=pulp.LpContinuous)

    prob += z

    print("building pulp problem - constraints")
    ##### create constraints
    # first connects r,c to x
    if self.mode == 'BIN':
      for block_id in range(self.num_block):
        prob += pulp.lpSum(x_list[block_id]) + c_list[block_id] >= 1, f"cpu_req_1_{block_id:05d}"
        prob += pulp.lpSum(x_list[block_id]) + num_part * c_list[block_id] <= num_part, f"cpu_req_2_{block_id:05d}"
    else:
      for block_id in range(self.num_block):
        prob += pulp.lpSum(x_list[block_id]) + c_list[block_id] >= 1, f"cpu_req_1_{block_id:05d}"
        for part_id in range(num_part):
          prob += 1 - c_list[block_id] >= x_list[block_id][part_id], f"remote_req_1_{block_id:05d}_{part_id}"

    print("building pulp problem - constraints - capacity")
    # capacity of each gpu
    for part_id in range(num_part):
      prob += pulp.lpSum([density_array[block_id] * x_list[block_id][part_id] for block_id in range(self.num_block)]) <= cache_percent, f"cap_{part_id}"

    print("building pulp problem - constraints - run time")
    # time of each gpu
    for part_id in range(num_part):
      stream_id = stream_mapping[part_id]
      # prob += pulp.lpSum([block_freq_array[block_id][part_id] * density_array[block_id] * (T_local * x_list[block_id][part_id] + T_remote * r_list[block_id][part_id] + T_cpu *c_list[block_id]) for block_id in range(self.num_block)]) - z <= 0, f"time_{part_id}"
      prob += pulp.lpSum([block_freq_array[block_id][stream_id] * density_array[block_id] * ((T_local-T_remote) * x_list[block_id][part_id] + T_remote + (T_cpu-T_remote) *c_list[block_id]) for block_id in range(self.num_block)]) - z <= 0, f"time_{part_id}"

    print("solving pulp problem")
    prob.writeLP("NoName.lp")
    prob.solve(solver)

    # verify the result
    for block_id in range(self.num_block):
      if density_array[block_id] == 0 or sum(block_freq_array[block_id]) == 0:
        continue
      if self.mode == 'BIN':
        round_c = round(pulp.value(c_list[block_id]))
        round_x = [round(pulp.value(x_list[block_id][part_id])) for part_id in range(num_part)]
      else:
        round_c = pulp.value(c_list[block_id])
        round_x = [pulp.value(x_list[block_id][part_id]) for part_id in range(num_part)]
      round_r = [1 - round_c - round_x[part_id] for part_id in range(num_part)]
      try:
        try_and_throw(max(1 - sum(round_x), 0) == round_c)
        for part_id in range(num_part):
          try_and_throw(abs(round_x[part_id] + round_r[part_id] + round_c - 1) < 1e-6)
      except Exception as e:
        print(round_x, round_r, round_c, density_array[block_id], block_freq_array[block_id])
    # now we only want 0/1 value of each block
    if self.mode == 'BIN':
      self.x_list_optimal = [[round(pulp.value(x)) for x in row] for row in x_list]
    else:
      self.x_list_optimal = [[pulp.value(x) for x in row] for row in x_list]

    # print(f"PULP's final result: {pulp.value(z)}")

    ret = [pulp.value(z)]

    ret += self.calculate_final_time(self.x_list_optimal, self.density_array, None, self.block_freq_array, T_local, T_remote, T_cpu)
    return ret

  def solve_cpp(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    cmd = f"./solver_gurobi -f density.bin block_freq.bin -m {self.mode} -c {cache_percent} --tl {T_local} --tr {T_remote} --tc {T_cpu} -t {num_threads} --sm {' '.join([str(i) for i in stream_mapping])}"
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    with open('solver_gurobi.out') as f:
      output = f.read()
    print(output)
    return [float(val) for val in output.split(' ')] + [0 for _ in range(2)]

  def get_naive_result(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    # naive_density_array, naive_freq_array, naive_block_freq_array = get_density_array(self.cfg_list, method='num', n_slot=100)
    # x_list_naive = []
    # for block_id in range(len(naive_density_array)):
    #   x_list_naive += [[1 if block_id_to_slot_id(block_id, part_id, 100) < cache_percent else 0 for part_id in range(num_part)]]
    # Problem.calculate_final_time(x_list_naive, naive_density_array, naive_freq_array, naive_block_freq_array, T_local, T_remote, T_cpu)
    naive_density_array, naive_freq_array, naive_block_freq_array = get_density_array(self.cfg_list, method='bin', n_slot=2, cache_percent=cache_percent)
    x_list_naive = []
    for block_id in range(len(naive_density_array)):
      x_list_naive += [[1 - block_id_to_slot_id(block_id, stream_mapping[part_id], 2) for part_id in range(num_part)]]
    return Problem.calculate_final_time(x_list_naive, naive_density_array, naive_freq_array, naive_block_freq_array, T_local, T_remote, T_cpu)

  @staticmethod
  def get_x_remote(x_list, part_id):
    x_local = x_list[part_id]
    return min(sum(x_list) - x_local, 1)
  @staticmethod
  def get_T(x_list, part_id, T_local, T_remote, T_cpu):
    x_cpu = max(1 - sum(x_list), 0)
    x_local = x_list[part_id]
    x_remote = 1 - x_local - x_cpu
    return x_local * T_local + x_remote*T_remote + x_cpu*T_cpu
  @staticmethod
  def calculate_final_time(x_list, density_array, _, block_freq_array, T_local = 2, T_remote = 15, T_cpu = 60):
    t_list = [0 for _ in range(num_part)]
    rate_rep = 0
    rate_part_list = [0 for _ in range(num_part)]
    rate_cpu = 0
    for block_id in range(len(x_list)):
      for part_id in range(num_part):
        stream_id = stream_mapping[part_id]
        t_list[part_id] += density_array[block_id] * block_freq_array[block_id][stream_id] * Problem.get_T(x_list[block_id], part_id, T_local, T_remote, T_cpu)
      #   rate_part_list[part_id] += density_array[block_id] * x_list[block_id][part_id] * (1 - Problem.get_x_remote(x_list[block_id], part_id))
      # rate_rep     += density_array[block_id] * (sum(x_list[block_id]) == num_part)
      # rate_cpu     += density_array[block_id] * (sum(x_list[block_id]) == 0)
    # return [max(t_list), *t_list, rate_rep, *rate_part_list, rate_cpu]
    return [max(t_list), *t_list, rate_rep, rate_cpu]

cfg_list_collector.select('part_num', [num_stream])
stream_1_conf_list = cfg_list_collector.copy().select('part_idx', [0])

summary_list = []

def print_summary(summary_list):
  if len(summary_list) == 0:
    return
  title_list = [" ", "Optimal", "Naive", "Info"]
  print(" | ".join([("{:^" + str(len(summary_list[0][i])) + "}").format(title_list[i]) for i in range(4)]))
  print(" | ".join([
    f"{'Boost':^6}",
    # (f"{'Pulp':^7}={'Time':^7}=max(" + ",".join([f"{'Time' + str(i):^7}" for i in range(num_part)]) + f") {'Rep':^6} (" + ",".join([f"{'Part' + str(i):^5}" for i in range(num_part)]) + f") {'CPU':^5}"),
    # (f"{'Time':^7}=max(" + ",".join([f"{'Time' + str(i):^7}" for i in range(num_part)]) + f") {'Rep':^6} (" + ",".join([f"{'Part' + str(i):^5}" for i in range(num_part)]) + f") {'CPU':^5}"),
    (f"{'Pulp':^7}={'Time':^7}=max(" + ",".join([f"{'Time' + str(i):^7}" for i in range(num_part)]) + f") {'Rep':^6} {'CPU':^5} {'LPTime':^6}"),
    (f"{'Time':^7}=max(" + ",".join([f"{'Time' + str(i):^7}" for i in range(num_part)]) + f") {'Rep':^6} {'CPU':^5} {'LPTime':^6}"),
    f"{'GPU':^4} {'T_L':^5} {'T_R':^5} {'T_C':^5} {'LOG'}"
  ]))
  for summary in summary_list:
    print(" | ".join(summary))

for i in range(len(stream_1_conf_list.conf_list)):
  stream1_cfg = stream_1_conf_list.conf_list[i]
  same_part_list = cfg_list_collector.copy().select('app', [stream1_cfg.app]).select('dataset', [stream1_cfg.dataset]).select('sample_type', [stream1_cfg.sample_type])
  tr_list = [i for i in range(1, 30, 1)]
  for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11),(75, 1, 500/240, 500/11)]:
  # for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11)]:
  # for (cache_size, T_local, T_remote, T_cpu) in [(cache, 1, tr, tc) for tc in [30,] for cache in range(11, 36, 2) for tr in tr_list]:
  # for (cache_size, T_local, T_remote, T_cpu) in [(cache, 1, tr, tc) for tc in [30,] for cache in range(17, 22, 1) for tr in tr_list]:
    print(f"solving with cache={cache_size} {T_local} {T_remote} {T_cpu} {stream1_cfg.get_log_fname()} ")
    cache_percent = min(math.floor(100 * cache_size / stream1_cfg.dataset.FeatGB()), 100)
    result_list = [math.nan for _ in range(4 + 2 * num_part)]
    t0 = time.time()
    p = Problem(same_part_list.conf_list, mode='BIN')
    # result_list = p.solve(cache_percent, T_local, T_remote, T_cpu)
    result_list = p.solve_cpp(cache_percent, T_local, T_remote, T_cpu)
    t1 = time.time()
    naive_result = p.get_naive_result(cache_percent, T_local, T_remote, T_cpu)
    t2 = time.time()
    summary = [
      "{:6.2f}".format(result_list[0] / naive_result[0] * 100),
      # ("{:7.2f}={:7.2f}=max(" + ",".join([r"{:7.2f}" for _ in range(num_part)]) + ") {:6.2f} (" + ",".join([r"{:5.2f}" for _ in range(num_part)]) + ") {:5.2f}").format(*result_list),
      # ("{:7.2f}=max(" + ",".join([r"{:7.2f}" for _ in range(num_part)]) + ") {:6.2f} (" + ",".join([r"{:5.2f}" for _ in range(num_part)]) + ") {:5.2f}").format(*naive_result),
      ("{:7.2f}={:7.2f}=max(" + ",".join([r"{:7.2f}" for _ in range(num_part)]) + ") {:6.2f} {:5.2f} {:6.2f}").format(*result_list, t1-t0),
      ("{:7.2f}=max(" + ",".join([r"{:7.2f}" for _ in range(num_part)]) + ") {:6.2f} {:5.2f} {:6.2f}").format(*naive_result, t2-t1),
      "{:2}GB {:5.2f} {:5.2f} {:5.2f} {}".format(cache_size, T_local, T_remote, T_cpu, stream1_cfg.get_log_fname())
    ]
    summary_list.append(summary)
    print_summary(summary_list)
  #   break
  # break
