import math, time
import sys, pulp, subprocess, numpy, os, re

# from runner_helper import ConfigList


num_threads = 108
coefficient = 1.05

def try_and_throw(exp):
  if not exp:
    raise Exception("")

def block_id_to_slot_id(block_id, stream_id, num_slot, num_stream):
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
  num_stream = len(cfg_list)
  block_freq_array = numpy.memmap('block_freq.bin', dtype='float64', mode='r', shape=(len(density_array), num_stream))
  return density_array, None, block_freq_array

class Result:
  def __init__(self, num_part):
    self.goal = 0
    self.per_part_goal = [0 for _ in range(num_part)]

    self.local_density = [0 for _ in range(num_part)]
    self.remote_density = [0 for _ in range(num_part)]
    self.cpu_density = 0
    self.local_weight = [0 for _ in range(num_part)]
    self.remote_weight = [0 for _ in range(num_part)]
    self.cpu_weight = [0 for _ in range(num_part)]
  def get_L1_header(self):
    return f"{'Time':^17}" + f"{'Location':^22}" + f"{'Access':^22}"
  def get_L2_header(self):
    return f"({'Max':^7} {'Min':^7})" + f"({'L':^6} {'R':^6} {'C':^6})" + f"({'L':^6} {'R':^6} {'C':^6})"
  def get_summary(self):
    return ("({:7.2f} {:7.2f})({:6.2f} {:6.2f} {:6.2f})({:6.2f} {:6.2f} {:6.2f})").format(
        max(self.per_part_goal), min(self.per_part_goal),
        self.local_density[0], self.remote_density[0], self.cpu_density,
        self.local_weight[0],  self.remote_weight[0],  self.cpu_weight[0])
  def get_detail(self):
    return "\n".join([
      f"{self.goal:7.2f}~{max(self.per_part_goal):7.2f}=max(" + ",".join(f"{v:7.2f}" for v in self.per_part_goal) + ")",
      "local_density " + ",".join(f"{v:7.2f}" for v in self.local_density),
      "remote_density" + ",".join(f"{v:7.2f}" for v in self.remote_density),
      "cpu_density   " + ",".join(f"{v:7.2f}" for v in self.cpu_density),
      "local_weight " + ",".join(f"{v:7.2f}" for v in self.local_weight),
      "remote_weight" + ",".join(f"{v:7.2f}" for v in self.remote_weight),
      "cpu_weight   " + ",".join(f"{v:7.2f}" for v in self.cpu_weight),
    ]) 
  def __str__(self):
    return str(self.goal) + str(self.per_part_goal) + str(self.local_density) + str(self.cpu_density) + str(self.local_weight) + str(self.remote_weight) + str(self.cpu_weight) + str(self.remote_density)

class Problem:
  def __init__(self, cfg_list:list, mode = 'CONT', num_slot=10, stream_mapping=None):
    '''
    layout: consider [i][j], 0<=i,j<100; i means top (i+1)% in first partition, j means top (j+1)% in second partition
    E1 & E2 means frequency of each i or j.
    '''
    # split each part's node ranking to 100 slots. with 2 part, we have 100x100 blocks.
    t0 = time.time()
    # this stores how many nodes this blocks have
    self.cfg_list = cfg_list
    self.num_stream = len(cfg_list)
    self.num_block = num_slot ** self.num_stream
    self.density_array, _, self.block_freq_array = get_density_array(cfg_list, n_slot=num_slot)
    self.mode = mode
    if stream_mapping != None:
      self.stream_mapping = stream_mapping
      self.num_part = len(stream_mapping)
    else:
      self.stream_mapping = [i for i in range(self.num_stream)]
      self.num_part = self.num_stream
    assert(mode in ['BIN', 'CONT'])
    self._build_time = time.time() - t0
    self._solve_time = 0
    self.optimal_result = None
    self.naive_result = None
    pass

  def solve(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    num_part = self.num_part
    t0 = time.time()
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
      stream_id = self.stream_mapping[part_id]
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
      x_list_optimal = [[round(pulp.value(x)) for x in row] for row in x_list]
    else:
      x_list_optimal = [[pulp.value(x) for x in row] for row in x_list]

    # print(f"PULP's final result: {pulp.value(z)}")

    optimal_rst = self.calculate_final_time(x_list_optimal, self.density_array, None, self.block_freq_array, self.stream_mapping, T_local, T_remote, T_cpu)
    self.optimal_result = optimal_rst
    self.optimal_result.goal = pulp.value(z)
    self._solve_time = time.time() - t0

  def solve_cpp(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    t0 = time.time()
    cmd = f"./solver_gurobi -f density.bin block_freq.bin -m {self.mode} -c {cache_percent} --tl {T_local} --tr {T_remote} --tc {T_cpu} -t {num_threads} --sm {' '.join([str(i) for i in self.stream_mapping])}"
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    with open('solver_gurobi.out') as f:
      output = f.read()
    # print(output)
    rst = Result(self.num_part)
    rst.goal = float(re.search(r"^solver_time ([0-9\.]*)$", output ,flags=re.MULTILINE).group(1))
    rst.per_part_goal = [float(val) for val in re.search(r"^time_list (.*)$",     output, flags=re.MULTILINE).group(1).split(' ')]
    # weight is in [0,1]
    rst.local_weight  = [float(val)*100 for val in re.search(r"^local_weight (.*)$",  output, flags=re.MULTILINE).group(1).split(' ')]
    rst.cpu_weight    = [float(val)*100 for val in re.search(r"^cpu_weight (.*)$",    output, flags=re.MULTILINE).group(1).split(' ')]
    rst.remote_weight  = [100 - rst.local_weight[i] - rst.cpu_weight[i] for i in range(len(rst.local_weight))]
    # density is in [0, 100]
    rst.local_density = [float(val) for val in re.search(r"^local_density (.*)$", output, flags=re.MULTILINE).group(1).split(' ')]
    rst.cpu_density = float(re.search(r"^cpu_density ([0-9\.]*)$", output ,flags=re.MULTILINE).group(1))
    rst.remote_density = [100 - rst.local_density[i] - rst.cpu_density for i in range(len(rst.local_density))]
    self.optimal_result = rst
    self._solve_time = time.time() - t0

  def get_naive_result(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    t0 = time.time()
    naive_density_array, naive_freq_array, naive_block_freq_array = get_density_array(self.cfg_list, method='bin', n_slot=2, cache_percent=cache_percent)
    x_list_naive = []
    for block_id in range(len(naive_density_array)):
      x_list_naive += [[1 - block_id_to_slot_id(block_id, self.stream_mapping[part_id], 2, self.num_stream) for part_id in range(self.num_part)]]
    naive_rst = Problem.calculate_final_time(x_list_naive, naive_density_array, naive_freq_array, naive_block_freq_array, self.stream_mapping, T_local, T_remote, T_cpu)
    self.naive_result = naive_rst
    self._solve_naive_time = time.time() - t0

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
  def calculate_final_time(x_list, density_array, _, block_freq_array, stream_mapping, T_local = 2, T_remote = 15, T_cpu = 60):
    num_part = len(stream_mapping)
    rst = Result(num_part)
    total_weight_list = [0 for _ in range(num_part)]
    for block_id in range(len(x_list)):
      for part_id in range(num_part):
        stream_id = stream_mapping[part_id]
        rst.per_part_goal[part_id] += density_array[block_id] * block_freq_array[block_id][stream_id] * Problem.get_T(x_list[block_id], part_id, T_local, T_remote, T_cpu)
        rst.local_weight[part_id]  += density_array[block_id] * block_freq_array[block_id][stream_id] * x_list[block_id][part_id]
        rst.cpu_weight[part_id]    += density_array[block_id] * block_freq_array[block_id][stream_id] * max(1 - sum(x_list[block_id]), 0)
        total_weight_list[part_id] += density_array[block_id] * block_freq_array[block_id][stream_id]
        rst.local_density[part_id] += density_array[block_id] * x_list[block_id][part_id]
      rst.cpu_density     += density_array[block_id] * max(1 - sum(x_list[block_id]), 0)
    # weight's sum is undetermined, so manual calculate it's percentage
    for part_id in range(num_part):
      rst.local_weight[part_id]  /= total_weight_list[part_id] / 100
      rst.cpu_weight[part_id]    /= total_weight_list[part_id] / 100
    rst.remote_weight  = [100 - rst.local_weight[i]  - rst.cpu_weight[i] for i in range(num_part)]
    # density's sum already is 100
    rst.remote_density = [100 - rst.local_density[i] - rst.cpu_density for i in range(num_part)]
    rst.goal = max(rst.per_part_goal)
    return rst

  def get_header(self):
    L0_title_list = ["Boost", "Optimal", "Naive", 
      # "Info"
    ]
    L1_title_list = [
      f"{' ':^6}",
      f"{'Pulp':^7} ~ " + self.optimal_result.get_L1_header() + f" {'LPTime':^6}",
      self.naive_result.get_L1_header() + f" {'LPTime':^6}",
      # f"{'Cache':^4} {'SizeGB':^4} {'T_L':^5} {'T_R':^5} {'T_C':^5} {'LOG'}",
    ]
    L2_title_list = [
      f"{' ':^6}",
      f"{'Pulp':^7} ~ " + self.optimal_result.get_L2_header() + f" {'LPTime':^6}",
      self.naive_result.get_L2_header() + f" {'LPTime':^6}",
      # f"{'Cache':^4} {'SizeGB':^4} {'T_L':^5} {'T_R':^5} {'T_C':^5} {'LOG'}",
    ]
    a = "\n".join([
      " | ".join([("{:^" + str(len(L1_title_list[i])) + "}").format(L0_title_list[i]) for i in range(len(L1_title_list))]),
      " | ".join(L1_title_list),
      " | ".join(L2_title_list),
    ])
    return a

  def get_summary(self):
    return " | ".join([
      "{:6.2f}".format(self.optimal_result.goal / self.naive_result.goal * 100),
      f"{self.optimal_result.goal:7.2f} ~ " + self.optimal_result.get_summary() + f" {self._build_time+self._solve_time:6.2f}",
      self.naive_result.get_summary() + f" {self._solve_naive_time:6.2f}",
      # self. "{:2}GB {:5.2f} {:5.2f} {:5.2f} {}".format(cache_size, T_local, T_remote, T_cpu, stream1_cfg.get_log_fname())
    ])

if __name__ == "__main__":
  # how many different frequency stream
  num_stream = 1
  # how many trainer
  # num_part = num_stream
  # stream_mapping = [i for i in range(num_stream)]
  num_part = 4
  stream_mapping = [0, 0, 0, 0]
  num_slot = 100

  summary_list = []

  from runner import cfg_list_collector, Dataset, App, SampleType
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

  cfg_list_collector.select('part_num', [num_stream])
  stream_1_conf_list = cfg_list_collector.copy().select('part_idx', [0])
  for i in range(len(stream_1_conf_list.conf_list)):
    stream1_cfg = stream_1_conf_list.conf_list[i]
    same_part_list = cfg_list_collector.copy().select('app', [stream1_cfg.app]).select('dataset', [stream1_cfg.dataset]).select('sample_type', [stream1_cfg.sample_type])
    tr_list = [i for i in range(1, 30, 1)]
    # for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11),(75, 1, 500/240, 500/11)]:
    for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11)]:
    # for (cache_size, T_local, T_remote, T_cpu) in [(cache, 1, tr, tc) for tc in [30,] for cache in range(11, 36, 2) for tr in tr_list]:
    # for (cache_size, T_local, T_remote, T_cpu) in [(cache, 1, tr, tc) for tc in [30,] for cache in range(17, 22, 1) for tr in tr_list]:
      print(f"solving with cache={cache_size} {T_local} {T_remote} {T_cpu} {stream1_cfg.get_log_fname()} ")
      cache_percent = min(math.floor(100 * cache_size / stream1_cfg.dataset.FeatGB()), 100)
      p = Problem(same_part_list.conf_list, mode='CONT', num_slot=num_slot, stream_mapping=stream_mapping)
      p.solve_cpp(cache_percent, T_local, T_remote, T_cpu)
      p.get_naive_result(cache_percent, T_local, T_remote, T_cpu)

      if len(summary_list) == 0:
        header = p.get_header() + f" | {'Cache':^6} {'SizeGB':^8} {'T_L':^5} {'T_R':^5} {'T_C':^5} {'LOG'}"
        summary_list.append(header)
      summary = p.get_summary() + " | {:6.2f} {:6.2f}GB {:5.2f} {:5.2f} {:5.2f} {}".format(cache_percent, cache_size, T_local, T_remote, T_cpu, stream1_cfg.get_log_fname())
      summary_list.append(summary)

      for line in summary_list:
        print(line)
