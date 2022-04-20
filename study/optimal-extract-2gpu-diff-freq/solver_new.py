from math import floor, nan
import sys, pulp, subprocess, numpy, os
from unittest import result

from runner import cfg_list_collector, Dataset, App, SampleType

(cfg_list_collector
  .select('dataset', [
    Dataset.uk_2006_05,
    # Dataset.twitter,
  ])
  .select('app', [App.pinsage])
  .select('app', [
    # App.gcn,
    App.pinsage,
    # App.graphsage,
  ])
  # .select('sample_type', [
  #   SampleType.kKHop2,
  #   SampleType.kRandomWalk,
  #   SampleType.kWeightedKHopPrefix,
  # ])
  # .select('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=0/2', 'export SAMGRAPH_TRAIN_SET_PART=1/2'])
  # .select('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=0/4', 'export SAMGRAPH_TRAIN_SET_PART=1/4'])
)

num_slot = 100
num_part = 2

# row is rank in part1, col is rank in part2
def get_density_array(cfg1, cfg2):
  cmd = "./gen_density_matrix_new " + " ".join([cfg.get_log_fname() + suffix for suffix in ["_optimal_cache_bin.bin", "_optimal_cache_freq_bin.bin"] for cfg in [cfg1, cfg2]])
  print(cmd)
  subprocess.check_output(cmd, shell=True)
  # density_array = numpy.memmap('density.bin', mode='r', shape=((num_slot,)*len(cfg_list)))
  # freq_array = numpy.memmap('freq.bin', mode='r', shape=(num_slot,len(cfg_list)))
  density_array = numpy.memmap('density.bin', dtype='float64', mode='r', shape=(num_slot, num_slot))
  freq_array = numpy.memmap('freq.bin', dtype='float64', mode='r', shape=(num_slot, num_part))
  return density_array, freq_array
# def get_percent_per_block(cfg1, cfg2):
#   part1_bin_file_name = cfg1.get_log_fname() + "_optimal_cache_bin.bin"
#   part2_bin_file_name = cfg2.get_log_fname() + "_optimal_cache_bin.bin"
#   output = subprocess.check_output(f"./gen_density_matrix {part1_bin_file_name} {part2_bin_file_name}", shell=True).decode('utf-8').strip()
#   # print(f"./gen_density_matrix {part1_bin_file_name} {part2_bin_file_name}")
#   # print(output)
#   full_matrix = [line.split('\t') for line in output.split('\n')]
#   core_matrix = [line[1:-1] for line in full_matrix[1:]]
#   val_matrix = [[float(entry) / 100 for entry in line] for line in core_matrix]
#   # print(val_matrix)
#   return val_matrix

# def get_E_per_rank(freq_histogram_file_name):
#   # part1_bin_file_name = cfg_list.conf_list[0].get_log_fname() + "_frequency_histogram.txt"
#   # part2_bin_file_name = cfg_list.conf_list[1].get_log_fname() + "_frequency_histogram.txt"
#   with open(freq_histogram_file_name) as f:
#     lines = f.readlines()
#   E1_per_block = [None for _ in range(101)]
#   for idx in range(101):
#     line = lines[idx]
#     assert(int(line.split("\t")[0]) == idx)
#     freq = float(line.split("\t")[1].strip())
#     E1_per_block[idx] = freq
#   E1_per_block = E1_per_block[1:]
#   E1_per_block[-1] = float(0)
#   # print(E1_per_block)
#   return E1_per_block

class Problem:
  def __init__(self, cfg_part1, cfg_part2):
    '''
    layout: consider [i][j], 0<=i,j<100; i means top (i+1)% in first partition, j means top (j+1)% in second partition
    E1 & E2 means frequency of each i or j.
    '''
    # split each part's node ranking to 100 slots. with 2 part, we have 100x100 blocks.

    # this stores how many nodes this blocks have
    self.cfg1 = cfg_part1
    self.density_array, self.freq_array = get_density_array(cfg_part1, cfg_part2)
    # self.percent_per_block = get_percent_per_block(cfg_part1, cfg_part2)
    # self.E1_per_row = get_E_per_rank(cfg_part1.get_log_fname() + "_frequency_histogram.txt")
    # self.E2_per_col = get_E_per_rank(cfg_part2.get_log_fname() + "_frequency_histogram.txt")
    self.percent_per_block = self.density_array
    self.E1_per_row = self.freq_array[:,0]
    self.E2_per_col = self.freq_array[:,1]
    self.x1_list_optimal = None
    self.x2_list_optimal = None
    pass

  def solve(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    solver = pulp.getSolver('GUROBI_CMD')
    # solver = pulp.getSolver('PULP_CBC_CMD')
    prob = pulp.LpProblem(name='NoName', sense=pulp.LpMinimize)

    ##### get super parameter
    percent_per_block = self.percent_per_block
    E1_per_row = self.freq_array[:,0]
    E2_per_col = self.freq_array[:,1]

    # binary variables
    x1_list = [[None for _ in range(100)] for _ in range(100)]
    x2_list = [[None for _ in range(100)] for _ in range(100)]
    r1_list = [[None for _ in range(100)] for _ in range(100)]
    r2_list = [[None for _ in range(100)] for _ in range(100)]
    c_list  = [[None for _ in range(100)] for _ in range(100)]

    for i in range(100):
      for j in range(100):
        x1_list[i][j] = pulp.LpVariable(f'x_g1_{i}_{j}', cat=pulp.LpBinary, e=None)
        x2_list[i][j] = pulp.LpVariable(f'x_g2_{i}_{j}', cat=pulp.LpBinary, e=None)
        r1_list[i][j] = pulp.LpVariable(f'r_g1_{i}_{j}', cat=pulp.LpBinary, e=None)
        r2_list[i][j] = pulp.LpVariable(f'r_g2_{i}_{j}', cat=pulp.LpBinary, e=None)
        c_list[i][j] = pulp.LpVariable(f'c_{i}_{j}', cat=pulp.LpBinary, e=None)

    # final goal: max of each gpu's time
    z = pulp.LpVariable('z', cat=pulp.LpContinuous)

    prob += z

    ##### create constraints
    # first connects r1,r2,c to x1,x2
    # c = ~(x1|x2)     => x1+x2+c >= 1, x1+x2+2*c <= 2
    # r1 = (~x1) & x2  => r1+x1 >= x2, 3*r1+x1 <= x2+2
    # r2 = x1 & (~x2)  => r2+x2 >= x1, 3*r2+x2 <= x1+2
    for i in range(100):
      for j in range(100):
        prob += x1_list[i][j] + x2_list[i][j] + c_list[i][j] >= 1, f"cpu_req_1_{i}_{j}"
        prob += x1_list[i][j] + x2_list[i][j] + 2*c_list[i][j] <= 2, f"cpu_req_2_{i}_{j}"
        prob += r1_list[i][j] + x1_list[i][j] - x2_list[i][j] >= 0, f"remote_1_req_1_{i}_{j}"
        prob += r2_list[i][j] + x2_list[i][j] - x1_list[i][j] >= 0, f"remote_2_req_1_{i}_{j}"
        prob += 3*r1_list[i][j] + x1_list[i][j] - x2_list[i][j] <= 2, f"remote_1_req_2_{i}_{j}"
        prob += 3*r2_list[i][j] + x2_list[i][j] - x1_list[i][j] <= 2, f"remote_2_req_2_{i}_{j}"

    # capacity of each gpu
    prob += pulp.lpSum([percent_per_block[i][j] * x1_list[i][j] for j in range(100) for i in range(100)]) <= cache_percent, "cap_1"
    prob += pulp.lpSum([percent_per_block[i][j] * x2_list[i][j] for j in range(100) for i in range(100)]) <= cache_percent, "cap_2"

    # time of each gpu
    prob += pulp.lpSum([E1_per_row[i]*percent_per_block[i][j]*(T_local*x1_list[i][j] + T_remote*r1_list[i][j] + T_cpu*c_list[i][j]) for j in range(100) for i in range(100)]) - z <= 0, "time_g1"
    prob += pulp.lpSum([E2_per_col[j]*percent_per_block[i][j]*(T_local*x2_list[i][j] + T_remote*r2_list[i][j] + T_cpu*c_list[i][j]) for i in range(100) for j in range(100)]) - z <= 0, "time_g2"

    prob.solve(solver)

    # verify the result
    for i in range(100):
      for j in range(100):
        x1 = round(pulp.value(x1_list[i][j]))
        x2 = round(pulp.value(x2_list[i][j]))
        r1 = round(pulp.value(r1_list[i][j]))
        r2 = round(pulp.value(r2_list[i][j]))
        c = round(pulp.value(c_list[i][j]))
        assert(c == 1 - (x1 | x2))
        assert(r1 == x2 * (1 - x1))
        assert(r2 == x1 * (1 - x2))

    # now we only want 0/1 value of each block
    self.x1_list_optimal = [[round(pulp.value(x1)) for x1 in row] for row in x1_list]
    self.x2_list_optimal = [[round(pulp.value(x2)) for x2 in row] for row in x2_list]

    # print(f"PULP's final result: {pulp.value(z)}")

    ret = [pulp.value(z)]

    ret += self.calculate_final_time(self.x1_list_optimal, self.x2_list_optimal, self.density_array, self.E1_per_row, self.E2_per_col, T_local, T_remote, T_cpu)
    return ret

  def get_naive_result(self, cache_percent, T_local = 2, T_remote = 15, T_cpu = 60):
    x1_list_naive = [[1 if i < cache_percent else 0 for j in range(100)] for i in range(100)]
    x2_list_naive = [[1 if j < cache_percent else 0 for j in range(100)] for i in range(100)]
    ret = []
    ret += self.calculate_final_time(x1_list_naive, x2_list_naive, self.density_array, self.E1_per_row, self.E2_per_col, T_local, T_remote, T_cpu)
    return ret

  @staticmethod
  def get_T(x_self, x_remote, T_local, T_remote, T_cpu):
    return x_self * T_local + (1-x_self)*x_remote*T_remote + (1-x_self)*(1-x_remote)*T_cpu
  @staticmethod
  def calculate_final_time(x1_list, x2_list, percent_per_block, E1_per_row, E2_per_col, T_local = 2, T_remote = 15, T_cpu = 60):
    t_g1 = 0
    t_g2 = 0
    rate_rep = 0
    rate_part_g1 = 0
    rate_part_g2 = 0
    rate_cpu = 0
    t_g1_cpu = 0
    for i in range(100):
      for j in range(100):
        t_g1 += percent_per_block[i][j] * E1_per_row[i] * Problem.get_T(x1_list[i][j], x2_list[i][j], T_local, T_remote, T_cpu)
        t_g2 += percent_per_block[i][j] * E2_per_col[j] * Problem.get_T(x2_list[i][j], x1_list[i][j], T_local, T_remote, T_cpu)
        if x1_list[i][j] + x2_list[i][j] == 0:
          t_g1_cpu += percent_per_block[i][j] * E2_per_col[j] * Problem.get_T(x2_list[i][j], x1_list[i][j], T_local, T_remote, T_cpu)
        rate_rep     += percent_per_block[i][j] * x1_list[i][j] * x2_list[i][j]
        rate_part_g1 += percent_per_block[i][j] * x1_list[i][j] * (1 - x2_list[i][j])
        rate_part_g2 += percent_per_block[i][j] * (1 - x1_list[i][j]) * x2_list[i][j]
        rate_cpu     += percent_per_block[i][j] * (1 - x1_list[i][j]) * (1- x2_list[i][j])
    print(t_g1_cpu)
    return [max(t_g1, t_g2), t_g1, t_g2, rate_rep, rate_part_g1, rate_part_g2, rate_cpu]
    # print(f"{max(t_g1, t_g2):8.2f}=max({t_g1:8.2f},{t_g2:8.2f}). rep={rate_rep:5.2f}% part1={rate_part_g1:5.2f}% part2={rate_part_g2:5.2f}% cpu={rate_cpu:5.2f}%")

# p = Problem(cfg_list_collector.conf_list[0], cfg_list_collector.conf_list[1])
# p.solve(16)

part1_conf_list = cfg_list_collector.copy().select('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=0/2'])
part2_conf_list = cfg_list_collector.copy().select('custom_env', ['export SAMGRAPH_TRAIN_SET_PART=1/2'])

summary_list = []


for i in range(len(part1_conf_list.conf_list)):
  for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11),(75, 1, 500/240, 500/11)]:
    print(f"solving with cache={cache_size} {T_local} {T_remote} {T_cpu} {part1_conf_list.conf_list[i].get_log_fname()} ")
    p = Problem(part1_conf_list.conf_list[i], part2_conf_list.conf_list[i])
    cache_percent = min(floor(100 * cache_size / part1_conf_list.conf_list[i].dataset.FeatGB()), 100)
    # result_list = p.solve(cache_percent, T_local, T_remote, T_cpu)
    naive_result_list = p.get_naive_result(cache_percent, T_local, T_remote, T_cpu)
    summary = ("{:6.2f} | "
              #  "{:8.2f}={:8.2f}=max({:8.2f},{:8.2f}) {:5.2f} {:5.2f} {:5.2f} {:5.2f} | "
               "{:8.2f}=max({:8.2f},{:8.2f}) {:6.2f} {:5.2f} {:5.2f} {:5.2f} | "
               "{:2}GB {:5.2f} {:5.2f} {:5.2f} {}".format(
                  # result_list[0]/result_list[8]*100,
                  nan,
                  # *result_list,
                  *naive_result_list,
                  cache_size, T_local, T_remote, T_cpu, part1_conf_list.conf_list[i].get_log_fname()))
    print(summary)
    summary_list.append(summary)
  #   break
  # break


print("{:6} | "
      "{:^64} | "
      "{:^56} | "
      "{:30}".format(
        " ", 
        "Optimal",
        "Selfish",
        "Info"))

print("{:6} | "
      "{:8} {:8}     {:8} {:8}  {:5} {:5} {:5} {:5} | "
      "{:8}     {:8} {:8}  {:6} {:5} {:5} {:5} | "
      "{:4} {:5} {:5} {:5} {}".format(
        "Boost", 
        "Pulp", "Time", "Time1", "Time2", "Rep", "Part1", "Part2", "CPU",
        "Time", "Time1", "Time2", "Rep", "Part1", "Part2", "CPU",
        "GPU", "T_L", "T_R", "T_C", "LOG"))

for summary in summary_list:
  print(summary)