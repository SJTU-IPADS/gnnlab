from math import floor, nan
import sys

from runner import cfg_list_collector

# cache_rate = 36
# T_local = 2
# T_remote = 15
# T_cpu = 60
# T_partition = (T_local + T_remote) / 2
# freq_cdf_fname = "run-logs/report_optimal_samgraph_cache_gcn_kKHop2_uk-2006-05_cache_by_presample_1_cache_rate_000_batch_size_8000_optimal_cache_hit.txt"
# freq_abs_val_fname = "run-logs/report_optimal_samgraph_cache_gcn_kKHop2_uk-2006-05_cache_by_presample_1_cache_rate_000_batch_size_8000_frequency_histogram.txt"

def r_rate_to_gpu_rate(rate : int):
  return (cache_rate - rate) * 2 + rate

def query_freq_cdf(rate : int):
  if rate == 0:
    return 0.0
  with open(freq_cdf_fname) as f:
    lines = f.readlines()
  # print(lines, rate)
  for line in lines:
    if rate == int(line.split("\t")[0]):
      return float(line.split("\t")[1].strip())

def query_freq_abs(rate : int):
  if rate == 0:
    return 0.0
  with open(freq_abs_val_fname) as f:
    lines = f.readlines()
  # print(lines, rate)
  for line in lines:
    if rate == int(line.split("\t")[0]):
      return float(line.split("\t")[1].strip())

def performance(replicate_rate : int):
  partition_rate = min((cache_rate - replicate_rate) * 2, 100)
  cpu_rate = 100 - replicate_rate - partition_rate
  replicate_freq = query_freq_cdf(replicate_rate)
  partition_freq = query_freq_cdf(replicate_rate + partition_rate) - replicate_freq
  cpu_freq = 1 - replicate_freq - partition_freq
  return replicate_freq * T_local + partition_freq * T_partition + cpu_freq * T_cpu

def optimal_r_rate():
  mu = (T_cpu - T_partition) / (T_partition - T_local)
  with open(freq_abs_val_fname) as f:
    lines = f.readlines()
  idx_to_freq = [None for _ in range(101)]
  for idx in range(101):
    line = lines[idx]
    assert(int(line.split("\t")[0]) == idx)
    freq = float(line.split("\t")[1].strip())
    idx_to_freq[idx] = freq
  idx_to_freq[0] = 0.0
  idx_to_freq[100] = 0.0
  # print(idx_to_freq)
  # print(cache_rate)
  # print(mu)
  for r_rate in range(cache_rate, -1, -1):
    gpu_rate = r_rate_to_gpu_rate(r_rate)
    # print(idx_to_freq[r_rate], idx_to_freq[gpu_rate + 1])
    if idx_to_freq[gpu_rate + 1] == 0:
      return r_rate
    if idx_to_freq[r_rate] / idx_to_freq[gpu_rate + 1] > mu:
      return r_rate
  assert(False)

if __name__ == "__main__":

  print("{:6} {:6} {:6} "
        "{:6} {:7} "
        "{:7} {:7} {:7} {:9} | "
        "{:7} {:5} {}".format(
          "Rep", "Part", "Full", 
          "Opt", "Boost", 
          "Rep", "Part", "CPU", "FreqBound",
          "Cache", "Cache", "Config"))
  for cfg in cfg_list_collector.conf_list:
    for (cache_size, T_local, T_remote, T_cpu) in [(11, 2, 15, 60),(27, 2, 15, 60),(35, 1, 500/240, 500/11),(75, 1, 500/240, 500/11)]:
      T_partition = (T_local + T_remote) / 2

      cache_rate = min(floor(100 * cache_size / cfg.dataset.FeatGB()), 100)
      if cache_rate == 100:
        continue
      # print(cfg.get_log_fname())
      freq_cdf_fname = f"{cfg.get_log_fname()}_optimal_cache_hit.txt"
      freq_abs_val_fname = f"{cfg.get_log_fname()}_frequency_histogram.txt"
      optimal_performance = min(performance(optimal_r_rate()),performance(optimal_r_rate()+1),performance(optimal_r_rate()-1))
      print("{:6.2f} {:6.2f} {:6.2f} "
            "{:6.2f} {:6.2f}% "
            "{:6.2f}% {:6.2f}% {:6.2f}% {:9.4f} | "
            "{:5}GB {:4}% {}".format(
          performance(cache_rate),  performance(0), nan if cache_rate < 50 else performance(100 - (100 - cache_rate) * 2), 
          # optimal_performance, optimal_performance / min(performance(cache_rate), performance(0)) * 100,
          optimal_performance, optimal_performance / performance(cache_rate) * 100,
          optimal_r_rate(), r_rate_to_gpu_rate(optimal_r_rate()) - optimal_r_rate(), 100 - r_rate_to_gpu_rate(optimal_r_rate()), query_freq_abs(r_rate_to_gpu_rate(optimal_r_rate())+1),
          cache_size,
          cache_rate,
          cfg.get_log_fname()
      ))

  # # performance of all replicate
  # print(performance(cache_rate))

  # # performance of all parition
  # print(performance(0))

  # # performance of store all and use all size
  # if cache_rate > 50:
  #   print(performance(100 - (100 - cache_rate) * 2))

  # # performance of optimal
  # print(performance(optimal_r_rate()))
  # print(performance(optimal_r_rate()+1))
  # print(performance(optimal_r_rate()-1))