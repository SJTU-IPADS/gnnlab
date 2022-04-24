#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cassert>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <omp.h>

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

#define CPU_VAR(block_id) "c_block_" << (block_id)
#define GPU_VAR(block_id, part_id) "x_block_" << (block_id) << "_part_" << (part_id)
// #define REMOTE_VAR(block_id, part_id) "r_block_" << (block_id) << "_part_" << (part_id)

uint32_t num_part;
uint32_t num_block;
uint32_t cache_percent = 14;
double * density_array;
double * block_freq_array;
double T_local = 2, T_remote = 15, T_cpu = 60;

uint32_t NUM_THREADS = omp_get_num_threads();

inline bool ignore_block(uint32_t block_id, double weight) {
  return (weight == 0) && (block_id > 0);
}

void do_parallel(std::ostream & ss, std::function<void(std::ostream& ss, uint32_t iter)> func, uint32_t iter_begin, uint32_t iter_end) {
  std::vector<std::stringstream> local_ss_list(NUM_THREADS);
#pragma omp parallel for num_threads(NUM_THREADS)
  for (uint32_t thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
    for (uint32_t iter = iter_begin + thread_idx; iter < iter_end; iter += NUM_THREADS) {
      func(local_ss_list[thread_idx], iter);
    }
  }
  for (auto & local_ss : local_ss_list) {
    ss << local_ss.str();
  }
}
double do_parallel_sum(std::ostream & ss, std::function<double(std::ostream& ss, uint32_t iter)> func, uint32_t iter_begin, uint32_t iter_end) {
  std::vector<std::stringstream> local_ss_list(NUM_THREADS);
  std::vector<double> local_sum_list(NUM_THREADS, 0);
#pragma omp parallel for num_threads(NUM_THREADS)
  for (uint32_t thread_idx = 0; thread_idx < NUM_THREADS; thread_idx++) {
    for (uint32_t iter = iter_begin + thread_idx; iter < iter_end; iter += NUM_THREADS) {
      local_sum_list[thread_idx] += func(local_ss_list[thread_idx], iter);
    }
  }
  double sum = 0;
  FOR_LOOP(thread_idx, NUM_THREADS) {
    ss << local_ss_list[thread_idx].str();
    sum += local_sum_list[thread_idx];
  }
  return sum;
}

void constraint_connect_c_x(std::ostream &ss, uint32_t block_id) {
  if (density_array[block_id] == 0) return;
  ss << "cpu_req_1_" << block_id << ": " 
     << CPU_VAR(block_id);
  FOR_LOOP(part_id, num_part) {
    ss << " + " << GPU_VAR(block_id, part_id);
  }
  ss << " >= 1\n";

  ss << "cpu_req_2_" << block_id << ": " 
     << num_part << " " << CPU_VAR(block_id);
  FOR_LOOP(part_id, num_part) {
    ss << " + " << GPU_VAR(block_id, part_id);
  }
  ss << " <= " << num_part << "\n";
}
void constraint_connect_r_x(std::ostream & ss, uint32_t block_id, uint32_t part_id) {
}
void constraint_capacity(std::ostream & ss, uint32_t part_id) {
  ss << "cap_" << part_id << ": ";
  ss << density_array[0] << " " << GPU_VAR(0, part_id) << "\n";
  do_parallel(ss, [part_id](std::ostream & ss, uint32_t block_id){
    if (density_array[block_id] == 0) return;
    ss << " + " << density_array[block_id] << " " << GPU_VAR(block_id, part_id) << "\n";
  }, 1, num_block);
  // FOR_LOOP_1(block_id, num_block) {
  //   if (density_array[block_id] == 0) continue;
  //   ss << " + " << density_array[block_id] << " " << GPU_VAR(block_id, part_id) << "\n";
  // }
  ss << " <= " << cache_percent << "\n";
}
void constraint_time(std::ostream & ss, uint32_t part_id) {
  ss << "time_" << part_id << ": ";
  double sum_weight = 0;
  {
    uint32_t block_id = 0;
    double weight = density_array[block_id] * block_freq_array[block_id * num_part + part_id];
    sum_weight += weight;
    ss << " + " << (weight * (T_cpu - T_remote)) << " " << CPU_VAR(block_id) << "\n";
    ss << " - " << (weight * (T_remote - T_local)) << " " << GPU_VAR(block_id, part_id) << "\n";
  }
  sum_weight += do_parallel_sum(ss, [part_id](std::ostream & ss, uint32_t block_id)->double{
    double weight = density_array[block_id] * block_freq_array[(block_id) * num_part + part_id];
    if (weight == 0) return weight;
    ss << " + " << (weight * (T_cpu - T_remote)) << " " << CPU_VAR(block_id) << "\n";
    ss << " - " << (weight * (T_remote - T_local)) << " " << GPU_VAR(block_id, part_id) << "\n";
    return weight;
  }, 1, num_block);
  // FOR_LOOP_1(block_id, num_block) {
  //   double weight = density_array[block_id] * block_freq_array[block_id * num_part + part_id];
  //   if (weight == 0) continue;
  //   sum_weight += weight;
  //   ss << " + " << (weight * (T_cpu - T_remote)) << " " << CPU_VAR(block_id) << "\n";
  //   ss << " - " << (weight * (T_remote - T_local)) << " " << GPU_VAR(block_id, part_id) << "\n";
  // }
  ss << " - z <= " << (-sum_weight * T_remote) << "\n";
}

int main(int argc, char** argv) {
  std::vector<std::string> bin_filename_list;
  std::string method = "exp";
  CLI::App _app;
  _app.add_option("-f,--file", bin_filename_list, "list of binary file: density, then freq")->required();
  _app.add_option("-c,--cache", cache_percent, "cache percent");
  _app.add_option("--tl", T_local, "time local");
  _app.add_option("--tr", T_remote, "time remote");
  _app.add_option("--tc", T_cpu, "time cpu");
  _app.add_option("-t,--thread", NUM_THREADS, "num of threads");
  try {
    _app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return _app.exit(e);
  }

  assert(bin_filename_list.size() == 2);
  struct stat st;
  stat(bin_filename_list[0].c_str(), &st);
  num_block = st.st_size / sizeof(double);
  stat(bin_filename_list[1].c_str(), &st);
  num_part = st.st_size / sizeof(double) / num_block;

  std::cout << num_block << " blocks, " << num_part << "parts\n";

  int density_fd = open(bin_filename_list[0].c_str(), O_RDONLY);
  int block_freq_fd = open(bin_filename_list[1].c_str(), O_RDONLY);
  density_array    = (double*)mmap(nullptr, num_block *            sizeof(double), PROT_READ, MAP_PRIVATE, density_fd, 0);
  block_freq_array = (double*)mmap(nullptr, num_block * num_part * sizeof(double), PROT_READ, MAP_PRIVATE, block_freq_fd, 0);

  std::ofstream ss("NoName-cppgen.lp", std::ios_base::out);
  // std::stringstream ss;
  ss  << "\\* NoName *\\\n"
      << "Minimize\n"
      << "OBJ: z\n"
      << "Subject To\n";
  std::cout << "Capaticy...\n";
  FOR_LOOP(part_id, num_part) {constraint_capacity(ss, part_id);}

  std::cout << "Connect CPU...\n";
  // FOR_LOOP(block_id, num_block) {constraint_connect_c_x(ss, block_id);}
  do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
    constraint_connect_c_x(ss, block_id);
  }, 0, num_block);

  std::cout << "Connect Remote...\n";
  // FOR_LOOP(block_id, num_block) {
  //   FOR_LOOP(part_id, num_part) {constraint_connect_r_x(ss, block_id, part_id);}
  // }
  do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
    FOR_LOOP(part_id, num_part) {constraint_connect_r_x(ss, block_id, part_id);}
  }, 0, num_block);

  std::cout << "Time...\n";
  FOR_LOOP(part_id, num_part) {constraint_time(ss, part_id);}

  ss  << "Bounds\n"
      << " z free\n"
      << "Binaries\n";

  std::cout << "CPU VAR...\n";
  // FOR_LOOP(block_id, num_block) {
  //   if (ignore_block(block_id, density_array[block_id])) continue; ss << " " << CPU_VAR(block_id) << "\n";
  // }
  do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
    if (ignore_block(block_id, density_array[block_id])) return; ss << " " << CPU_VAR(block_id) << "\n";
  }, 0, num_block);

  std::cout << "GPU VAR...\n";
  // FOR_LOOP(block_id, num_block) {
  //   if (ignore_block(block_id, density_array[block_id])) continue; FOR_LOOP(part_id, num_part) {ss << " " << GPU_VAR(block_id, part_id) << "\n";}
  // }
  do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
    if (ignore_block(block_id, density_array[block_id])) return; FOR_LOOP(part_id, num_part) {ss << " " << GPU_VAR(block_id, part_id) << "\n";}
  }, 0, num_block);
  ss << "END\n";
  ss.close();
}