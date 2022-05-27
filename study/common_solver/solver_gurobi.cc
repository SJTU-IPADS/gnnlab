#include "gurobi_c++.h"

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


template<typename T>
struct ndarray_view;
template<typename T>
struct ndarray {
  std::vector<uint32_t> _shape;
  uint32_t _num_shape, _num_elem;
  std::vector<uint32_t> _len_of_each_dim;
  T* _data;
  ndarray() {}
  ndarray(const std::vector<uint32_t> & shape) {
    _shape = shape;
    _num_elem = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    _len_of_each_dim.resize(shape.size());
    _len_of_each_dim.back() = 1;
    for (int i = _shape.size() - 1; i > 0; i--) {
      _len_of_each_dim[i - 1] = _len_of_each_dim[i] * _shape[i];
    }
    _data = new T[_num_elem] {};
    _num_shape = _shape.size();
  }

  T& at(const std::vector<uint32_t> & idx) {
    assert(idx.size() <= _num_shape);
    return this->at(idx.data(), idx.size());
  }
  T& at(const uint32_t * idx) {
    return this->at(idx, _num_shape);
  }
  T& at(const uint32_t * idx, const uint32_t idx_len) {
    assert(idx_len > 0);
    uint32_t offset = idx[0];
    for (uint32_t dim_idx = 1; dim_idx < _num_shape; dim_idx++) {
      offset *= _shape[dim_idx];
      offset += (dim_idx < idx_len) ? idx[dim_idx] : 0;
    }
    return _data[offset];
  }

  ndarray_view<T> operator[](const uint32_t idx);
  ndarray_view<T> operator[](const std::vector<uint32_t> & idx_array);
  ndarray_view<T> operator[](const ndarray_view<uint32_t> & idx_array);
  //  {
  //   return ndarray_view<T>(&this)[idx];
  // }
};

template<typename T>
struct ndarray_view {
  uint32_t* _shape;
  uint32_t* _len_of_each_dim;
  uint32_t _num_shape;
  T* _data;
  ndarray_view(ndarray<T> & array) {
    _data = array._data;
    _shape = array._shape.data();
    _len_of_each_dim = array._len_of_each_dim.data();
    _num_shape = array._shape.size();
  }
  ndarray_view(const ndarray_view<T> & view, const uint32_t first_idx) {
    _data = view._data + first_idx * view._len_of_each_dim[0];
    _shape = view._shape + 1;
    _len_of_each_dim = view._len_of_each_dim + 1;
    _num_shape = view._num_shape - 1;
  }
  ndarray_view<T> operator[](const uint32_t idx) {
    return ndarray_view<T>(*this, idx);
  }
  ndarray_view<T> operator[](const std::vector<uint32_t> & idx_array) {
    return _sub_array(idx_array.data(), idx_array.size());
  }
  ndarray_view<T> operator[](const ndarray_view<uint32_t> & idx_array) {
    assert(idx_array._num_shape == 1);
    return _sub_array(idx_array._data, *idx_array._shape);
  }

  T& ref() {
    assert(_num_shape == 0);
    return *_data;
  }
 private:
  ndarray_view<T> _sub_array(const uint32_t * const idx_array, const uint32_t num_idx) {
    ndarray_view<T> ret = *this;
    ret._shape += num_idx;
    ret._len_of_each_dim += num_idx;
    ret._num_shape -= num_idx;
    for (int i = 0; i < num_idx; i++) {
      ret._data += idx_array[i] * _len_of_each_dim[i];
    }
    return ret;
  }
};

template<typename T>
ndarray_view<T>ndarray<T>::operator[](const uint32_t idx){
  return ndarray_view<T>(*this)[idx];
}
template<typename T>
ndarray_view<T>ndarray<T>::operator[](const std::vector<uint32_t> & idx_array){
  return ndarray_view<T>(*this)[idx_array];
}
template<typename T>
ndarray_view<T>ndarray<T>::operator[](const ndarray_view<uint32_t> & idx_array){
  return ndarray_view<T>(*this)[idx_array];
}



#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

GRBVar z;
ndarray<GRBVar> c_list;
ndarray<GRBVar> x_list;
std::vector<GRBLinExpr> time_list;
std::vector<GRBLinExpr> local_weight_list;
std::vector<GRBLinExpr> local_density_list;
// std::vector<GRBLinExpr> remote_weight_list;
std::vector<double> total_weight_list;
std::vector<GRBLinExpr> cpu_weight_list;
GRBLinExpr cpu_density;

std::vector<int> stream_mapping;

uint32_t num_stream;
uint32_t num_part;
uint32_t num_block;
uint32_t cache_percent = 14;
double * density_array;
double * block_freq_array;
double T_local = 2, T_remote = 15, T_cpu = 60;
std::string mode = "CONT";

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

void constraint_connect_c_x(GRBModel & model, uint32_t block_id) {
  if (density_array[block_id] == 0) return;
  GRBLinExpr exp;
  exp += c_list[block_id].ref();
  FOR_LOOP(part_id, num_part) {
    exp += x_list[block_id][part_id].ref();
  }
  model.addConstr(exp >= 1);
  
  if (mode == "BIN") {
    GRBLinExpr exp;
    exp += c_list[block_id].ref() * num_part;
    FOR_LOOP(part_id, num_part) {
      exp += x_list[block_id][part_id].ref();
    }
    model.addConstr(exp <= num_part);
  }
}
void constraint_connect_r_x(GRBModel & model, uint32_t block_id, uint32_t part_id) {
  if (mode == "CONT") {
    if (density_array[block_id] == 0) return;
    model.addConstr(c_list[block_id].ref() + x_list[block_id][part_id].ref() <= 1);
  }
}
void constraint_capacity(GRBModel & model, uint32_t part_id) {
  GRBLinExpr exp;
  // do_parallel(ss, [part_id](std::ostream & ss, uint32_t block_id){
  //   if (density_array[block_id] == 0) return;
  //   ss << " + " << density_array[block_id] << " " << GPU_VAR(block_id, part_id) << "\n";
  // }, 1, num_block);
  FOR_LOOP(block_id, num_block) {
    if (density_array[block_id] == 0) continue;
    exp += x_list[block_id][part_id].ref() * density_array[block_id];
  }
  model.addConstr(exp <= cache_percent);
}
void constraint_time(GRBModel & model, uint32_t part_id) {
  uint32_t stream_id = stream_mapping[part_id];
  double sum_weight = 0;
  // {
  //   uint32_t block_id = 0;
  //   double weight = density_array[block_id] * block_freq_array[block_id * num_part + part_id];
  //   sum_weight += weight;
  //   ss << " + " << (weight * (T_cpu - T_remote)) << " " << CPU_VAR(block_id) << "\n";
  //   ss << " - " << (weight * (T_remote - T_local)) << " " << GPU_VAR(block_id, part_id) << "\n";
  // }
  // sum_weight += do_parallel_sum(ss, [part_id](std::ostream & ss, uint32_t block_id)->double{
  //   double weight = density_array[block_id] * block_freq_array[(block_id) * num_part + part_id];
  //   if (weight == 0) return weight;
  //   ss << " + " << (weight * (T_cpu - T_remote)) << " " << CPU_VAR(block_id) << "\n";
  //   ss << " - " << (weight * (T_remote - T_local)) << " " << GPU_VAR(block_id, part_id) << "\n";
  //   return weight;
  // }, 1, num_block);
  GRBLinExpr &exp = time_list[part_id];
  FOR_LOOP(block_id, num_block) {
    double weight = density_array[block_id] * block_freq_array[(block_id) * num_stream + stream_id];
    if (weight == 0) continue;
    sum_weight += weight;
    exp += c_list[block_id].ref() * (weight * (T_cpu - T_remote)) - x_list[block_id][part_id].ref() * (weight * (T_remote - T_local));

    local_weight_list[part_id] +=  weight * x_list[block_id][part_id].ref();
    cpu_weight_list[part_id]   +=  weight * c_list[block_id].ref();
    local_density_list[part_id] += x_list[block_id][part_id].ref() * density_array[block_id];
  }
  exp += sum_weight * T_remote;
  total_weight_list[part_id] = sum_weight;
  model.addConstr(exp <= z, "time_" + std::to_string(part_id));
  if (part_id != 0) return;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, density_array[block_id])) continue;
    cpu_density += c_list[block_id].ref() * density_array[block_id];
  }
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
  _app.add_option("-m,--mode", mode, "BIN or CONT")->check(CLI::IsMember({"BIN","CONT"}));
  _app.add_option("-t,--thread", NUM_THREADS, "num of threads");
  _app.add_option("--sm,--stream-mapping", stream_mapping, "mapping from part to stream. useful when stream is shared across different parts");
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
  num_stream = st.st_size / sizeof(double) / num_block;

  if (stream_mapping.size() == 0) {
    // by default, each stream is a part
    num_part = num_stream;
    stream_mapping.resize(num_stream);
    for (int i = 0; i < num_part; i++) {
      stream_mapping[i] = i;
    }
  } else {
    num_part = stream_mapping.size();
  }

  std::cerr << num_block << " blocks, " << num_part << "parts\n";

  int density_fd = open(bin_filename_list[0].c_str(), O_RDONLY);
  int block_freq_fd = open(bin_filename_list[1].c_str(), O_RDONLY);
  density_array    = (double*)mmap(nullptr, num_block *              sizeof(double), PROT_READ, MAP_PRIVATE, density_fd, 0);
  block_freq_array = (double*)mmap(nullptr, num_block * num_stream * sizeof(double), PROT_READ, MAP_PRIVATE, block_freq_fd, 0);

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  // BarConvTol
  env.set(GRB_IntParam_Presolve, 0);
  env.set(GRB_IntParam_Method, 2);
  env.set(GRB_IntParam_Threads, NUM_THREADS);
  env.set(GRB_DoubleParam_BarConvTol, 1e-2);
  env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  env.start();

  GRBModel model = GRBModel(env);

  z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");

  c_list = ndarray<GRBVar> ({num_block});
  x_list = ndarray<GRBVar> ({num_block, num_part});
  time_list.resize(num_part);
  local_weight_list.resize(num_part);
  total_weight_list.resize(num_part);
  cpu_weight_list.resize(num_part);
  local_density_list.resize(num_part);

  std::cerr << "Add Var...\n";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, density_array[block_id])) {
      continue;
    }
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(part_id, num_part) {
      x_list[block_id][part_id].ref() = model.addVar(0, 1, 0, var_type);
    }
  }

  std::cerr << "Capacity...\n";
  FOR_LOOP(part_id, num_part) {constraint_capacity(model, part_id);}

  std::cerr << "Connect CPU...\n";
  FOR_LOOP(block_id, num_block) {constraint_connect_c_x(model, block_id);}
  // do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
  //   constraint_connect_c_x(ss, block_id);
  // }, 0, num_block);

  std::cerr << "Connect Remote...\n";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(part_id, num_part) {constraint_connect_r_x(model, block_id, part_id);}
  }
  // do_parallel(ss, [](std::ostream & ss, uint32_t block_id){
  //   FOR_LOOP(part_id, num_part) {constraint_connect_r_x(ss, block_id, part_id);}
  // }, 0, num_block);

  std::cerr << "Time...\n";
  FOR_LOOP(part_id, num_part) {constraint_time(model, part_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.optimize();

  std::ofstream fs("solver_gurobi.out");
  fs << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X);
  double max_gpu_time = 0, min_gpu_time = std::numeric_limits<double>::max();
  FOR_LOOP(part_id, num_part) {
    max_gpu_time = std::max(max_gpu_time, time_list[part_id].getValue());
    min_gpu_time = std::min(min_gpu_time, time_list[part_id].getValue());
  }
  fs << " " << max_gpu_time;
  FOR_LOOP(part_id, num_part) {
    fs << " " << time_list[part_id].getValue();
  }
  fs << " \n";
  fs << "solver_time " << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  fs << "max_time " << max_gpu_time << "\n";
  fs << "time_list";
  FOR_LOOP(part_id, num_part) { fs << " " << time_list[part_id].getValue(); }
  fs << "\n";

  fs << "local_weight";
  FOR_LOOP(part_id, num_part) { fs << " " << local_weight_list[part_id].getValue() / total_weight_list[part_id]; }
  fs << "\n";

  fs << "cpu_weight";
  FOR_LOOP(part_id, num_part) { fs << " " << cpu_weight_list[part_id].getValue() / total_weight_list[part_id]; }
  fs << "\n";

  fs << "local_density";
  FOR_LOOP(part_id, num_part) { fs << " " << local_density_list[part_id].getValue(); }
  fs << "\n";

  fs << "cpu_density " << cpu_density.getValue() << "\n";

  std::ios_base::fmtflags f( fs.flags() );
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, density_array[block_id])) {
      continue;
    }
    fs << "block_" << std::setw(3) << std::left << block_id;
    fs.flags(f);
    fs << std::fixed << std::setw(6) << std::setprecision(4);
    fs << density_array[block_id] << " | ";
    FOR_LOOP(part_id, num_part) {
      fs << " " << x_list[block_id][part_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X);
    }
    fs.flags(f);
    fs << "\n";
  }
  fs.close();
}