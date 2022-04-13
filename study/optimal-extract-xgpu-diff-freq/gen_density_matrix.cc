#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <numeric>
#include <functional>
#include <fstream>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

float alpha = 0;
uint32_t num_slot = 20;
uint32_t num_node;
double coefficient = 1.4;
uint32_t cache_rate = 0;

int freq_to_slot_1(float freq, uint32_t rank) {
  if (freq == 0) return num_slot - 1;
  if (freq >= alpha) return 0;
  double exp = std::log2(alpha / (double)freq) / std::log2(coefficient);
  int slot = (int)std::ceil(exp);
  slot = std::min(slot, (int)num_slot - 2);
  return slot;
}
int freq_to_slot_2(float freq, uint32_t rank) {
  return rank * (uint64_t)num_slot / num_node;
}
int freq_to_slot_bin(float freq, uint32_t rank) {
  return rank * (uint64_t)100 >= (uint64_t)cache_rate * num_node;
}
std::function<int(float, uint32_t)> freq_to_slot = freq_to_slot_1;
// std::function<int(float, uint32_t)> freq_to_slot = freq_to_slot_2;

template<typename T>
struct ndarray_view;
template<typename T>
struct ndarray {
  std::vector<uint32_t> _shape;
  uint32_t _num_shape, _num_elem;
  std::vector<uint32_t> _len_of_each_dim;
  T* _data;
  ndarray(const std::vector<uint32_t> & shape) {
    _shape = shape;
    _num_elem = std::accumulate(shape.begin(), shape.end(), 1ul, std::multiplies<size_t>());
    _len_of_each_dim.resize(shape.size());
    _len_of_each_dim.back() = 1;
    for (int i = _shape.size() - 1; i > 0; i--) {
      _len_of_each_dim[i - 1] = _len_of_each_dim[i] * _shape[i];
    }
    _data = new T[_num_elem] {0};
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

int main(int argc, char** argv) {
  // if (argc < 2 || (argc - 1) % 2 != 0) {
  //   std::cerr << "Usage: ./gen_density_matrix <part1_id_bin_file> <part2_id_bin_file> ... <part1_freq_bin_file> <part2_freq_bin_file> ...\n";
  //   abort();
  // }

  std::vector<std::string> bin_filename_list;
  std::string method = "exp";
  CLI::App _app;
  _app.add_option("-s,--slot", num_slot, "number of slots per partition.");
  _app.add_option("-f,--file", bin_filename_list, "list of binary file: first file list of sorted id, then file list of corresponding frequency")->required();
  _app.add_option("-m,--method", method, "method to split slot. default is exp: to split exponential")->check(CLI::IsMember({"exp","num","bin"}));
  _app.add_option("-c,--cache", cache_rate, "cache rate. must be conjunction with -m bin");
  try {
    _app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return _app.exit(e);
  }

  if (method == "exp") {
    freq_to_slot = freq_to_slot_1;
  } else if (method == "num") {
    freq_to_slot = freq_to_slot_2;
  } else {
    freq_to_slot = freq_to_slot_bin;
    num_slot = 2;
  }

  uint32_t num_part = bin_filename_list.size() / 2;
  std::vector<uint32_t*> part_id_list(num_part);
  std::vector<float*> part_freq_list(num_part);
  struct stat st;
  stat(bin_filename_list[0].c_str(), &st);
  num_node = st.st_size / sizeof(uint32_t);
  // std::cerr << num_node << "\n";
  for (int i = 0; i < num_part; i++) {
    int fd_id = open(bin_filename_list[i].c_str(), O_RDONLY);
    int fd_freq = open(bin_filename_list[i + num_part].c_str(), O_RDONLY);
    part_id_list[i] = (uint32_t*)mmap(nullptr, num_node * sizeof(uint32_t), PROT_READ, MAP_PRIVATE, fd_id, 0);
    part_freq_list[i] = (float*)mmap(nullptr, num_node * sizeof(float), PROT_READ, MAP_PRIVATE, fd_freq, 0);
  }
  // identify freq boundary of first slot
  for (int i = 0; i < num_part; i++) {
    if (alpha < part_freq_list[i][(num_node - 1) / 100]) {
      alpha = part_freq_list[i][(num_node - 1) / 100];
    }
  }

  ndarray<uint32_t> nid_to_slot({num_node, num_part});
  ndarray<double> slot_freq_array({num_slot, num_part});
  ndarray<double> slot_density_array({num_slot, num_part});

  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t part_idx = 0; part_idx < num_part; part_idx++) {
      uint32_t nid = part_id_list[part_idx][orig_rank];
      float freq = part_freq_list[part_idx][orig_rank];
      int slot_id = freq_to_slot(freq, orig_rank);
      nid_to_slot[nid][part_idx].ref() = slot_id;
      slot_freq_array[slot_id][part_idx].ref() += freq;
      slot_density_array[slot_id][part_idx].ref() += 1;
    }
  }
  for (uint32_t slot_id = 0; slot_id < num_slot; slot_id++) {
    for (uint32_t part_id = 0; part_id < num_part; part_id++) {
      if (slot_density_array[slot_id][part_id].ref() > 0) {
        slot_freq_array[slot_id][part_id].ref() /= slot_density_array[slot_id][part_id].ref();
      }
    }
  }
  // std::vector<std::vector<double>> matrix(100, std::vector<double>(100, 0));
  ndarray<double> block_density_array(std::vector<uint32_t>(num_part, num_slot));
  for (uint32_t nid = 0; nid < num_node; nid++) {
    ndarray_view<uint32_t> block_idx = nid_to_slot[nid];
    block_density_array[block_idx].ref() += 1;
  }
  for (uint32_t block_id = 0; block_id < block_density_array._num_elem; block_id++) {
    // block_density_array._data[block_id] *= 100/(double)num_node ;
  }

  std::fstream ofs_density = std::fstream("density.bin", std::ios_base::binary | std::ios_base::out);
  std::fstream ofs_freq = std::fstream("freq.bin", std::ios_base::binary | std::ios_base::out);
  std::fstream ofs_slot_density = std::fstream("slot_density.bin", std::ios_base::binary | std::ios_base::out);

  ofs_density.write((char*)block_density_array._data, sizeof(double) * block_density_array._num_elem);
  ofs_density.close();
  ofs_freq.write((char*)slot_freq_array._data, sizeof(double)*slot_freq_array._num_elem);
  ofs_freq.close();
  ofs_slot_density.write((char*)slot_density_array._data, sizeof(double)*slot_density_array._num_elem);
  ofs_slot_density.close();

  // bellow is sample code for 2 partition
  std::cout.precision(2);
  std::cout << "p0\\p1\t\t";
  for (uint32_t j = 0; j < num_slot; j++) {
    std::cout << j << "\t";
  }
  std::cout << "sum\n";

  std::cout << "\tE0\\E1\t";
  for (uint32_t j = 0; j < num_slot; j++) {
    std::cout << slot_freq_array[j][1].ref() << "\t";
  }
  std::cout << "\n";

  for (uint32_t i = 0; i < num_slot; i++) {
    std::cout << i << "\t" << slot_freq_array[i][0].ref() << "\t";
    for (uint32_t j = 0; j < num_slot; j++) {
      std::cout << block_density_array.at({i, j}) << "\t";
    }
    std::cout << slot_density_array[i][0].ref() * 100/(double)num_node << "\n";
  }
  
  std::cout << "sum\t\t";
  for (uint32_t j = 0; j < num_slot; j++) {
    std::cout << slot_density_array[j][1].ref() * 100/(double)num_node << "\t";
  }
  std::cout << "\n";
}