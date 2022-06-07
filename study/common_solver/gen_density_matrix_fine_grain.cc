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
#include <omp.h>
#include <atomic>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include "ndarray.hpp"

float alpha = 0;
uint32_t num_slot = 20;
uint32_t num_node;
double coefficient = 1.4;
uint32_t cache_rate = 0;

uint32_t NUM_THREAD = omp_get_num_threads();

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

static std::atomic_uint32_t next_free_block(0);
auto alloc_block = []() {
  return next_free_block.fetch_add(1);
};
static uint32_t max_size_per_block = 10000;

struct block_identifer {
  uint32_t current_block_id = -1;
  uint32_t current_num = 0;
  bool ignore_limit = false;
  std::atomic_flag latch;
  block_identifer() : latch() {}
  void set_ignore_limit() {
    while (latch.test_and_set()) {}
    ignore_limit = true;
    latch.clear();
  }
  uint32_t add_node() {
    if (ignore_limit) {
      return current_block_id;
    }
    while (latch.test_and_set()) {}
    uint32_t selected_block = -1;
    if (current_num < max_size_per_block) {
      selected_block = current_block_id;
      current_num ++;
    } else {
      current_block_id = alloc_block();
      selected_block = current_block_id;
      current_num = 1;
    }
    latch.clear();
    return selected_block;
  }
};

int main(int argc, char** argv) {
  std::vector<std::string> bin_filename_list;
  std::string method = "exp";
  CLI::App _app;
  _app.add_option("-s,--slot", num_slot, "number of slots per streamition.");
  _app.add_option("-f,--file", bin_filename_list, "list of binary file: first file list of sorted id, then file list of corresponding frequency")->required();
  _app.add_option("-m,--method", method, "method to split slot. default is exp: to split exponential")->check(CLI::IsMember({"exp","num","bin"}));
  _app.add_option("-c,--cache", cache_rate, "cache rate. must be conjunction with -m bin");
  _app.add_option("-t,--threads", NUM_THREAD, "num of working threads");
  _app.add_option("--coe,--coefficient", coefficient, "coefficient. 1.4 by default");
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

  uint32_t num_stream = bin_filename_list.size() / 2;
  std::vector<uint32_t*> stream_id_list(num_stream);
  std::vector<float*> stream_freq_list(num_stream);
  struct stat st;
  stat(bin_filename_list[0].c_str(), &st);
  num_node = st.st_size / sizeof(uint32_t);
  // std::cerr << num_node << "\n";
  for (uint i = 0; i < num_stream; i++) {
    int fd_id = open(bin_filename_list[i].c_str(), O_RDONLY);
    int fd_freq = open(bin_filename_list[i + num_stream].c_str(), O_RDONLY);
    stream_id_list[i] = (uint32_t*)mmap(nullptr, num_node * sizeof(uint32_t), PROT_READ, MAP_PRIVATE, fd_id, 0);
    stream_freq_list[i] = (float*)mmap(nullptr, num_node * sizeof(float), PROT_READ, MAP_PRIVATE, fd_freq, 0);
  }

  ndarray<uint32_t> nid_to_rank({num_node, num_stream});
  ndarray<uint32_t> nid_to_slot({num_node, num_stream});
  ndarray<uint32_t> nid_to_block({num_node});

  ndarray<block_identifer> slot_array_to_full_block(std::vector<uint32_t>(num_stream, num_slot));
#pragma omp parallel for num_threads(NUM_THREAD)
  for (uint32_t slot_array_seq_id = 0; slot_array_seq_id < slot_array_to_full_block._num_elem; slot_array_seq_id++) {
    slot_array_to_full_block._data[slot_array_seq_id].current_block_id = slot_array_seq_id;
    if (method == "bin") {
      slot_array_to_full_block._data[slot_array_seq_id].set_ignore_limit();
    }
  }
  next_free_block.store(slot_array_to_full_block._num_elem);
  slot_array_to_full_block._data[slot_array_to_full_block._num_elem - 1].set_ignore_limit();


  // identify freq boundary of first slot
  for (uint i = 0; i < num_stream; i++) {
    if (alpha < stream_freq_list[i][(num_node - 1) / 100]) {
      alpha = stream_freq_list[i][(num_node - 1) / 100];
    }
  }

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */
  std::cerr << "mapping nid to rank...\n";
#pragma omp parallel for num_threads(NUM_THREAD)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list[stream_idx][orig_rank];
      nid_to_rank[nid][stream_idx].ref() = orig_rank;
    }
  }
#pragma omp parallel for num_threads(NUM_THREAD)
  for (uint32_t nid = 0; nid < num_node; nid++) {
    // for each nid, prepare a slot list
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
      double freq = stream_freq_list[stream_idx][orig_rank];
      int slot_id = freq_to_slot(freq, orig_rank);
      nid_to_slot[nid][stream_idx].ref() = slot_id;
    }
    // map the slot list to block
    nid_to_block[nid].ref() = slot_array_to_full_block[nid_to_slot[nid]].ref().add_node();
  }

  /**
   * Sum frequency & density of each block
   */
  std::cerr << "counting freq and density...\n";
  uint32_t total_num_blocks = next_free_block.load();
  ndarray<double> block_density_array({total_num_blocks});
  ndarray<double> block_freq_array({total_num_blocks, num_stream});
#pragma omp parallel for num_threads(NUM_THREAD)
  for (uint32_t thread_idx = 0; thread_idx < NUM_THREAD; thread_idx++) {
    for (uint32_t nid = 0; nid < num_node; nid++) {
      uint32_t block_id = nid_to_block[nid].ref();
      if (std::hash<uint64_t>()(block_id) % NUM_THREAD != thread_idx) {
        continue;
      }
      block_density_array[block_id].ref() += 1;
      for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
        uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
        float freq = stream_freq_list[stream_idx][orig_rank];
        block_freq_array[block_id][stream_idx].ref() += freq;
      }
    }
  }

  /**
   * Average the frequency for each block
   */
  std::cerr << "averaging freq and density...\n";
#pragma omp parallel for num_threads(NUM_THREAD)
  for (uint32_t block_id = 0; block_id < total_num_blocks; block_id++) {
    if (block_density_array[block_id].ref() == 0) continue; 
    for (uint32_t stream_id = 0; stream_id < num_stream; stream_id++) {
      block_freq_array[{block_id,stream_id}].ref() /= block_density_array[block_id].ref() ;
    }
    block_density_array[block_id].ref() *= 100/(double)num_node ;
    std::cout << block_density_array[block_id].ref() << " ";
  }
  std::cout << "\n";
  std::cerr << "writing to disk...\n";
  std::fstream ofs_density = std::fstream("density.bin", std::ios_base::binary | std::ios_base::out);
  std::fstream ofs_block_freq = std::fstream("block_freq.bin", std::ios_base::binary | std::ios_base::out);

  ofs_density.write((char*)block_density_array._data, sizeof(double) * block_density_array._num_elem);
  ofs_density.close();
  ofs_block_freq.write((char*)block_freq_array._data, sizeof(double)*block_freq_array._num_elem);
  ofs_block_freq.close();

}