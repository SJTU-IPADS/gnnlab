#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#ifdef __linux__
#include <parallel/algorithm>
#endif
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "common/utils.h"

namespace {

std::string raw_data_dir = "/graph-learning/data-raw/com-friendster/";
std::string output_dir = "/graph-learning/samgraph/com-friendster/";

size_t num_threads = 48;
size_t num_nodes = 65608366;
uint32_t max_nodeid = 124836179;
size_t num_edges = 1806067135;
size_t num_train_set = 1000000;
size_t num_test_set = 100000;
size_t num_valid_set = 200000;
size_t feat_dim = 256;
size_t num_class = 172;

struct ThreadCtx {
  int thread_idx;

  std::vector<uint32_t> v_list;
  std::vector<std::pair<uint32_t, uint32_t>> e_list;
  size_t v_cnt;
  size_t e_cnt;

  ThreadCtx(int thread_idx, size_t v_sz, size_t e_sz)
      : thread_idx(thread_idx), v_cnt(0), e_cnt(0) {
    v_list.reserve(v_sz);
    e_list.reserve(e_sz);
  }
};

struct File {
  char *data;
  size_t nbytes;
};

struct RawGraph {
  File vfile;
  File efile;
};

RawGraph getMMapFile() {
  int fd;
  struct stat st;
  size_t vbytes;
  size_t ebytes;
  void *vdata;
  void *edata;
  std::string vfile_path = raw_data_dir + "com-friendster.v";
  std::string efile_path = raw_data_dir + "com-friendster.e";

  fd = open(vfile_path.c_str(), O_RDONLY, 0);
  stat(vfile_path.c_str(), &st);
  vbytes = st.st_size;
  vdata = mmap(NULL, vbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(vdata, vbytes);
  close(fd);

  fd = open(efile_path.c_str(), O_RDONLY, 0);
  stat(efile_path.c_str(), &st);
  ebytes = st.st_size;
  edata = mmap(NULL, ebytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(edata, ebytes);
  close(fd);

  return {{(char *)vdata, vbytes}, {(char *)edata, ebytes}};
}

size_t locateLineStart(File f, size_t off) {
  if (off == 0) return 0;

  if (off >= f.nbytes) return f.nbytes - 1;
  while (off < (f.nbytes - 1) && f.data[off] != '\n') off++;
  return off == (f.nbytes - 1) ? off : off + 1;
}

size_t locateLineEnd(File f, size_t off) {
  if (off >= f.nbytes) return f.nbytes - 1;
  while (off < (f.nbytes - 1) && f.data[off] != '\n') off++;
  return off;
}

void threadLoadGraph(ThreadCtx &ctx, RawGraph &raw_graph, size_t v_start,
                     size_t v_end, size_t e_start, size_t e_end) {
  std::string str;
  auto vfile = raw_graph.vfile;
  auto efile = raw_graph.efile;

  for (size_t i = v_start; i < v_end;) {
    size_t k = i;
    while (vfile.data[k] != '\n') {
      k++;
    }

    str = std::string(vfile.data + i, k - i);
    uint32_t vid = std::stoi(str);
    ctx.v_list.push_back(vid);
    ctx.v_cnt++;
    i = k + 1;
  }

  for (size_t i = e_start; i < e_end;) {
    size_t k = i;
    while (efile.data[k] != ' ') {
      k++;
    }

    str = std::string(efile.data + i, k - i);
    uint32_t src = std::stoi(str);

    k++;
    i = k;

    while (efile.data[k] != '\n') {
      k++;
    }

    str = std::string(efile.data + i, k - i);
    uint32_t dst = std::stoi(str);

    ctx.e_list.push_back({src, dst});
    ctx.e_cnt++;
    i = k + 1;
  }
}

void threadPopulateHashtable(ThreadCtx &ctx,
                             std::vector<uint32_t> &o2n_hashtable,
                             uint32_t new_start) {
  for (size_t i = 0; i < ctx.v_cnt; i++) {
    o2n_hashtable[ctx.v_list[i]] = new_start + i;
  }
}

void threadMapEdges(ThreadCtx &ctx, std::vector<uint32_t> &o2n_hashtable,
                    std::vector<std::pair<uint32_t, uint32_t>> &new_edge_list,
                    uint32_t new_start) {
  for (size_t i = 0; i < ctx.e_cnt; i++) {
    // swap src and dst to make a csc graph
    new_edge_list[new_start + i] = {o2n_hashtable[ctx.e_list[i].second],
                                    o2n_hashtable[ctx.e_list[i].first]};
  }
}

void generateNodeSet(std::vector<uint32_t> &indptr) {
  std::vector<bool> bitmap(num_nodes, false);
  std::vector<uint32_t> train_set;
  std::vector<uint32_t> test_set;
  std::vector<uint32_t> valid_set;

  train_set.reserve(num_train_set);
  test_set.reserve(num_test_set);
  valid_set.reserve(num_valid_set);

  std::mt19937 generator;
  std::uniform_int_distribution<uint32_t> distribution(0, num_nodes - 1);

  while (train_set.size() < num_train_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      train_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (test_set.size() < num_test_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      test_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  while (valid_set.size() < num_valid_set) {
    uint32_t nodeid = distribution(generator);
    if (indptr[nodeid + 1] - indptr[nodeid] > 0 && !bitmap[nodeid]) {
      valid_set.push_back(nodeid);
      bitmap[nodeid] = true;
    }
  }

  std::string train_set_path = output_dir + "train_set.bin";
  std::string valid_set_path = output_dir + "valid_set.bin";
  std::string test_set_path = output_dir + "test_set.bin";

  std::ofstream ofs0(train_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs1(valid_set_path, std::ofstream::out |
                                         std::ofstream::binary |
                                         std::ofstream::trunc);
  std::ofstream ofs2(test_set_path, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);
  ofs0.write((const char *)train_set.data(),
             train_set.size() * sizeof(uint32_t));
  ofs1.write((const char *)valid_set.data(),
             valid_set.size() * sizeof(uint32_t));
  ofs2.write((const char *)test_set.data(), test_set.size() * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
  ofs2.close();
}

void writeMetaFile() {
  std::string meta_path = output_dir + "meta.txt";
  std::ofstream ofs(meta_path, std::ofstream::out | std::ofstream::trunc);
  std::string output = "NUM_NODE      " + std::to_string(num_nodes) +
                       "\nNUM_EDGE      " + std::to_string(num_edges) +
                       "\nFEAT_DIM      " + std::to_string(feat_dim) +
                       "\nNUM_CLASS     " + std::to_string(num_class) +
                       "\nNUM_TRAIN_SET " + std::to_string(num_train_set) +
                       "\nNUM_VALID_SET " + std::to_string(num_valid_set) +
                       "\nNUM_TEST_SET  " + std::to_string(num_test_set) + "\n";
  ofs << output;
  ofs.close();
}

void writeCSRToFile(std::vector<uint32_t> &indptr,
                    std::vector<uint32_t> &indices) {
  std::string indptr_path = output_dir + "indptr.bin";
  std::string indices_path = output_dir + "indices.bin";
  std::ofstream ofs0(indptr_path, std::ofstream::out | std::ofstream::binary |
                                      std::ofstream::trunc);
  std::ofstream ofs1(indices_path, std::ofstream::out | std::ofstream::binary |
                                       std::ofstream::trunc);

  ofs0.write((const char *)indptr.data(), (num_nodes + 1) * sizeof(uint32_t));
  ofs1.write((const char *)indices.data(), num_edges * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
}

void JoinThreads(std::vector<std::thread> &threads) {
  for (size_t i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  threads.clear();
}

}  // namespace

int main() {
  utility::Timer t0;
  auto raw_graph = getMMapFile();
  double mmap_time = t0.Passed();
  printf("mmap %.4f\n", mmap_time);

  utility::Timer t1;
  omp_set_num_threads(num_threads);
  std::vector<ThreadCtx *> thread_ctx;
  std::vector<std::thread> threads;

  // Load raw graph data using mmap
  size_t avg_vertex_num = (num_nodes + num_threads - 1) / num_threads;
  size_t avg_edge_num = (num_edges + num_threads - 1) / num_threads;

  size_t vfile_partition_nbytes =
      (raw_graph.vfile.nbytes + num_threads - 1) / num_threads;
  size_t efile_partition_nbytes =
      (raw_graph.efile.nbytes + num_threads - 1) / num_threads;

  auto vfile = raw_graph.vfile;
  auto efile = raw_graph.efile;
  for (size_t i = 0; i < num_threads; i++) {
    thread_ctx.push_back(new ThreadCtx(i, avg_vertex_num, avg_edge_num));
    size_t v_start = locateLineStart(vfile, i * vfile_partition_nbytes);
    size_t v_end = locateLineEnd(vfile, (i + 1) * vfile_partition_nbytes);
    size_t e_start = locateLineStart(efile, i * efile_partition_nbytes);
    size_t e_end = locateLineEnd(efile, (i + 1) * efile_partition_nbytes);

    threads.emplace_back(threadLoadGraph, std::ref(*thread_ctx[i]),
                         std::ref(raw_graph), v_start, v_end, e_start, e_end);
  }

  JoinThreads(threads);
  double read_file_time = t1.Passed();
  printf("read file %.4f\n", read_file_time);

  // Map nodes and edges to a new space
  utility::Timer t2;
  std::vector<uint32_t> v_cnt_prefix_sum(num_threads, 0);
  std::vector<uint32_t> e_cnt_prefix_sum(num_threads, 0);
  uint32_t v_sum = 0;
  uint32_t e_sum = 0;
  for (size_t i = 0; i < num_threads; i++) {
    v_cnt_prefix_sum[i] = v_sum;
    v_sum += thread_ctx[i]->v_cnt;

    e_cnt_prefix_sum[i] = e_sum;
    e_sum += thread_ctx[i]->e_cnt;
    // Add self-loop
    e_sum += thread_ctx[i]->v_cnt;
  }

  std::vector<uint32_t> o2n_hashtable(max_nodeid + 1);
  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(threadPopulateHashtable, std::ref(*thread_ctx[i]),
                         std::ref(o2n_hashtable), v_cnt_prefix_sum[i]);
  }

  JoinThreads(threads);
  double populate_time = t2.Passed();
  printf("populate %.4f\n", populate_time);

  utility::Timer t3;
  std::vector<std::pair<uint32_t, uint32_t>> new_edge_list(num_edges);

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(threadMapEdges, std::ref(*thread_ctx[i]),
                         std::ref(o2n_hashtable), std::ref(new_edge_list),
                         e_cnt_prefix_sum[i]);
  }

  JoinThreads(threads);
  double map_edges_time = t3.Passed();
  printf("map edges %.4f\n", map_edges_time);

  utility::Timer t4;
#ifdef __linux__
  __gnu_parallel::sort(new_edge_list.begin(), new_edge_list.end());
#else
  std::sort(new_edge_list.begin(), new_edge_list.end());
#endif

  std::vector<uint32_t> indptr(num_nodes + 1, 0);
  std::vector<uint32_t> indices(num_edges);

  std::vector<std::vector<uint32_t>> indptr_cnt(
      num_threads, std::vector<uint32_t>(num_nodes, 0));

#pragma omp parallel for
  for (size_t i = 0; i < num_edges; i++) {
    int thread_idx = omp_get_thread_num();
    indptr_cnt[thread_idx][new_edge_list[i].first]++;
    indices[i] = new_edge_list[i].second;
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_nodes; i++) {
    for (size_t j = 0; j < num_threads; j++) {
      indptr[i] += indptr_cnt[j][i];
    }
  }

  uint32_t indptr_sum = 0;
  for (size_t i = 0; i < num_nodes; i++) {
    uint32_t tmp = indptr[i];
    indptr[i] = indptr_sum;
    indptr_sum += tmp;
  }
  indptr[num_nodes] = indptr_sum;

  double to_csr_time = t4.Passed();
  printf("to csr %.4f \n", to_csr_time);

  utility::Timer t5;
  writeCSRToFile(indptr, indices);
  double write_file_time = t5.Passed();

  printf("write file %.4f \n", write_file_time);

  utility::Timer t6;
  generateNodeSet(indptr);
  double generate_nodeset_time = t6.Passed();

  writeMetaFile();
}
