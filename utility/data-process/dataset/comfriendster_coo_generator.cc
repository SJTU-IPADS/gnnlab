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
std::string output_dir = "/graph-learning/data-raw/com-friendster-bin/";

size_t num_threads = 48;
size_t num_nodes = 65608366;
uint32_t max_nodeid = 124836179;
size_t origin_num_edges = 1806067135;
size_t num_edges = origin_num_edges * 2;
size_t num_train_set = 1000000;
size_t num_test_set = 100000;
size_t num_valid_set = 200000;
size_t feat_dim = 300;
size_t num_class = 150;

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
    ctx.e_list.push_back({dst, src});
    ctx.e_cnt += 2;
    i = k + 1;
  }
}

void checkLoadedGraph(std::vector<ThreadCtx *> &thread_ctx) {
  size_t num_loaded_nodes = 0;
  size_t num_loaded_edges = 0;

  for (auto ctx : thread_ctx) {
    num_loaded_nodes += ctx->v_cnt;
    num_loaded_edges += ctx->e_cnt;
  }

  utility::Check(num_loaded_nodes == num_nodes, "error number of loaded nodes");
  utility::Check(num_loaded_edges == num_edges, "error number of loaded edges");
}

void threadPopulateHashtable(ThreadCtx &ctx,
                             std::vector<uint32_t> &o2n_hashtable,
                             uint32_t new_start) {
  for (size_t i = 0; i < ctx.v_cnt; i++) {
    o2n_hashtable[ctx.v_list[i]] = new_start + i;
  }
}

void threadMapEdges(ThreadCtx &ctx, std::vector<uint32_t> &o2n_hashtable,
                    std::vector<uint32_t> &coo_row,
                    std::vector<uint32_t> &coo_col, uint32_t new_start) {
  for (size_t i = 0; i < ctx.e_cnt; i++) {
    // swap src and dst to make a csc graph
    coo_row[new_start + i] = o2n_hashtable[ctx.e_list[i].first];
    coo_col[new_start + i] = o2n_hashtable[ctx.e_list[i].second];
  }
}

void joinThreads(std::vector<std::thread> &threads) {
  for (size_t i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  threads.clear();
}

void writeCooFile(const std::vector<uint32_t> &coo_row,
                  const std::vector<uint32_t> &coo_col) {
  std::string coo_row_path = output_dir + "coo_row.bin";
  std::string coo_col_path = output_dir + "coo_col.bin";
  std::ofstream ofs0(coo_row_path, std::ofstream::out | std::ofstream::binary |
                                       std::ofstream::trunc);
  std::ofstream ofs1(coo_col_path, std::ofstream::out | std::ofstream::binary |
                                       std::ofstream::trunc);

  ofs0.write((const char *)coo_row.data(), num_edges * sizeof(uint32_t));
  ofs1.write((const char *)coo_col.data(), num_edges * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
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

  joinThreads(threads);
  double read_file_time = t1.Passed();
  printf("read file %.4f\n", read_file_time);

  checkLoadedGraph(thread_ctx);

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
  }

  std::vector<uint32_t> o2n_hashtable(max_nodeid + 1);
  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(threadPopulateHashtable, std::ref(*thread_ctx[i]),
                         std::ref(o2n_hashtable), v_cnt_prefix_sum[i]);
  }

  joinThreads(threads);
  double populate_time = t2.Passed();
  printf("populate %.4f\n", populate_time);

  utility::Timer t3;
  std::vector<uint32_t> coo_row(num_edges);
  std::vector<uint32_t> coo_col(num_edges);

  for (size_t i = 0; i < num_threads; i++) {
    threads.emplace_back(threadMapEdges, std::ref(*thread_ctx[i]),
                         std::ref(o2n_hashtable), std::ref(coo_row),
                         std::ref(coo_col), e_cnt_prefix_sum[i]);
  }

  joinThreads(threads);
  double map_edges_time = t3.Passed();
  printf("map edges %.4f\n", map_edges_time);

  utility::Timer t4;
  writeCooFile(coo_row, coo_col);
  double write_coo_file_time = t3.Passed();
  printf("write coo file %.4f\n", write_coo_file_time);
}
