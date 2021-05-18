#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::string dataset = "com-friendster";
size_t num_nodes = 65608366;

std::string indptr_filepath =
    "/graph-learning/samgraph/" + dataset + "/indptr.bin";
std::string indices_filepath =
    "/graph-learning/samgraph/" + dataset + "/indices.bin";
std::string out0_filepath =
    "/graph-learning/samgraph/" + dataset + "/degrees.txt";
std::string out1_filepath =
    "/graph-learning/samgraph/" + dataset + "/out_degrees.bin";
std::string out2_filepath =
    "/graph-learning/samgraph/" + dataset + "/in_degrees.bin";

int main() {
  int fd;
  struct stat st;
  size_t nbytes;

  uint32_t *indptr;
  uint32_t *indices;

  size_t num_threads = 24;
  omp_set_num_threads(num_threads);

  fd = open(indptr_filepath.c_str(), O_RDONLY, 0);
  stat(indptr_filepath.c_str(), &st);
  nbytes = st.st_size;

  if (nbytes == 0) {
    std::cout << "Reading file error: " << indptr_filepath << std::endl;
    exit(1);
  }

  indptr =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(indptr, nbytes);
  close(fd);

  fd = open(indices_filepath.c_str(), O_RDONLY, 0);
  stat(indices_filepath.c_str(), &st);
  nbytes = st.st_size;
  if (nbytes == 0) {
    std::cout << "Reading file error: " << indices_filepath << std::endl;
    exit(1);
  }

  indices =
      (uint32_t *)mmap(NULL, nbytes, PROT_READ, MAP_SHARED | MAP_FILE, fd, 0);
  mlock(indices, nbytes);
  close(fd);

  std::vector<uint32_t> out_degrees(num_nodes, 0);
  std::vector<uint32_t> in_degrees(num_nodes, 0);

  std::vector<std::vector<uint32_t>> in_degrees_per_thread(
      num_threads, std::vector<uint32_t>(num_nodes, 0));

#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    uint32_t len = indptr[i + 1] - indptr[i];
    uint32_t off = indptr[i];
    out_degrees[i] = len;

    uint32_t thread_idx = omp_get_thread_num();
    for (uint32_t k = 0; k < len; k++) {
      in_degrees_per_thread[thread_idx][indices[off + k]]++;
    }
  }
#pragma omp parallel for
  for (uint32_t i = 0; i < num_nodes; i++) {
    for (uint32_t k = 0; k < num_threads; k++) {
      in_degrees[i] += in_degrees_per_thread[k][i];
    }
  }

  std::ofstream ofs0(out0_filepath, std::ofstream::out | std::ofstream::trunc);
  std::ofstream ofs1(out1_filepath, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);
  std::ofstream ofs2(out2_filepath, std::ofstream::out | std::ofstream::binary |
                                        std::ofstream::trunc);

  for (uint32_t i = 0; i < num_nodes; i++) {
    ofs0 << i << " " << out_degrees[i] << " " << in_degrees[i] << "\n";
  }

  ofs1.write((const char *)out_degrees.data(),
             out_degrees.size() * sizeof(uint32_t));
  ofs2.write((const char *)in_degrees.data(),
             in_degrees.size() * sizeof(uint32_t));

  ofs0.close();
  ofs1.close();
  ofs2.close();
}
