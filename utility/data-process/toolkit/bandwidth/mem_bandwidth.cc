/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <assert.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "common/utils.h"

using namespace utility;

int main() {
  size_t n_iters = 10;
  size_t mem_size = 1 * 1024 * 1024 * 1024;
  char *origin_data, *shm_data, *norm_mem_data;

  shm_data =
      (char *)mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_SHARED | 0x40000 /*MAP_HUGETLB*/, -1, 0);
  mlock(shm_data, mem_size);

  origin_data = (char *)malloc(mem_size);
  norm_mem_data = (char *)malloc(mem_size);

  double norm_time = 0;
  double shm_time = 0;

  size_t copy_size = 30 * 1024 * 1024;
  size_t current_offset_0 = 0;
  size_t current_offset_1 = mem_size - copy_size;
  for (int i = 0; i < n_iters; i++) {
    Timer t0;
    memcpy(norm_mem_data + current_offset_0, origin_data + current_offset_0,
           copy_size);
    norm_time += t0.Passed();

    Timer t1;
    memcpy(shm_data + current_offset_1, origin_data + current_offset_1,
           copy_size);
    shm_time += t1.Passed();
  }

  std::cout << "normal: " << norm_time / n_iters
            << " | shm: " << shm_time / n_iters << std::endl;
}