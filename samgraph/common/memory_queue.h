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

#pragma once
#ifndef SAMGRAPH_MEMORY_QUEUE_H
#define SAMGRAPH_MEMORY_QUEUE_H

#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <semaphore.h>

#include <iostream>
#include <memory>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>

#include "logging.h"
#include "run_config.h"

namespace samgraph {
namespace common {

template <typename T>
struct MQ_MetaData;

// constexpr size_t mq_size = 1200;

using QueueMetaData = MQ_MetaData<size_t>;


class SharedData {
 private:
  void* _data;
  size_t _key;
  QueueMetaData *_meta;
 public:
  ~SharedData();
  SharedData() { CHECK(0); };
  SharedData(void* data, size_t key, QueueMetaData *meta) :
    _data(data), _key(key), _meta(meta) {};
  const void* Data() { return _data; }
};

template <typename T>
struct MQ_MetaData {
  T send_cnt;
  T recv_cnt;
  T max_size;
  T mq_nbytes;
  sem_t *sem_list, *release_list;
  char *data;
  // sem_t sem_list[N];
  // sem_t release_list[N];
  // char data[0];
  static size_t GetRequiredNBytes(T mq_nbytes_t, T max_size_t) {
    return sizeof(MQ_MetaData<T>) + sizeof(sem_t) * max_size_t * 2 + mq_nbytes_t * max_size_t;
  }
  void Init(T mq_nbytes_t, T max_size_t) {
    send_cnt = 0; recv_cnt = 0; max_size = max_size_t;
    mq_nbytes = mq_nbytes_t;
    sem_list = (sem_t*)(&this[1]);
    release_list = &sem_list[max_size_t];
    data = (char*)&release_list[max_size_t];

    for (T i = 0; i < max_size; ++i) {
      sem_init(sem_list + i, 1, 0);
    }
    for (T i = 0; i < max_size; ++i) {
      sem_init(release_list + i, 1, 1);
    }
  }
  T Put() {
    return __sync_fetch_and_add(&send_cnt, 1);
  }
  T Get() {
    return __sync_fetch_and_add(&recv_cnt, 1);
  }
  int SemWait(T key) {
    int err = sem_wait(sem_list + (key % max_size));
    CHECK_NE(err, -1);
    return err;
  }
  int SemPost(T key) {
    int err = sem_post(sem_list + (key % max_size));
    CHECK_NE(err, -1);
    return err;
  }
  int ReleaseWait(T key) {
    int err = sem_wait(release_list + (key % max_size));
    CHECK_NE(err, -1);
    return err;
  }
  int ReleasePost(T key) {
    int err = sem_post(release_list + (key % max_size));
    CHECK_NE(err, -1);
    return err;
  }
  void* GetData(T key) {
    T pos = (key % max_size);
    return static_cast<void *>(data + (pos * mq_nbytes));
  }
};

class MemoryQueue {
 public:
  MemoryQueue(size_t mq_nbytes, size_t mq_size);
  ~MemoryQueue();
  static MemoryQueue* Get() { return _mq; }
  static void Create();
  static void Destory();
  void PinMemory();
  int Send(void* data, size_t size);
  std::shared_ptr<SharedData>  Recv();
  void* GetPtr(size_t &key);
  void  SimpleSend(size_t key);
 private:
  static MemoryQueue *_mq;
  QueueMetaData* _meta_data;
  size_t _meta_size;
};

}  // namespace common
}  // namespace samgraph

#endif // SAMGRAPH_MEMORY_QUEUE_H
