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

class SharedData {
 private:
  void* _data;
  size_t _size;
  std::string _shared_name;
 public:
  SharedData(void* data, size_t size, std::string name) : _data(data), _size(size), _shared_name(name) {};
  const void* Data() { return _data; }
  size_t Size() { return _size; }
  ~SharedData() {
    shm_unlink(_shared_name.c_str());
  };
};

template <typename T, T N>
struct MQ_MetaData {
  T send_cnt;
  T recv_cnt;
  T max_size;
  sem_t sem_list[N];
  void Init() {
    send_cnt = 0; recv_cnt = 0; max_size = N;
    for (int i = 0; i < max_size; ++i) {
      sem_init(sem_list + i, 1, 0);
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
};

using QueueMetaData = MQ_MetaData<size_t, 1024>;

class MemoryQueue {
 public:
  MemoryQueue(std::string meta_memory_name);
  ~MemoryQueue();
  static MemoryQueue* Get() { return _mq; }
  static void Create();
  static void Destory();
  int Send(void* data, size_t size);
  std::shared_ptr<SharedData> Recv();
 private:
  static MemoryQueue *_mq;
  QueueMetaData* _meta_data;
  std::string _meta_memory_name;
  std::string Key2String(size_t key);
  std::string prefix;
};

}  // namespace common
}  // namespace samgraph

#endif // SAMGRAPH_MEMORY_QUEUE_H
