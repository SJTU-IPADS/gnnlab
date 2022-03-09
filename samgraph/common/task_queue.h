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

#ifndef SAMGRAPH_TASK_QUEUE_H
#define SAMGRAPH_TASK_QUEUE_H

#include <memory>
#include <mutex>
#include <vector>

#include "common.h"
#include "memory_queue.h"

namespace samgraph {
namespace common {

class TaskQueue {
 public:
  TaskQueue(size_t max_len);
  virtual ~TaskQueue() {};

  void AddTask(std::shared_ptr<Task>);
  std::shared_ptr<Task> GetTask();
  bool Full();
  size_t PendingLength();

 private:
  std::vector<std::shared_ptr<Task>> _q;
  std::mutex _mutex;
  size_t _max_len;
};

class MessageTaskQueue : public TaskQueue {
 public:
  MessageTaskQueue(size_t max_len);
  void PinMemory() { _mq->PinMemory(); }
  void Send(std::shared_ptr<Task>);
  std::shared_ptr<Task> Recv();

 private:
  std::shared_ptr<MemoryQueue> _mq;
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_TASK_QUEUE_H
