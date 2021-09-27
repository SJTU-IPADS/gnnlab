#include "memory_queue.h"

#include <cuda_runtime.h>
#include <sys/mman.h>

namespace samgraph {
namespace common {

SharedData::~SharedData() {
  // send the release semaphore
  _meta->ReleasePost(_key);
};

MemoryQueue* MemoryQueue::_mq = nullptr;

MemoryQueue::MemoryQueue(size_t mq_nbytes) {
  _meta_size = (sizeof(QueueMetaData) + mq_nbytes * mq_size);
  _meta_data = reinterpret_cast<QueueMetaData*>(
      mmap(NULL, _meta_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED,
           -1, 0));
  mlock(_meta_data, _meta_size);
  CHECK_NE(_meta_data, MAP_FAILED);
  _meta_data->Init(mq_nbytes);
  LOG(INFO) << "MemoryQueue initialized";
}

MemoryQueue::~MemoryQueue() {
   munmap(_meta_data, _meta_size);
}

void MemoryQueue::PinMemory() {
  CUDA_CALL(cudaHostRegister(_meta_data, _meta_size, cudaHostRegisterPortable));
}

void MemoryQueue::Create() {
  LOG(FATAL) << "Can not be used now!";
  _mq = new MemoryQueue(1024);
}

void MemoryQueue::Destory() {
  LOG(FATAL) << "Can not be used now!";
  delete _mq;
  _mq = nullptr;
}

void* MemoryQueue::GetPtr(size_t &key) {
  key = _meta_data->Put();
  while (key >= _meta_data->recv_cnt + _meta_data->max_size) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  _meta_data->ReleaseWait(key);
  void* shared_memory = _meta_data->GetData(key);
  return shared_memory;
}

void MemoryQueue::SimpleSend(size_t key) {
  _meta_data->SemPost(key);
  LOG(DEBUG) << "MemoryQueue Send with key: " << key;
}

int MemoryQueue::Send(void* data, size_t size) {
  auto key = _meta_data->Put();
  while (key >= _meta_data->recv_cnt + _meta_data->max_size) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  _meta_data->ReleaseWait(key);
  /*
  std::string shared_memory_name = Key2String(key);

  int fd = shm_open(shared_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1);

  int ret = ftruncate(fd, size);
  CHECK_NE(ret, -1);

  void* shared_memory = reinterpret_cast<void*> (mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  CHECK_NE(shared_memory, MAP_FAILED);
  */
  void* shared_memory = _meta_data->GetData(key);

  std::memcpy(shared_memory, data, size);
  _meta_data->SemPost(key);
  LOG(DEBUG) << "MemoryQueue Send with key: " << key;

  return size;
}

std::shared_ptr<SharedData> MemoryQueue::Recv() {
  LOG(DEBUG) << "MemoryQueue Recv:"
             << " recv_cnt in meta_data is " << _meta_data->recv_cnt
             << ", send_cnt in meta_data is " << _meta_data->send_cnt;
  while (_meta_data->recv_cnt == _meta_data->send_cnt) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return std::make_shared<SharedData>(nullptr, -1, "null");
  }
  auto key = _meta_data->Get();
  LOG(DEBUG) << "MemoryQueue Recv with key: " << key;
  // std::string shared_memory_name = Key2String(key);

  _meta_data->SemWait(key);
  /*
  int fd = shm_open(shared_memory_name.c_str(), O_RDWR, 0);
  CHECK_NE(fd, -1);
  if (fd == -1) {
    return std::make_shared<SharedData>(nullptr, -1, "null");
  }

  struct stat st;
  CHECK_NE(fstat(fd, &st), -1);
  size_t size = st.st_size;

  void* shared_memory = reinterpret_cast<void*> (mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  CHECK_NE(shared_memory, MAP_FAILED);

  std::shared_ptr<SharedData> ret = std::make_shared<SharedData>(shared_memory, size, shared_memory_name);
  LOG(DEBUG) << "MemoryQueue Recv success with key: " << key;

  return ret;
  */
  auto ptr = _meta_data->GetData(key);
  return std::make_shared<SharedData>(ptr, key, _meta_data);
}

}  // namespace common
}  // namespace samgraph

