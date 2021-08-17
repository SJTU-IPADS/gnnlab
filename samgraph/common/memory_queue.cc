#include "memory_queue.h"

namespace samgraph {
namespace common {

MemoryQueue* MemoryQueue::_mq = nullptr;

MemoryQueue::MemoryQueue(std::string meta_memory_name) {
  _meta_memory_name = meta_memory_name;
  int fd = shm_open(_meta_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd != -1) { // first open
    int ret = ftruncate(fd, sizeof(QueueMetaData));
    CHECK_NE(ret, -1);
    _meta_data = reinterpret_cast<QueueMetaData*> (mmap(NULL, sizeof(QueueMetaData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK_NE(_meta_data, MAP_FAILED);
    _meta_data->Init();
  } else { // second open
    fd = shm_open(_meta_memory_name.c_str(), O_RDWR, 0);
    CHECK_NE(fd, -1);
    _meta_data = reinterpret_cast<QueueMetaData*> (mmap(NULL, sizeof(QueueMetaData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK_NE(_meta_data, MAP_FAILED);
  }
}

std::string MemoryQueue::Key2String(size_t key) {
  return "shared_memory_" + std::to_string(key);
}

MemoryQueue::~MemoryQueue() {
  int len = _meta_data->send_cnt;
  for (int i = 0; i < len; ++i) {
    std::string shared_memory_name = Key2String(i);
    shm_unlink(shared_memory_name.c_str());
  }
  shm_unlink(RunConfig::shared_meta_path.c_str());
}

void MemoryQueue::Create() {
  _mq = new MemoryQueue(RunConfig::shared_meta_path);
}

void MemoryQueue::Destory() {
  delete _mq;
  _mq = nullptr;
}

int MemoryQueue::Send(void* data, size_t size) {
  auto key = _meta_data->Put();
  while (key >= _meta_data->recv_cnt + _meta_data->max_size) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  std::string shared_memory_name = Key2String(key);

  int fd = shm_open(shared_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1);

  int ret = ftruncate(fd, size);
  CHECK_NE(ret, -1);

  void* shared_memory = reinterpret_cast<void*> (mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  CHECK_NE(shared_memory, MAP_FAILED);

  std::memcpy(shared_memory, data, size);
  _meta_data->SemPost(key);

  return size;
}

std::shared_ptr<SharedData> MemoryQueue::Recv() {
  while (_meta_data->recv_cnt == _meta_data->send_cnt) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    // return std::make_shared<SharedData>(nullptr, -1, "null");
  }
  auto key = _meta_data->Get();
  std::string shared_memory_name = Key2String(key);

  _meta_data->SemWait(key);
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

  return ret;
}

}  // namespace common
}  // namespace samgraph

