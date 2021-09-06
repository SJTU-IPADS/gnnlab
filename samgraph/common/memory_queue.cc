#include "memory_queue.h"

namespace samgraph {
namespace common {

MemoryQueue* MemoryQueue::_mq = nullptr;

MemoryQueue::MemoryQueue(std::string meta_memory_name, size_t mq_nbytes) {
  _meta_memory_name = meta_memory_name;
  size_t meta_nbytes = (sizeof(QueueMetaData) + mq_nbytes * mq_size);
  int fd = shm_open(_meta_memory_name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd != -1) { // first open
    int ret = ftruncate(fd, meta_nbytes);
    CHECK_NE(ret, -1);
    _meta_data = reinterpret_cast<QueueMetaData*> (mmap(NULL, meta_nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK_NE(_meta_data, MAP_FAILED);
    _meta_data->Init(mq_nbytes);
  } else { // second open
    fd = shm_open(_meta_memory_name.c_str(), O_RDWR, 0);
    CHECK_NE(fd, -1);
    _meta_data = reinterpret_cast<QueueMetaData*> (mmap(NULL, meta_nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK_NE(_meta_data, MAP_FAILED);
  }
  _prefix = meta_memory_name;
  LOG(INFO) << "MemoryQueue initialized with prefix name: " << _prefix;
}

std::string MemoryQueue::Key2String(size_t key) {
  // return "shared_memory_" + std::to_string(key);
  return _prefix + std::to_string(key);
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
  LOG(FATAL) << "Can not be used now!";
  _mq = new MemoryQueue(RunConfig::shared_meta_path, 1024);
}

void MemoryQueue::Destory() {
  LOG(FATAL) << "Can not be used now!";
  delete _mq;
  _mq = nullptr;
}

int MemoryQueue::Send(void* data, size_t size) {
  auto key = _meta_data->Put();
  while (key >= _meta_data->recv_cnt + _meta_data->max_size) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
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

void* MemoryQueue::Recv() {
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

  return _meta_data->GetData(key);
}

}  // namespace common
}  // namespace samgraph

