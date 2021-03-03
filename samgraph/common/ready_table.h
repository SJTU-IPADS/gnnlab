#ifndef SAMGRAPH_READY_TABLE_H
#define SAMGRAPH_READY_TABLE_H

#include <mutex>
#include <thread>
#include <unordered_map>

namespace samgraph {
namespace common {

class ReadyTable {
 public:
  ReadyTable(int ready_count, const char* name) {
    _ready_count = ready_count;
    _table_name = std::string(name);
  }
  // methods to access or modify the _ready_table
  bool IsKeyReady(uint64_t key);
  int AddReadyCount(uint64_t key);
  int SetReadyCount(uint64_t key, int cnt);
  void ClearReadyCount(uint64_t key);

 private:
  // (key, ready_signal_count) pair, only valid for root device
  std::unordered_map<uint64_t, int> _ready_table;
  // use this mutex to access/modify the _ready_table
  std::mutex _table_mutex;
  int _ready_count;
  std::string _table_name;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_READY_TABLE_H