#ifndef SAMGRAPH_READY_TABLE_H
#define SAMGRAPH_READY_TABLE_H

#include <unordered_map>
#include <mutex>
#include <string>

namespace samgraph {
namespace common {

class ReadyTable {
 public:
  ReadyTable(int ready_count, const char *name) : _ready_count(ready_count), _table_name(name) {}

  bool IsKeyReady(uint64_t key);
  int AddReadyCount(uint64_t key);
  void ClearReadyCount(uint64_t key);

 private:
  std::unordered_map<uint64_t, int> _ready_table;
  std::mutex _table_mutex;
  int _ready_count;
  std::string _table_name;
};

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_READY_TABLE_H
