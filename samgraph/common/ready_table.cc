#include "ready_table.h"
#include "logging.h"

namespace samgraph {
namespace common {

// below are methods for accessing/modifying the _ready_table
bool ReadyTable::IsKeyReady(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  return _ready_table[key] == (_ready_count);
}

int ReadyTable::AddReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  SAM_CHECK_LT(_ready_table[key], _ready_count)
      << _table_name << ": " << _ready_table[key] << ", " << (_ready_count);
  return ++_ready_table[key];
}

int ReadyTable::SetReadyCount(uint64_t key, int cnt) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = cnt;
}

void ReadyTable::ClearReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = 0;
}

} // namespace common
} // namespace samgraph