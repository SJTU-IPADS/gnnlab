#ifndef SAMGRAPH_THREAD_LOCAL_H
#define SAMGRAPH_THREAD_LOCAL_H

namespace samgraph {
namespace common {

template <typename T>
class ThreadLocalStore {
 public:
  static T* Get() {
    static thread_local T inst;
    return &inst;
  }
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_THREAD_LOCAL_H
