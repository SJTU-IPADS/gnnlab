#ifndef SAMGRAPH_OPERATIONS_H
#define SAMGRAPH_OPERATIONS_H

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char*path, int sam_device, int train_device, int batch_size, int *fanout, int num_fanout, int num_epoch);

void samgraph_shutdown();

}

} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_OPERATIONS_H
