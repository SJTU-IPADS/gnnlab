#pragma once
namespace samgraph {
namespace sam_backend {

extern "C" {
void samgraph_backend_init_model();

void samgraph_backend_train_current_batch();
}

} // namespace sam_backend
} // namespace samgraph