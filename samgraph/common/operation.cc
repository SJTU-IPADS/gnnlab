#include <string>
#include <vector>

#include "operation.h"
#include "logging.h"
#include "engine.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_init(const char *path, int sample_device, int train_device, size_t batch_size,
                   int *fanout, size_t num_fanout, int num_epoch, int engine_type) {
    SamGraphEngine::CreateEngine(static_cast<EngineType>(engine_type));
    SamGraphEngine::Get()->Init(path, sample_device, train_device,  batch_size,
                         std::vector<int>(fanout, fanout + num_fanout), num_epoch);
    SAM_LOG(INFO) << "SamGraph has been initialied successfully";
    return;
}

void samgraph_start() {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());
    SamGraphEngine::Get()->Start();
    SAM_LOG(INFO) << "SamGraph has been started successfully";
}

int samgraph_num_epoch() {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());
    return SamGraphEngine::Get()->GetNumEpoch();
}

size_t samgraph_num_step_per_epoch() {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());
    return SamGraphEngine::Get()->GetNumStep();
}

size_t samgraph_dataset_num_class() {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());
    return SamGraphEngine::Get()->GetGraphDataset()->num_class;
}

size_t samgraph_dataset_num_feat_dim() {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());
    return SamGraphEngine::Get()->GetGraphDataset()->feat->shape().at(1);
}

uint64_t samgraph_get_next_batch(int epoch, int step) {
    SAM_CHECK(SamGraphEngine::Get()->IsInitialized() && !SamGraphEngine::Get()->IsShutdown());

    uint64_t key = encodeBatchKey(epoch, step);
    SAM_LOG(DEBUG) << "samgraph_get_next_batch encodeKey with epoch " << epoch << " step " << step << " and key " << key;
    
    // RunSingleLoopOnce();

    auto graph = SamGraphEngine::Get()->GetGraphPool()->GetGraphBatch(key);

    SAM_LOG(DEBUG) << "Get next batch with key " << key;
    SamGraphEngine::Get()->SetGraphBatch(graph);

    return key;
}

uint64_t samgraph_get_graph_key(uint64_t batch_key, int graph_id) {
    return encodeGraphID(batch_key, graph_id);
}

size_t samgraph_get_graph_num_row(uint64_t key) {
    int layer_idx = decodeGraphID(key);

    auto batch = SamGraphEngine::Get()->GetGraphBatch();

    return batch->output_graph[layer_idx]->num_row;
}

size_t samgraph_get_graph_num_col(uint64_t key) {
    int layer_idx = decodeGraphID(key);

    auto batch = SamGraphEngine::Get()->GetGraphBatch();

    return batch->output_graph[layer_idx]->num_column;
}

size_t samgraph_get_graph_num_edge(uint64_t key) {
    int layer_idx = decodeGraphID(key);

    auto batch = SamGraphEngine::Get()->GetGraphBatch();

    return batch->output_graph[layer_idx]->num_edge;
}

void samgraph_shutdown() {
    SamGraphEngine::Get()->Shutdown();
    SAM_LOG(DEBUG) << "SamGraph has been completely shutdown now";
    return;
}

}

} // namespace common
} // namespace samgraph