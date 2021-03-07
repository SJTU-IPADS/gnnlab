#include <thread>
#include <chrono>

#include <cuda_runtime.h>

#include "loops.h"
#include "engine.h"
#include "cuda_sampling.h"

namespace samgraph {
namespace common {

bool RunHostPermutateLoopOnce() {
    auto next_op = ID_COPYH2D;
    auto next_q = SamGraphEngine::GetTaskQueue(next_op);

    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto p = SamGraphEngine::GetRandomPermutation();
    auto batch = p->GetBatch();

    if (batch) {
        // Create task entry
        auto task = std::make_shared<TaskEntry>();
        task->key = encodeKey(p->cur_epoch(), p->cur_batch());
        task->train_nodes = batch;

        next_q->AddTask(task);
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunIdCopyHost2DeviceLoopOnce() {
    auto next_op = DEV_SAMPLE;
    auto next_q = SamGraphEngine::GetTaskQueue(next_op);
    
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = ID_COPYH2D;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();
    
    if (task) {
        auto nodes = task->train_nodes;
        auto id_copy_h2d_stream = SamGraphEngine::GetIdCopyHost2DeviceStream();
        auto device = SamGraphEngine::GetSampleDevice();

        auto d_nodes = Tensor::Empty(nodes->dtype(), nodes->shape(), device);
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMemcpyAsync((void *)(d_nodes->mutable_data()),
                                  (const void*)(nodes->data()),
                                  (size_t) nodes->size(),
                                  (cudaMemcpyKind)cudaMemcpyHostToDevice,
                                  (cudaStream_t) *id_copy_h2d_stream));
        CUDA_CALL(cudaStreamSynchronize((cudaStream_t)*id_copy_h2d_stream));

        task->train_nodes = d_nodes;
        next_q.AddTask(task);
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunDeviceSampleLoopOnce() {
    auto this_op = DEV_SAMPLE;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto fanouts = SamGraphEngine::GetFanout();
        auto num_layers = fanouts.size();
        auto last_layer_idx = num_layers - 1;

        auto dataset = SamGraphEngine::GetGraphDataset();
        auto sample_stream = SamGraphEngine::GetSampleStream();

        task->output_graph.resize(num_layers);
        for (int i = last_layer_idx; i >= 0; i--) {
            const nodeid_t *indptr = static_cast<const nodeid_t *>(dataset->indptr->data());;
            const nodeid_t *indices = static_cast<const nodeid_t *>(dataset->indices->data());
            const size_t num_node = dataset->num_node;
            const size_t num_edge = dataset->num_edge;
            const int fanout = fanouts[i];
            nodeid_t *out_src;
            nodeid_t *out_dst;
            size_t *num_out;
            nodeid_t *input;
            size_t num_input;
            
            if (i == last_layer_idx) {
                input = static_cast<nodeid_t *>(task->train_nodes->mutable_data());
                num_input = task->train_nodes->shape()[0];
            } else {

            }

            CUDA_CALL(cudaMalloc(&out_src, num_node * fanout * sizeof(nodeid_t)));
            CUDA_CALL(cudaMalloc(&out_dst, num_node * fanout * sizeof(nodeid_t)));
            CUDA_CALL(cudaMalloc(&num_out, sizeof(size_t)));

            cuda::DeviceSample((const nodeid_t *)indptr, (const nodeid_t *)indices,
                               (const size_t) num_node, (const size_t) num_edge,
                               (const nodeid_t *) input, (const size_t) num_input, (const int) fanout,
                               (nodeid_t *) out_src, (nodeid_t *) out_dst, (size_t *) num_out, 
                               (cudaStream_t) sample_stream);
            
        }
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunGraphCopyDevice2DeviceLoopOnce() {
    return true;
}

bool RunIdCopyDevice2HostLoopOnce() {
    return true;
}

bool RunHostFeatureSelectLoopOnce() {
    return true;
}

bool RunFeatureCopyHost2DeviceLoop() {
    return true;
}

bool RunSubmitLoopOnce() {
    return true;
}

void HostPermutateLoop() {
    while(RunHostPermutateLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void IdCopyHost2Device() {
    CUDA_CALL(cudaSetDevice(SamGraphEngine::GetSampleDevice()));
    while(RunIdCopyHost2DeviceLoopOnce() && !SamGraphEngine::ShouldShutdown()) {        
    }
    SamGraphEngine::ReportThreadFinish();
}

void DeviceSampleLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphEngine::GetSampleDevice()));
    while(RunDeviceSampleLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void GraphCopyDevice2DeviceLoop() {
    while(RunGraphCopyDevice2DeviceLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void IdCopyDevice2HostLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphEngine::GetSampleDevice()));
    while(RunIdCopyDevice2HostLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void HostFeatureSelectLoop() {
    while(RunHostFeatureSelectLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void FeatureCopyHost2DeviceLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphEngine::GetTrainDevice()));
    while(RunFeatureCopyHost2DeviceLoop() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void SubmitLoop() {
    while(RunSubmitLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

} // namespace common
} // namespace samgraph
