#include <thread>
#include <chrono>
#include <numeric>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "loops.h"
#include "logging.h"
#include "engine.h"
#include "cuda_sampling.h"
#include "cuda_hashtable.h"
#include "cuda_mapping.h"
#include "cuda_convert.h"
#include "cuda_util.h"

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
        task->key = encodeBatchKey(p->cur_epoch(), p->cur_batch());
        task->train_nodes = batch;

        next_q->AddTask(task);

        SAM_LOG(DEBUG) << "HostPermuateLoop: process task with key " << task->key;
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

        auto d_nodes = Tensor::Empty(nodes->dtype(), nodes->shape(), device, "task.train_nodes_" + std::to_string(task->key));
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMemcpyAsync((void *)(d_nodes->mutable_data()),
                                  (const void*)(nodes->data()),
                                  (size_t) nodes->size(),
                                  (cudaMemcpyKind)cudaMemcpyHostToDevice,
                                  (cudaStream_t) *id_copy_h2d_stream));
        CUDA_CALL(cudaStreamSynchronize((cudaStream_t)*id_copy_h2d_stream));

        task->train_nodes = d_nodes;
        task->cur_input = d_nodes;
        next_q->AddTask(task);

        SAM_LOG(DEBUG) << "IdCopyHost2DeviceLoop: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunDeviceSampleLoopOnce() {
    std::vector<QueueType> next_ops = {GRAPH_COPYD2D, ID_COPYD2H};
    for (auto next_op : next_ops) {
        auto q = SamGraphEngine::GetTaskQueue(next_op);
        if (q->ExceedThreshold()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
            return true;
        }
    }

    auto this_op = DEV_SAMPLE;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto fanouts = SamGraphEngine::GetFanout();
        auto num_layers = fanouts.size();
        auto last_layer_idx = num_layers - 1;

        auto dataset = SamGraphEngine::GetGraphDataset();
        auto sample_device = SamGraphEngine::GetSampleDevice();
        auto sample_stream = *SamGraphEngine::GetSampleStream();

        cusparseHandle_t cusparse_handle;
        CUSPARSE_CALL(cusparseCreate(&cusparse_handle));
        CUSPARSE_CALL(cusparseSetStream(cusparse_handle, sample_stream));

        auto train_nodes = task->train_nodes;
        size_t predict_node_num = train_nodes->shape()[0] * std::accumulate(fanouts.begin(), fanouts.end(), 1ul, std::multiplies<size_t>());
        cuda::OrderedHashTable hash_table(predict_node_num, sample_device, sample_stream, 3);
        
        size_t num_train_node = train_nodes->shape()[0];
        hash_table.FillWithUnique(static_cast<const IdType *const>(train_nodes->data()), num_train_node, sample_stream);
        task->output_graph.resize(num_layers);
        
        const IdType *indptr = static_cast<const IdType *>(dataset->indptr->data());
        const IdType *indices = static_cast<const IdType *>(dataset->indices->data());
        
        for (int i = last_layer_idx; i >= 0; i--) {
            CUDA_CALL(cudaStreamSynchronize(sample_stream));
            const int fanout = fanouts[i];
            const IdType *input = static_cast<const IdType *>(task->cur_input->data());
            const size_t num_input = task->cur_input->shape()[0];
            SAM_LOG(DEBUG) << "DeviceSampleLoop: begin sample layer " << i;

            IdType *out_src;
            IdType *out_dst;
            size_t *num_out;
            size_t host_num_out;

            CUDA_CALL(cudaMalloc(&out_src, num_input * fanout * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&out_dst, num_input * fanout * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&num_out, sizeof(size_t)));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: size of out_src " << num_input * fanout;
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda out_src malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda out_dst malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda num_out malloc " << toReadableSize(sizeof(size_t));

            // Sample a compact coo graph
            cuda::DeviceSample((const IdType *)indptr, (const IdType *)indices,
                               (const IdType *) input, (const size_t) num_input, (const size_t) fanout,
                               (IdType *) out_src, (IdType *) out_dst, (size_t *) num_out, 
                               (cudaStream_t) sample_stream);
            // Get nnz
            CUDA_CALL(cudaMemcpyAsync((void *)&host_num_out, (const void*)num_out, (size_t)sizeof(size_t), 
                                      (cudaMemcpyKind)cudaMemcpyDeviceToHost, (cudaStream_t)sample_stream));
            CUDA_CALL(cudaStreamSynchronize(sample_stream));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: number of samples " << host_num_out;

            // Populate the hash table with newly sampled nodes
            IdType *unique;
            size_t num_unique;

            // std::swap(out_src, out_dst); // swap the src and dst
            CUDA_CALL(cudaMalloc(&unique, (host_num_out + hash_table.NumItems()) * sizeof(IdType)));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda unique malloc " << toReadableSize((host_num_out + + hash_table.NumItems()) * sizeof(IdType));

            hash_table.FillWithDuplicates(out_dst, host_num_out, unique, &num_unique, sample_stream);
            // No need to Synchronize
            // CUDA_CALL(cudaStreamSynchronize(sample_stream));

            // Mapping edges
            IdType *new_src;
            IdType *new_dst;

            CUDA_CALL(cudaMalloc(&new_src, host_num_out * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&new_dst, host_num_out * sizeof(IdType)));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: size of new_src " << host_num_out;
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda new_src malloc " << toReadableSize(host_num_out * sizeof(IdType));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda new_dst malloc " << toReadableSize(host_num_out * sizeof(IdType));

            cuda::MapEdges((const IdType *) out_src,
                           (IdType * const) new_src,
                           (const IdType * const) out_dst,
                           (IdType * const) new_dst,
                           (const size_t) host_num_out,
                           (cuda::DeviceOrderedHashTable) hash_table.DeviceHandle(),
                           (cudaStream_t) sample_stream);

            // Convert COO format to CSR format
            IdType *new_indptr;
            CUDA_CALL(cudaMalloc(&new_indptr, (num_input + 1) * sizeof(IdType)));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: cuda new_indptr malloc " << toReadableSize((num_unique + 1) * sizeof(IdType));
            cuda::ConvertCoo2Csr(new_src, new_dst, num_input, num_unique, host_num_out, new_indptr,
                                 sample_device, cusparse_handle, sample_stream);

            auto train_graph = std::make_shared<TrainGraph>();
            train_graph->indptr = Tensor::FromBlob(new_indptr, DataType::kSamI32, {num_input + 1}, sample_device, "train_graph.inptr_cuda_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            train_graph->indices = Tensor::FromBlob(new_dst, DataType::kSamI32, {host_num_out}, sample_device, "train_graph.indices_cuda_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            train_graph->num_row = num_input;
            train_graph->num_column = num_unique;
            train_graph->num_edge = host_num_out;

            task->output_graph[i] = train_graph;

            // Do some clean jobs
            CUDA_CALL(cudaStreamSynchronize(sample_stream));
            CUDA_CALL(cudaFree(out_src));
            CUDA_CALL(cudaFree(out_dst));
            CUDA_CALL(cudaFree(num_out));
            CUDA_CALL(cudaFree(new_src));

            task->cur_input = Tensor::FromBlob((void *)unique, DataType::kSamI32, {num_unique}, sample_device, "cur_input_unique_cuda_" + std::to_string(task->key) + "_" + std::to_string(i));
            SAM_LOG(DEBUG) << "DeviceSampleLoop: finish layer " << i;
        }

        CUDA_CALL(cudaStreamSynchronize(sample_stream));
        CUSPARSE_CALL(cusparseDestroy(cusparse_handle));

        task->output_nodes = task->cur_input;

        // Deliver the taks to next worker thread
        std::vector<QueueType> next_ops = {GRAPH_COPYD2D, ID_COPYD2H};
        for (auto next_op : next_ops) {
            auto next_q = SamGraphEngine::GetTaskQueue(next_op);
            next_q->AddTask(task);
        }

        SAM_LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunGraphCopyDevice2DeviceLoopOnce() {
    auto this_op = GRAPH_COPYD2D;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto train_device = SamGraphEngine::GetTrainDevice();
        auto graph_copy_stream = *SamGraphEngine::GetGraphCopyDevice2DeviceStream();

        for (size_t i = 0; i < task->output_graph.size(); i++) {
            auto graph = task->output_graph[i];
            void *train_indptr;
            void *train_indices;
            CUDA_CALL(cudaSetDevice(train_device));
            CUDA_CALL(cudaMalloc(&train_indptr, graph->indptr->size()));
            CUDA_CALL(cudaMalloc(&train_indices, graph->indices->size()));
            SAM_LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_indptr malloc " << toReadableSize(graph->indptr->size());
            SAM_LOG(DEBUG) << "GraphCopyDevice2DeviceLoop: cuda train_indices malloc " << toReadableSize(graph->indices->size());

            CUDA_CALL(cudaMemcpyAsync(train_indptr, graph->indptr->data(), graph->indptr->size(),
                                      cudaMemcpyDeviceToDevice, graph_copy_stream));
            CUDA_CALL(cudaMemcpyAsync(train_indices, graph->indices->data(), graph->indices->size(),
                                      cudaMemcpyDeviceToDevice, graph_copy_stream));

            CUDA_CALL(cudaStreamSynchronize(graph_copy_stream));
            graph->indptr = Tensor::FromBlob(train_indptr, graph->indptr->dtype(), graph->indptr->shape(), train_device, "train_graph.inptr_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
            graph->indices = Tensor::FromBlob(train_indices, graph->indptr->dtype(), graph->indices->shape(), train_device, "train_graph.inptr_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
        }

        CUDA_CALL(cudaStreamSynchronize(graph_copy_stream));

        auto ready_table = SamGraphEngine::GetSubmitTable();
        ready_table->AddReadyCount(task->key);

        SAM_LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunIdCopyDevice2HostLoopOnce() {
    auto next_op = FEAT_EXTRACT;
    auto next_q = SamGraphEngine::GetTaskQueue(next_op);

    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = ID_COPYD2H;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto id_copy_d2h_stream = *SamGraphEngine::GetIdCopyDevice2HostStream();
        void *node_ids = malloc(task->output_nodes->size());

        CUDA_CALL(cudaMemcpyAsync(node_ids, task->output_nodes->data(), task->output_nodes->size(),
                                  cudaMemcpyDeviceToHost, id_copy_d2h_stream));
        CUDA_CALL(cudaStreamSynchronize(id_copy_d2h_stream));
        
        task->output_nodes = Tensor::FromBlob(node_ids, task->output_nodes->dtype(),
                                              task->output_nodes->shape(), CPU_DEVICE_ID, "task.output_nodes_cpu_" + std::to_string(task->key));

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunHostFeatureExtractLoopOnce() {
    auto next_op = FEAT_COPYH2D;
    auto next_q = SamGraphEngine::GetTaskQueue(next_op);
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = FEAT_EXTRACT;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto dataset = SamGraphEngine::GetGraphDataset();
        auto output_nodes = task->output_nodes;
        SAM_CHECK_EQ(output_nodes->device(), CPU_DEVICE_ID);
        auto feat_dim = dataset->feat->shape()[1];
        auto feat_type = dataset->feat->dtype();
        auto label_type = dataset->label->dtype();
        auto idx = reinterpret_cast<const IdType *>(output_nodes->data()); 
        auto num_idx = output_nodes->shape()[0];

        task->output_feat = Tensor::Empty(feat_type, {num_idx, feat_dim}, CPU_DEVICE_ID, "task.output_feat_cpu_" + std::to_string(task->key));
        task->output_label = Tensor::Empty(label_type, {num_idx}, CPU_DEVICE_ID, "task.output_label_cpu" + std::to_string(task->key));

        auto extractor = SamGraphEngine::GetCpuExtractor();

        auto feat_dst = task->output_feat->mutable_data();
        auto feat_src = dataset->feat->data();
        extractor->extract(feat_dst, feat_src, idx, num_idx, feat_dim, feat_type);

        auto label_dst = task->output_label->mutable_data();
        auto label_src = dataset->label->data();
        extractor->extract(label_dst, label_src, idx, num_idx, 1, label_type);

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunFeatureCopyHost2DeviceLoopOnce() {
    auto next_op = SUBMIT;
    auto next_q = SamGraphEngine::GetTaskQueue(next_op);
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = FEAT_COPYH2D;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto train_device = SamGraphEngine::GetTrainDevice();
        CUDA_CALL(cudaSetDevice(train_device));
        auto feat_copy_h2d_stream = *SamGraphEngine::GetFeatureCopyHost2DeviceStream();

        auto feat = task->output_feat;
        auto label = task->output_label;

        auto d_feat = Tensor::Empty(feat->dtype(), feat->shape(), train_device, "task.train_feat_cuda_" + std::to_string(task->key));
        auto d_label = Tensor::Empty(label->dtype(), label->shape(), train_device, "task.train_label_cuda" + std::to_string(task->key));

        CUDA_CALL(cudaMemcpyAsync(d_feat->mutable_data(), feat->data(), feat->size(),
                                  cudaMemcpyHostToDevice, feat_copy_h2d_stream));
        CUDA_CALL(cudaMemcpyAsync(d_label->mutable_data(), label->data(), label->size(),
                                  cudaMemcpyHostToDevice, feat_copy_h2d_stream));
        CUDA_CALL(cudaStreamSynchronize(feat_copy_h2d_stream));

        auto ready_table = SamGraphEngine::GetSubmitTable();
        ready_table->AddReadyCount(task->key);

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunSubmitLoopOnce() {
    auto graph_pool = SamGraphEngine::GetGraphPool();
    if (graph_pool->ExceedThreshold()) {
        return true;
    }

    auto this_op = SUBMIT;
    auto q = SamGraphEngine::GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        graph_pool->AddGraphBatch(task->key, task);
        SAM_LOG(DEBUG) << "Submit: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

void HostPermutateLoop() {
    while(RunHostPermutateLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void IdCopyHost2DeviceLoop() {
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

void HostFeatureExtractLoop() {
    while(RunHostFeatureExtractLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void FeatureCopyHost2DeviceLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphEngine::GetTrainDevice()));
    while(RunFeatureCopyHost2DeviceLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void SubmitLoop() {
    while(RunSubmitLoopOnce() && !SamGraphEngine::ShouldShutdown()) {
    }
    SamGraphEngine::ReportThreadFinish();
}

void SingleLoop() {
    RunHostPermutateLoopOnce();
    RunIdCopyHost2DeviceLoopOnce();
    RunDeviceSampleLoopOnce();
    RunGraphCopyDevice2DeviceLoopOnce();
    RunIdCopyDevice2HostLoopOnce();
    RunHostFeatureExtractLoopOnce();
    RunFeatureCopyHost2DeviceLoopOnce();
    RunSubmitLoopOnce();
    SamGraphEngine::ReportThreadFinish();
}

} // namespace common
} // namespace samgraph
