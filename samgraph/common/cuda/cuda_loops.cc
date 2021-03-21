#include <thread>
#include <chrono>
#include <numeric>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "../logging.h"
#include "cuda_loops.h"
#include "cuda_engine.h"
#include "cuda_function.h"
#include "cuda_hashtable.h"

namespace samgraph {
namespace common {
namespace cuda {

bool RunHostPermutateLoopOnce() {
    auto next_op = CUDA_ID_COPYH2D;
    auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);

    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto p = SamGraphCudaEngine::GetEngine()->GetRandomPermutation();
    auto batch = p->GetBatch();

    if (batch) {
        // Create task entry
        auto task = std::make_shared<TaskEntry>();
        task->key = encodeBatchKey(p->cur_epoch(), p->cur_step());
        task->train_nodes = batch;
        task->output_nodes = batch;

        next_q->AddTask(task);

        SAM_LOG(DEBUG) << "HostPermuateLoop: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunIdCopyHost2DeviceLoopOnce() {
    auto next_op = CUDA_SAMPLE;
    auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);
    
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = CUDA_ID_COPYH2D;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto nodes = task->train_nodes;
        auto id_copy_h2d_stream = SamGraphCudaEngine::GetEngine()->GetIdCopyHost2DeviceStream();
        auto device = SamGraphCudaEngine::GetEngine()->GetSampleDevice();

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

bool RunCudaSampleOnce() {
    std::vector<CudaQueueType> next_ops = {CUDA_GRAPH_COPYD2D, CUDA_ID_COPYD2H};
    for (auto next_op : next_ops) {
        auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);
        if (q->ExceedThreshold()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
            return true;
        }
    }

    auto this_op = CUDA_SAMPLE;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto fanouts = SamGraphCudaEngine::GetEngine()->GetFanout();
        auto num_layers = fanouts.size();
        auto last_layer_idx = num_layers - 1;

        auto dataset = SamGraphCudaEngine::GetEngine()->GetGraphDataset();
        auto sample_device = SamGraphCudaEngine::GetEngine()->GetSampleDevice();
        auto sample_stream = *SamGraphCudaEngine::GetEngine()->GetSampleStream();

        cusparseHandle_t cusparse_handle;
        CUSPARSE_CALL(cusparseCreate(&cusparse_handle));
        CUSPARSE_CALL(cusparseSetStream(cusparse_handle, sample_stream));

        auto train_nodes = task->train_nodes;
        size_t predict_node_num = train_nodes->shape()[0] + 
                                  train_nodes->shape()[0] * std::accumulate(fanouts.begin(), fanouts.end(), 1ul, std::multiplies<size_t>());
        OrderedHashTable hash_table(predict_node_num, sample_device, sample_stream, 3);
        
        size_t num_train_node = train_nodes->shape()[0];
        hash_table.FillWithUnique(static_cast<const IdType *const>(train_nodes->data()), num_train_node, sample_stream);
        task->output_graph.resize(num_layers);
        
        const IdType *indptr = static_cast<const IdType *>(dataset->indptr->data());
        const IdType *indices = static_cast<const IdType *>(dataset->indices->data());
        
        for (int i = last_layer_idx; i >= 0; i--) {
            const int fanout = fanouts[i];
            const IdType *input = static_cast<const IdType *>(task->cur_input->data());
            const size_t num_input = task->cur_input->shape()[0];
            SAM_LOG(DEBUG) << "CudaSample: begin sample layer " << i;

            IdType *out_src;
            IdType *out_dst;
            size_t *num_out;
            size_t host_num_out;

            CUDA_CALL(cudaMalloc(&out_src, num_input * fanout * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&out_dst, num_input * fanout * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&num_out, sizeof(size_t)));
            SAM_LOG(DEBUG) << "CudaSample: size of out_src " << num_input * fanout;
            SAM_LOG(DEBUG) << "CudaSample: cuda out_src malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
            SAM_LOG(DEBUG) << "CudaSample: cuda out_dst malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
            SAM_LOG(DEBUG) << "CudaSample: cuda num_out malloc " << toReadableSize(sizeof(size_t));

            // Sample a compact coo graph
            CudaSample((const IdType *)indptr, (const IdType *)indices,
                               (const IdType *) input, (const size_t) num_input, (const size_t) fanout,
                               (IdType *) out_src, (IdType *) out_dst, (size_t *) num_out, 
                               (cudaStream_t) sample_stream);
            // Get nnz
            CUDA_CALL(cudaMemcpyAsync((void *)&host_num_out, (const void*)num_out, (size_t)sizeof(size_t), 
                                      (cudaMemcpyKind)cudaMemcpyDeviceToHost, (cudaStream_t)sample_stream));
            CUDA_CALL(cudaStreamSynchronize(sample_stream));
            SAM_LOG(DEBUG) << "CudaSample: number of samples " << host_num_out;

            // Populate the hash table with newly sampled nodes
            IdType *unique;
            size_t num_unique;

            CUDA_CALL(cudaMalloc(&unique, (host_num_out + hash_table.NumItems()) * sizeof(IdType)));
            SAM_LOG(DEBUG) << "CudaSample: cuda unique malloc " << toReadableSize((host_num_out + + hash_table.NumItems()) * sizeof(IdType));

            hash_table.FillWithDuplicates(out_dst, host_num_out, unique, &num_unique, sample_stream);
            // No need to Synchronize
            // CUDA_CALL(cudaStreamSynchronize(sample_stream));

            // Mapping edges
            IdType *new_src;
            IdType *new_dst;

            CUDA_CALL(cudaMalloc(&new_src, host_num_out * sizeof(IdType)));
            CUDA_CALL(cudaMalloc(&new_dst, host_num_out * sizeof(IdType)));
            SAM_LOG(DEBUG) << "CudaSample: size of new_src " << host_num_out;
            SAM_LOG(DEBUG) << "CudaSample: cuda new_src malloc " << toReadableSize(host_num_out * sizeof(IdType));
            SAM_LOG(DEBUG) << "CudaSample: cuda new_dst malloc " << toReadableSize(host_num_out * sizeof(IdType));

            MapEdges((const IdType *) out_src,
                           (IdType * const) new_src,
                           (const IdType * const) out_dst,
                           (IdType * const) new_dst,
                           (const size_t) host_num_out,
                           (DeviceOrderedHashTable) hash_table.DeviceHandle(),
                           (cudaStream_t) sample_stream);

            // Convert COO format to CSR format
            IdType *new_indptr;
            CUDA_CALL(cudaMalloc(&new_indptr, (num_input + 1) * sizeof(IdType)));
            SAM_LOG(DEBUG) << "CudaSample: cuda new_indptr malloc " << toReadableSize((num_unique + 1) * sizeof(IdType));
            ConvertCoo2Csr(new_src, new_dst, num_input, num_unique, host_num_out, new_indptr,
                                 sample_device, cusparse_handle, sample_stream);

            auto train_graph = std::make_shared<TrainGraph>();
            train_graph->indptr = Tensor::FromBlob(new_indptr, DataType::kSamI32, {num_input + 1}, sample_device, "train_graph.inptr_cuda_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            train_graph->indices = Tensor::FromBlob(new_dst, DataType::kSamI32, {host_num_out}, sample_device, "train_graph.indices_cuda_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            train_graph->num_row = num_input;
            train_graph->num_column = num_unique;
            train_graph->num_edge = host_num_out;

            task->output_graph[i] = train_graph;

            // Do some clean jobs
            CUDA_CALL(cudaFree(out_src));
            CUDA_CALL(cudaFree(out_dst));
            CUDA_CALL(cudaFree(num_out));
            CUDA_CALL(cudaFree(new_src));

            task->cur_input = Tensor::FromBlob((void *)unique, DataType::kSamI32, {num_unique}, sample_device, "cur_input_unique_cuda_" + std::to_string(task->key) + "_" + std::to_string(i));
            SAM_LOG(DEBUG) << "CudaSample: finish layer " << i;
        }

        CUSPARSE_CALL(cusparseDestroy(cusparse_handle));

        task->input_nodes = task->cur_input;

        // Deliver the taks to next worker thread
        std::vector<CudaQueueType> next_ops = {CUDA_GRAPH_COPYD2D, CUDA_ID_COPYD2H};
        for (auto next_op : next_ops) {
            auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);
            next_q->AddTask(task);
        }

        SAM_LOG(DEBUG) << "SampleLoop: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunGraphCopyDevice2DeviceLoopOnce() {
    auto this_op = CUDA_GRAPH_COPYD2D;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto train_device = SamGraphCudaEngine::GetEngine()->GetTrainDevice();
        auto graph_copy_stream = *SamGraphCudaEngine::GetEngine()->GetGraphCopyDevice2DeviceStream();

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
            CUDA_CALL(cudaStreamSynchronize(graph_copy_stream));
    
            CUDA_CALL(cudaMemcpyAsync(train_indices, graph->indices->data(), graph->indices->size(),
                                      cudaMemcpyDeviceToDevice, graph_copy_stream));
            CUDA_CALL(cudaStreamSynchronize(graph_copy_stream));

            graph->indptr = Tensor::FromBlob(train_indptr, graph->indptr->dtype(), graph->indptr->shape(), train_device, "train_graph.inptr_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
            graph->indices = Tensor::FromBlob(train_indices, graph->indptr->dtype(), graph->indices->shape(), train_device, "train_graph.inptr_cuda_train_" + std::to_string(task->key) + "_" + std::to_string(i));
        }

        CUDA_CALL(cudaStreamSynchronize(graph_copy_stream));

        auto ready_table = SamGraphCudaEngine::GetEngine()->GetSubmitTable();
        ready_table->AddReadyCount(task->key);

        SAM_LOG(DEBUG) << "GraphCopyDevice2Device: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunIdCopyDevice2HostLoopOnce() {
    auto next_op = CUDA_FEAT_EXTRACT;
    auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);

    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = CUDA_ID_COPYD2H;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto id_copy_d2h_stream = *SamGraphCudaEngine::GetEngine()->GetIdCopyDevice2HostStream();
        void *node_ids = malloc(task->input_nodes->size());
        SAM_LOG(DEBUG) << "IdCopyDevice2Host node_ids cpu malloc " << toReadableSize(task->input_nodes->size());

        CUDA_CALL(cudaMemcpyAsync(node_ids, task->input_nodes->data(), task->input_nodes->size(),
                                  cudaMemcpyDeviceToHost, id_copy_d2h_stream));
        CUDA_CALL(cudaStreamSynchronize(id_copy_d2h_stream));

        task->input_nodes = Tensor::FromBlob(node_ids, task->input_nodes->dtype(),
                                              task->input_nodes->shape(), CPU_DEVICE_ID, "task.input_nodes_cpu_" + std::to_string(task->key));

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "IdCopyDevice2Host: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunHostFeatureExtractLoopOnce() {
    auto next_op = CUDA_FEAT_COPYH2D;
    auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = CUDA_FEAT_EXTRACT;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto dataset = SamGraphCudaEngine::GetEngine()->GetGraphDataset();
        auto input_nodes = task->input_nodes;
        auto output_nodes = task->output_nodes;
        SAM_CHECK_EQ(input_nodes->device(), CPU_DEVICE_ID);
        SAM_CHECK_EQ(output_nodes->device(), CPU_DEVICE_ID);

        auto feat = dataset->feat;
        auto label = dataset->label;
    
        auto feat_dim = dataset->feat->shape()[1];
        auto feat_type = dataset->feat->dtype();
        auto label_type = dataset->label->dtype();

        auto input_data = reinterpret_cast<const IdType *>(input_nodes->data());
        auto output_data = reinterpret_cast<const IdType *>(output_nodes->data()); 
        auto num_input = input_nodes->shape()[0];
        auto num_ouput = output_nodes->shape()[0];

        task->input_feat = Tensor::Empty(feat_type,{num_input, feat_dim}, CPU_DEVICE_ID, "task.input_feat_cpu_" + std::to_string(task->key));
        task->output_label = Tensor::Empty(label_type, {num_ouput}, CPU_DEVICE_ID, "task.output_label_cpu" + std::to_string(task->key));

        auto extractor = SamGraphCudaEngine::GetEngine()->GetExtractor();

        auto feat_dst = task->input_feat->mutable_data();
        auto feat_src = dataset->feat->data();
        extractor->extract(feat_dst, feat_src, input_data, num_input, feat_dim, feat_type);

        auto label_dst = task->output_label->mutable_data();
        auto label_src = dataset->label->data();
        extractor->extract(label_dst, label_src, output_data, num_ouput, 1, label_type);

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "HostFeatureExtract: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunFeatureCopyHost2DeviceLoopOnce() {
    auto next_op = CUDA_SUBMIT;
    auto next_q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(next_op);
    if (next_q->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = CUDA_FEAT_COPYH2D;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        auto train_device = SamGraphCudaEngine::GetEngine()->GetTrainDevice();
        CUDA_CALL(cudaSetDevice(train_device));
        auto feat_copy_h2d_stream = *SamGraphCudaEngine::GetEngine()->GetFeatureCopyHost2DeviceStream();

        auto feat = task->input_feat;
        auto label = task->output_label;

        auto d_feat = Tensor::Empty(feat->dtype(), feat->shape(), train_device, "task.train_feat_cuda_" + std::to_string(task->key));
        auto d_label = Tensor::Empty(label->dtype(), label->shape(), train_device, "task.train_label_cuda" + std::to_string(task->key));

        CUDA_CALL(cudaMemcpyAsync(d_feat->mutable_data(), feat->data(), feat->size(),
                                  cudaMemcpyHostToDevice, feat_copy_h2d_stream));
        CUDA_CALL(cudaStreamSynchronize(feat_copy_h2d_stream));

        CUDA_CALL(cudaMemcpyAsync(d_label->mutable_data(), label->data(), label->size(),
                                  cudaMemcpyHostToDevice, feat_copy_h2d_stream));
        CUDA_CALL(cudaStreamSynchronize(feat_copy_h2d_stream));

        task->input_feat = d_feat;
        task->output_label = d_label;

        auto ready_table = SamGraphCudaEngine::GetEngine()->GetSubmitTable();
        ready_table->AddReadyCount(task->key);

        next_q->AddTask(task);
        SAM_LOG(DEBUG) << "FeatureCopyHost2Device: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

bool RunSubmitLoopOnce() {
    auto graph_pool = SamGraphCudaEngine::GetEngine()->GetGraphPool();
    if (graph_pool->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto this_op = CUDA_SUBMIT;
    auto q = SamGraphCudaEngine::GetEngine()->GetTaskQueue(this_op);
    auto task = q->GetTask();

    if (task) {
        SAM_LOG(DEBUG) << "Submit: process task with key " << task->key;
        graph_pool->AddGraphBatch(task->key, task);
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

void HostPermutateLoop() {
    while(RunHostPermutateLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void IdCopyHost2DeviceLoop() {
    while(RunIdCopyHost2DeviceLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {        
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void CudaSample() {
    CUDA_CALL(cudaSetDevice(SamGraphCudaEngine::GetEngine()->GetSampleDevice()));
    while(RunCudaSampleOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void GraphCopyDevice2DeviceLoop() {
    while(RunGraphCopyDevice2DeviceLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void IdCopyDevice2HostLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphCudaEngine::GetEngine()->GetSampleDevice()));
    while(RunIdCopyDevice2HostLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void HostFeatureExtractLoop() {
    while(RunHostFeatureExtractLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void FeatureCopyHost2DeviceLoop() {
    CUDA_CALL(cudaSetDevice(SamGraphCudaEngine::GetEngine()->GetTrainDevice()));
    while(RunFeatureCopyHost2DeviceLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

void SubmitLoop() {
    while(RunSubmitLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

bool RunSingleLoopOnce() {
    auto tic = std::chrono::system_clock::now();;

    RunHostPermutateLoopOnce();
    RunIdCopyHost2DeviceLoopOnce();
    RunCudaSampleOnce();
    RunGraphCopyDevice2DeviceLoopOnce();
    RunIdCopyDevice2HostLoopOnce();
    RunHostFeatureExtractLoopOnce();
    RunFeatureCopyHost2DeviceLoopOnce();
    RunSubmitLoopOnce();

    auto toc = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = toc - tic;

    SAM_LOG(DEBUG) << "Sampling one batch uses " << duration.count() << " secs";

    return true;
}

void SingleLoop() {
    while(RunSingleLoopOnce() && !SamGraphCudaEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCudaEngine::GetEngine()->ReportThreadFinish();
}

} // namespace cuda
} // namespace common
} // namespace samgraph
