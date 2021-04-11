#include <numeric>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

#include "../logging.h"
#include "../timer.h"
#include "../profiler.h"
#include "cpu_loops.h"
#include "cpu_engine.h"
#include "cpu_hashtable.h"
#include "cpu_function.h"

namespace samgraph {
namespace common {
namespace cpu {

bool RunCpuSampleLoopOnce() {
    auto graph_pool = SamGraphCpuEngine::GetEngine()->GetGraphPool();
    if (graph_pool->ExceedThreshold()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
        return true;
    }

    auto p = SamGraphCpuEngine::GetEngine()->GetRandomPermutation();
    auto batch = p->GetBatch();

    if (batch) {
        // Create task entry
        Timer t;
        auto task = std::make_shared<TaskEntry>();
        task->key = encodeBatchKey(p->cur_epoch(), p->cur_step());
        task->train_nodes = batch;
        task->cur_input = batch;
        task->output_nodes = batch;
        task->epoch = p->cur_epoch();
        task->step = p->cur_step();

        uint64_t profile_idx = Profiler::GetEntryIndex(task->epoch, task->step);
        auto fanouts = SamGraphCpuEngine::GetEngine()->GetFanout();
        auto num_layers = fanouts.size();
        auto last_layer_idx = num_layers - 1;

        auto dataset = SamGraphCpuEngine::GetEngine()->GetGraphDataset();
        // auto train_device = SamGraphCpuEngine::GetEngine()->GetTrainDevice();
        // auto work_stream = *SamGraphCpuEngine::GetEngine()->GetWorkStream();

        auto train_nodes = task->train_nodes;

        auto hash_table = SamGraphCpuEngine::GetEngine()->GetHashTable();
        hash_table->Clear();
        
        size_t num_train_node = train_nodes->shape()[0];
        hash_table->Populate(static_cast<const IdType *>(train_nodes->data()), num_train_node);

        task->output_graph.resize(num_layers);
    
        const IdType *indptr = static_cast<const IdType *>(dataset->indptr->data());
        const IdType *indices = static_cast<const IdType *>(dataset->indices->data());

        for (int i = last_layer_idx; i >= 0; i--) {
            Timer t0;
            const int fanout = fanouts[i];
            const IdType *input = static_cast<const IdType *>(task->cur_input->data());
            const size_t num_input = task->cur_input->shape()[0];
            SAM_LOG(DEBUG) << "CpuSample: begin sample layer " << i;

            IdType *out_src = (IdType *)malloc(num_input * fanout * sizeof(IdType));
            IdType *out_dst = (IdType *)malloc(num_input * fanout * sizeof(IdType));
            size_t num_out;
            SAM_LOG(DEBUG) << "CpuSample: cpu out_src malloc " << toReadableSize(num_input * fanout * sizeof(IdType));
            SAM_LOG(DEBUG) << "CpuSample: cpu out_src malloc " << toReadableSize(num_input * fanout * sizeof(IdType));

            // Sample a compact coo graph
            CpuSample(indptr, indices, input, num_input, out_src, out_dst, &num_out, fanout);
            SAM_LOG(DEBUG) << "CpuSample: num_out " << num_out;
            Profiler::Get()->num_samples[profile_idx] += num_out;
            double ns_time = t0.Passed();

            Timer t1;
            Timer t2;

            // Populate the hash table with newly sampled nodes
            hash_table->Populate(out_dst, num_out);

            double populate_time = t2.Passed();

            Timer t3;
            size_t num_unique = hash_table->NumItem();
            IdType *unique = (IdType *) malloc(num_unique * sizeof(IdType));
            hash_table->MapNodes(unique, num_unique);

            double map_nodes_time = t3.Passed();

            Timer t4;
            // Mapping edges
            IdType *new_src = (IdType *) malloc(num_out * sizeof(IdType));
            IdType *new_dst = (IdType *) malloc(num_out * sizeof(IdType));
            SAM_LOG(DEBUG) << "CpuSample: cpu new_src malloc " << toReadableSize(num_out * sizeof(IdType));
            SAM_LOG(DEBUG) << "CpuSample: cpu new_src malloc " << toReadableSize(num_out * sizeof(IdType));
            hash_table->MapEdges(out_src, out_dst, num_out, new_src, new_dst);

            double map_edges_time = t4.Passed();

            double remap_time = t1.Passed();

            // IdType *new_indptr = (IdType *) malloc((num_input + 1) * sizeof(IdType));
            // IdType *new_indices = (IdType *) malloc(num_out * sizeof(IdType));
            // SAM_LOG(DEBUG) << "CpuSample: cpu new_indptr malloc " << toReadableSize((num_input + 1) * sizeof(IdType));
            // SAM_LOG(DEBUG) << "CpuSample: cpu new_indices malloc " << toReadableSize(num_out * sizeof(IdType));
            // ConvertCoo2Csr(new_src, new_dst, new_indptr, new_indices, num_input, num_out);

            // IdType *d_indptr;
            // IdType *d_indices;
            // CUDA_CALL(cudaMalloc(&d_indptr, (num_input + 1) * sizeof(IdType)));
            // CUDA_CALL(cudaMalloc(&d_indices, num_out * sizeof(IdType)));
            // CUDA_CALL(cudaMemcpyAsync(d_indptr, new_indptr, (num_input + 1) * sizeof(IdType), cudaMemcpyHostToDevice, work_stream));
            // CUDA_CALL(cudaMemcpyAsync(d_indices, new_indices, num_out * sizeof(IdType), cudaMemcpyHostToDevice, work_stream));
            // CUDA_CALL(cudaStreamSynchronize(work_stream));
            // SAM_LOG(DEBUG) << "CpuSample: cuda d_indptr malloc " << toReadableSize((num_input + 1) * sizeof(IdType));
            // SAM_LOG(DEBUG) << "CpuSample: cuda d_indices malloc " << toReadableSize(num_out * sizeof(IdType));

            // auto train_graph = std::make_shared<TrainGraph>();
            // train_graph->indptr = Tensor::FromBlob(d_indptr, DataType::kSamI32, {num_input + 1}, train_device, "train_graph.inptr_cpu_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            // train_graph->indices = Tensor::FromBlob(d_indices, DataType::kSamI32, {num_out}, train_device, "train_graph.indices_cpu_sample_" + std::to_string(task->key) + "_" + std::to_string(i));
            // train_graph->num_row = num_input;
            // train_graph->num_column = num_unique;
            // train_graph->num_edge = num_out;

            // task->output_graph[i] = train_graph;
            task->cur_input = Tensor::FromBlob((void *)unique, DataType::kSamI32, {num_unique}, CPU_DEVICE_ID, "cur_input_unique_cpu_" + std::to_string(task->key) + "_" + std::to_string(i));

            free(out_src);
            free(out_dst);
            free(new_src);
            free(new_dst);
            // free(new_indptr);
            // free(new_indices);

            Profiler::Get()->ns_time[profile_idx] += ns_time;
            Profiler::Get()->remap_time[profile_idx] += remap_time;
            Profiler::Get()->populate_time[profile_idx] += populate_time;
            Profiler::Get()->map_node_time[profile_idx] += map_nodes_time;
            Profiler::Get()->map_edge_time[profile_idx] += map_edges_time;


            SAM_LOG(DEBUG) << "layer " << i << " ns " << ns_time << " remap " << remap_time;
    
            SAM_LOG(DEBUG) << "CpuSample: finish layer " << i;
        }

        // task->input_nodes = task->cur_input;

        // // Extract feature
        // auto input_nodes = task->input_nodes;
        // auto output_nodes = task->output_nodes;
        // SAM_CHECK_EQ(input_nodes->device(), CPU_DEVICE_ID);
        // SAM_CHECK_EQ(output_nodes->device(), CPU_DEVICE_ID);
    
        // auto feat_dim = dataset->feat->shape()[1];
        // auto feat_type = dataset->feat->dtype();
        // auto label_type = dataset->label->dtype();

        // auto input_data = reinterpret_cast<const IdType *>(input_nodes->data());
        // auto output_data = reinterpret_cast<const IdType *>(output_nodes->data()); 
        // auto num_input = input_nodes->shape()[0];
        // auto num_ouput = output_nodes->shape()[0];

        // auto feat = Tensor::Empty(feat_type,{num_input, feat_dim}, CPU_DEVICE_ID, "task.input_feat_cpu_" + std::to_string(task->key));
        // auto label = Tensor::Empty(label_type, {num_ouput}, CPU_DEVICE_ID, "task.output_label_cpu" + std::to_string(task->key));

        // auto extractor = SamGraphCpuEngine::GetEngine()->GetExtractor();

        // auto feat_dst = feat->mutable_data();
        // auto feat_src = dataset->feat->data();
        // extractor->extract(feat_dst, feat_src, input_data, num_input, feat_dim, feat_type);

        // auto label_dst = label->mutable_data();
        // auto label_src = dataset->label->data();
        // extractor->extract(label_dst, label_src, output_data, num_ouput, 1, label_type);
        
        // // Copy data
        // auto d_feat = Tensor::Empty(feat->dtype(), feat->shape(), train_device, "task.train_feat_cuda_" + std::to_string(task->key));
        // auto d_label = Tensor::Empty(label->dtype(), label->shape(), train_device, "task.train_label_cuda" + std::to_string(task->key));

        // CUDA_CALL(cudaMemcpyAsync(d_feat->mutable_data(), feat->data(), feat->size(),
        //                           cudaMemcpyHostToDevice, work_stream));
        // CUDA_CALL(cudaStreamSynchronize(work_stream));

        // CUDA_CALL(cudaMemcpyAsync(d_label->mutable_data(), label->data(), label->size(),
        //                           cudaMemcpyHostToDevice, work_stream));
        // CUDA_CALL(cudaStreamSynchronize(work_stream));

        // task->input_feat = d_feat;
        // task->output_label = d_label;

        // // Submit
        // auto graph_pool = SamGraphCpuEngine::GetEngine()->GetGraphPool();
        // graph_pool->AddGraphBatch(task->key, task);

        double sam_time = t.Passed();
        Profiler::Get()->sample_time[profile_idx] += sam_time;

        Profiler::Get()->Report(p->cur_epoch(), p->cur_step());


        SAM_LOG(DEBUG) << "CpuSampleLoop: process task with key " << task->key;
    } else {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    return true;
}

void CpuSampleLoop() {
    auto train_device = SamGraphCpuEngine::GetEngine()->GetTrainDevice();
    CUDA_CALL(cudaSetDevice(train_device));
    while(RunCpuSampleLoopOnce() && !SamGraphCpuEngine::GetEngine()->ShouldShutdown()) {
    }
    SamGraphCpuEngine::GetEngine()->ReportThreadFinish();
}

} // namespace cpu
} // namespace common
} // namespace samgraph
