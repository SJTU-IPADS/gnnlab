#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <cstdio>

#include <gflags/gflags.h>

#include "data/dataset.hpp"
#include "sampling/cpu/sampler.hpp"
#include "sampling/cpu/block.hpp"
#include "sampling/cpu/shuffler.hpp"

DEFINE_string(dataset_key, "papers100M","");
DEFINE_string(dataset_folder, "/graph-learning/preprocess/papers100M", "");
DEFINE_string(fanout, "15,10,5", "");
DEFINE_int32(num_epoch, 200, "");
DEFINE_uint64(batch_size, 8192, "");

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("some usage message");
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(FLAGS_dataset_key, FLAGS_dataset_folder);

    SamplingTask task;
    std::stringstream ss(FLAGS_fanout);
    for (int i; ss >> i;) {
        task.fanout.push_back(i);
        if (ss.peek() == ',') {
            ss.ignore();
        }
    }
    task.num_blocks = task.fanout.size();

    Shuffler shuffler(dataset->GetTrainSet().ids, dataset->GetTrainSet().len, FLAGS_batch_size);
    for (int epoch = 0; epoch < FLAGS_num_epoch; epoch++) {
        shuffler.Shuffle();
        for (int step = 0; shuffler.HasNext(); step++) {
            auto tic = std::chrono::system_clock::now();
            NodesBatch batch = shuffler.GetNextBatch();
            std::vector<std::shared_ptr<Block>> blocks = SampleMultiHops(dataset, batch, task);
            auto toc = std::chrono::system_clock::now();

            std::chrono::duration<double> duration = toc - tic;

            printf("Epoch %d, step %d, time %.4f\n", epoch, step, duration.count());
            for (int bid = 0; bid < task.num_blocks; bid++) {
                printf("  bid %d, num_src_nodes: %lu, num_dst_nodes %lu\n", bid, blocks[bid]->num_src_nodes, blocks[bid]->num_dst_nodes);
            }
        }
    }

    gflags::ShutDownCommandLineFlags();
    return 0;
}