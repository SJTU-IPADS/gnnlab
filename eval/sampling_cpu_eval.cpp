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
#include "util/performance.hpp"
#include "util/tictoc.hpp"

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

    TicToc t;
    Performance &p = Performance::Instance();
    p.SetMeta(task.num_blocks);
    Shuffler shuffler(dataset->GetTrainSet().ids, dataset->GetTrainSet().len, FLAGS_batch_size);
    for (int epoch = 0; epoch < FLAGS_num_epoch; epoch++) {
        t.Tic(0);
        shuffler.Shuffle();
        p.shuffles.Log(t.Toc(0));

        for (int step = 0; shuffler.HasNext(); step++) {
            t.Tic(1);
            NodesBatch batch = shuffler.GetNextBatch();
            std::vector<std::shared_ptr<Block>> blocks = SampleMultiHops(dataset, batch, task);
            p.steps.Log(t.Toc(1));
            p.ReportStep(epoch, step);
        }

        p.epochs.Log(t.Toc(0));
        p.ReportEpoch();
    }

    gflags::ShutDownCommandLineFlags();
    return 0;
}