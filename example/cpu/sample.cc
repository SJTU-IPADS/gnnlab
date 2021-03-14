#include <vector>
#include <sstream>
#include <memory>
#include <chrono>
#include <cstdio>
#include <iostream>

#include "../../samgraph-cpu/dataset.h"
#include "../../samgraph-cpu/sampler.h"
#include "../../samgraph-cpu/block.h"
#include "../../samgraph-cpu/shuffler.h"

std::string dataset_key = "papers100M";
std::string dataset_folder = "/graph-learning/samgraph/papers100M";
std::string fanout = "15,10,5";
int num_epoch = 1;
uint64_t batch_size = 8192;

int main(int argc, char *argv[]) {
    if (argc > 1) {
        batch_size = std::atoll(argv[1]);
    }
    std::cout << "Batch size : " << batch_size << std::endl; 
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(dataset_key, dataset_folder);

    SamplingTask task;
    std::stringstream ss(fanout);
    for (int i; ss >> i;) {
        task.fanout.push_back(i);
        if (ss.peek() == ',') {
            ss.ignore();
        }
    }
    task.num_blocks = task.fanout.size();

    auto tic = std::chrono::system_clock::now();
    Shuffler shuffler(dataset->GetTrainSet().ids, dataset->GetTrainSet().len, batch_size);
    for (int epoch = 0; epoch < num_epoch; epoch++) {
        shuffler.Shuffle();

        for (int step = 0; shuffler.HasNext(); step++) {
            NodesBatch batch = shuffler.GetNextBatch();
            std::vector<std::shared_ptr<Block>> blocks = SampleMultiHops(dataset, batch, task);

            auto toc = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = toc - tic;

            printf("Sampling one batch uses %.2lf secs\n", duration.count());
            return 0;
        }
    }

    return 0;
}