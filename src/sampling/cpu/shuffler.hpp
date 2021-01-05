#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
struct NodesBatch {
    bool valid = false;
    bool is_last = false;
    size_t num_samples = 0;
    const uint32_t *ids = nullptr;
};

class Shuffler {
public:
    Shuffler(const uint32_t *train_ids, size_t num_ids, size_t batch_size) : samples_ids(train_ids, train_ids + num_ids), batch_size(batch_size) {
        size_t num_batches = (num_ids - 1) / batch_size + 1;
        max_batch_id = num_batches - 1;
        last_batch_size = num_ids - num_batches * batch_size;
        if (last_batch_size == 0) {
            last_batch_size = batch_size;
        }
    }

    void Shuffle() {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(samples_ids.begin(), samples_ids.end(), std::default_random_engine(seed));
        cur_batch_id = 0;
    }

    bool HasNext() {
        if (cur_batch_id <= max_batch_id) {
            return true;
        } else {
            return false;
        }
    }

    NodesBatch GetNextBatch() {
        NodesBatch batch;
        batch.valid = cur_batch_id <= max_batch_id;
        if (batch.valid) {
            batch.is_last = cur_batch_id == max_batch_id;
            batch.num_samples = batch.is_last ? last_batch_size : batch_size;
            batch.ids = samples_ids.data() + (cur_batch_id * batch_size);
            cur_batch_id++;
        }
        return batch;
    }
    
private:
    std::vector<uint32_t> samples_ids;
    size_t batch_size;
    size_t last_batch_size;
    size_t max_batch_id;
    size_t cur_batch_id;
};
