#include <cstdlib>
#include <cstring>
#include <chrono>

#include "../logging.h"
#include "../config.h"
#include "cpu_hashtable.h"

namespace samgraph {
namespace common {
namespace cpu {

HashTable::HashTable(size_t sz) {
    _table = (Bucket *) malloc(sz * sizeof(Bucket));
    _mapping = (Mapping *) malloc(sz * sizeof(Mapping));

    _map_offset = 0;
    _capacity = sz;
    _version = 0;

    #pragma omp parallel for num_threads(Config::kOmpThreadNum)
    for (size_t i = 0; i < sz; i++) {
        _table[i].version = Config::kEmptyKey;
    }
}

HashTable::~HashTable() {
    free(_table);
    free(_mapping);
}

void HashTable::Populate(const IdType *input, const size_t num_input) {
    auto tic = std::chrono::system_clock::now();
    #pragma omp parallel for num_threads(Config::kOmpThreadNum)
    for (size_t i = 0; i < num_input; i++) {
        IdType id = input[i];
        SAM_CHECK_LT(id, _capacity);
        IdType old_version = __atomic_exchange_n(&_table[id].version, _version, __ATOMIC_ACQ_REL);
        if (old_version < _version) {
            IdType local = __atomic_fetch_add(&_map_offset, 1, __ATOMIC_ACQ_REL);
            _table[id].local = local;
            _mapping[local].global = id;
        }
    }
    auto toc = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = toc - tic;
    SAM_LOG(INFO) << "HashTable::Populate " << duration.count();
}

void HashTable::MapNodes(IdType *output, size_t num_ouput) {
    auto tic = std::chrono::system_clock::now();
    SAM_CHECK_LE(num_ouput, _map_offset);
    #pragma omp parallel for num_threads(Config::kOmpThreadNum)
    for (size_t i = 0; i < num_ouput; i++) {
        output[i] = _mapping[i].global;
    }

    auto toc = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = toc - tic;
    SAM_LOG(INFO) << "HashTable::MapNodes " << duration.count();
}

void HashTable::MapEdges(const IdType *src, const IdType *dst, const size_t len, IdType *new_src, IdType *new_dst) {
    auto tic = std::chrono::system_clock::now();
    #pragma omp parallel for num_threads(Config::kOmpThreadNum)
    for (size_t i = 0; i < len; i++) {
        SAM_CHECK_LT(src[i], _capacity);
        SAM_CHECK_LT(dst[i], _capacity);

        Bucket &bucket0 = _table[src[i]];
        Bucket &bucket1 = _table[dst[i]];

        SAM_CHECK_EQ(bucket0.version, _version);
        SAM_CHECK_EQ(bucket1.version, _version);

        new_src[i] = bucket0.local;
        new_dst[i] = bucket1.local;
    }

    auto toc = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = toc - tic;
    SAM_LOG(INFO) << "HashTable::MapEdges " << duration.count();
}

void HashTable::Clear() {
    _map_offset = 0;
    _version++;
}

} // namespace cpu
} // namespace common
} // namespace samgraph
