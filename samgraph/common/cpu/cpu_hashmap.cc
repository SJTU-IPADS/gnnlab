#include "../logging.h"
#include "cpu_hashmap.h"

namespace samgraph {
namespace common {
namespace cpu {

HashTable::HashTable(size_t sz) {
    _oldv2newv.reserve(sz);
}

void HashTable::Fill(const IdType *input, const size_t num_input) {
    for (size_t i = 0; i < num_input; ++i) {
        if (_oldv2newv.find(input[i]) == _oldv2newv.end()) {
            _oldv2newv.insert({input[i], _oldv2newv.size()});
        }
    }
}

void HashTable::GetUnique(IdType *output, size_t num_output) const {
    SAM_CHECK_GE(num_output, _oldv2newv.size());
    for (auto pair : _oldv2newv) {
        output[pair.second] = pair.first;
    }
}

void HashTable::Map(const IdType *src, const IdType *dst, const size_t len, IdType *new_src, IdType *new_dst) {
    for (size_t i = 0; i < len; i++) {
        new_src[i] = _oldv2newv.at(src[i]);
        new_dst[i] = _oldv2newv.at(dst[i]);
    }
}

} // namespace cpu
} // namespace common
} // namespace samgraph
