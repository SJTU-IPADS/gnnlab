#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <parallel_hashmap/phmap.h>

class IdHashMap {
 public:
  IdHashMap() : filter_(kFilterSize, false) {}
  explicit IdHashMap(uint32_t *ids, size_t len): filter_(kFilterSize, false) {
    oldv2newv_.reserve(len);
    Update(ids, len);
  }

  void Update(uint32_t *ids, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      const uint32_t id = ids[i];
      // phmap::flat_hash_map::insert assures that an insertion will not happen if the
      // key already exists.
      oldv2newv_.insert({id, oldv2newv_.size()});
      filter_[id & kFilterMask] = true;
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(uint32_t id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  uint32_t Map(uint32_t id, uint32_t default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

  // Return the new id of each id in the given array.
  uint32_t* Map(uint32_t *ids, uint32_t len, uint32_t default_val) const {
    uint32_t *values = (uint32_t *)malloc(len * sizeof(uint32_t));
    for (int64_t i = 0; i < len; ++i)
      values[i] = Map(ids[i], default_val);
    return values;
  }

  // Return all the old ids collected so far, ordered by new id.
  uint32_t* Values() const {
    uint32_t *values = (uint32_t *)malloc(oldv2newv_.size() * sizeof(uint32_t));
    for (auto pair : oldv2newv_)
      values[pair.second] = pair.first;
    return values;
  }

  inline size_t Size() const {
    return oldv2newv_.size();
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  phmap::flat_hash_map<uint32_t, uint32_t> oldv2newv_;
};
