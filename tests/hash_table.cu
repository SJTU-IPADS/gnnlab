#include <iostream>
#include <cstdio>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "common.h"
#include "hash_table.cuh"


class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable {
public:
 typedef typename DeviceOrderedHashTable::Bucket *Iterator;

 explicit MutableDeviceOrderedHashTable(OrderedHashTable *const hostTable)
     : DeviceOrderedHashTable(hostTable->DeviceHandle()) {}

 inline __device__ Iterator Search(const unsigned int id) {
   const unsigned int pos = SearchForPosition(id);

   return GetMutable(pos);
 }

 inline __device__ bool AttemptInsertAt(const unsigned int pos, const unsigned int id,
                                        const unsigned int index, const unsigned int version) {
   const unsigned int key = atomicCAS(&GetMutable(pos)->key, kEmptyKey, id);
   if (key == kEmptyKey || key == id) {
     atomicMin(&GetMutable(pos)->index, index);
     atomicCAS(&GetMutable(pos)->version, kEmptyKey, version);
     return true;
   } else {
     // we need to search elsewhere
     return false;
   }
 }

 inline __device__ Iterator Insert(const unsigned int id, const unsigned int index, const unsigned int version) {
   size_t pos = Hash(id);

   // linearly scan for an empty slot or matching entry
   unsigned int delta = 1;
   while (!AttemptInsertAt(pos, id, index, version)) {
     pos = Hash(pos + delta);
     delta += 1;
   }

   return GetMutable(pos);
 }

private:
 inline __device__ Iterator GetMutable(const size_t pos) {
   assert(pos < this->_size);
   // The parent class Device is read-only, but we ensure this can only be
   // constructed from a mutable version of OrderedHashTable, making this
   // a safe cast to perform.
   return const_cast<Iterator>(this->_table + pos);
 }
};

/**
* @brief Calculate the number of buckets in the hashtable. To guarantee we can
* fill the hashtable in the worst case, we must use a number of buckets which
* is a power of two.
* https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
*/
size_t TableSize(const size_t num, const size_t scale) {
 const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
 return next_pow2 << scale;
}

template <typename T>
struct BlockPrefixCallbackOp {
 T _running_total;

 __device__ BlockPrefixCallbackOp(const T running_total)
     : _running_total(running_total) {}

 __device__ T operator()(const T block_aggregate) {
   const T old_prefix = _running_total;
   _running_total += block_aggregate;
   return old_prefix;
 }
};

template <unsigned int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_duplicates(const unsigned int *const items,
                           const size_t num_items,
                           unsigned int version,
                           MutableDeviceOrderedHashTable table) {
 assert(BLOCK_SIZE == blockDim.x);

 const size_t block_start = TILE_SIZE * blockIdx.x;
 const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
 for (size_t index = threadIdx.x + block_start; index < block_end;
      index += BLOCK_SIZE) {
   if (index < num_items) {
     printf("block %d, thread %d insert %lu with val %d\n", blockIdx.x, threadIdx.x, index, items[index]);
     table.Insert(items[index], index, version);
   }
 }
}

template <unsigned int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const unsigned int *const items,
                                       const size_t num_items,
                                       MutableDeviceOrderedHashTable table,
                                       unsigned int global_offset,
                                       unsigned int version) {
 assert(BLOCK_SIZE == blockDim.x);

 using Iterator = typename MutableDeviceOrderedHashTable::Iterator;

 const size_t block_start = TILE_SIZE * blockIdx.x;
 const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
 for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
   if (index < num_items) {
     const Iterator pos = table.Insert(items[index], index, version);

     // since we are only inserting unique items, we know their local id
     // will be equal to their index
     pos->local = global_offset + static_cast<unsigned int>(index);
   }
 }
}

template <unsigned int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(const unsigned int *items, const size_t num_items,
                             DeviceOrderedHashTable table,
                             const unsigned int version,
                             unsigned int *const num_unique) {
 assert(BLOCK_SIZE == blockDim.x);

 using BlockReduce = typename cub::BlockReduce<unsigned int, BLOCK_SIZE>;
 using Bucket = typename DeviceOrderedHashTable::Bucket;

 const size_t block_start = TILE_SIZE * blockIdx.x;
 const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

 size_t count = 0;

#pragma unroll
 for (size_t index = threadIdx.x + block_start; index < block_end;
      index += BLOCK_SIZE) {
   if (index < num_items) {
     const Bucket &bucket = *table.Search(items[index]);
     printf("block %d, thread %d insert %lu with val %d, target index is %d\n", blockIdx.x, threadIdx.x, index, items[index], bucket.index);
     if (bucket.index == index && bucket.version == version) {
       ++count;
     }
   }
 }

 __shared__ typename BlockReduce::TempStorage temp_space;

 count = BlockReduce(temp_space).Sum(count);

 if (threadIdx.x == 0) {
   printf("block %d insert %lu elements\n", blockIdx.x, count);
   num_unique[blockIdx.x] = count;
   if (blockIdx.x == 0) {
     num_unique[gridDim.x] = 0;
   }
 }
}

template <unsigned int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(const unsigned int *const items,
                               const size_t num_items,
                               MutableDeviceOrderedHashTable table,
                               const unsigned int *const num_items_prefix,
                               unsigned int * const mapping,
                               size_t *const num_unique_items,
                               unsigned int global_offset,
                               unsigned int version) {
 assert(BLOCK_SIZE == blockDim.x);

 using FlagType = size_t;
 using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
 using Bucket = typename DeviceOrderedHashTable::Bucket;

 constexpr size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

 __shared__ typename BlockScan::TempStorage temp_space;

 const size_t offset = num_items_prefix[blockIdx.x];

 BlockPrefixCallbackOp<FlagType> prefix_op(0);

 // count successful placements
 for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
   const size_t index =
       threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

   FlagType flag;
   Bucket *kv;
   if (index < num_items) {
     kv = table.Search(items[index]);
     flag = kv->version == version && kv->index == index;
   } else {
     flag = 0;
   }

   if (!flag) {
       kv = nullptr;
   }

   BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
   __syncthreads();

   if (kv) {
        const unsigned int pos = global_offset + offset + flag;
        kv->local = pos;
        mapping[pos] = items[index];
        printf("Insert mapping %u with val %u\n", pos, items[index]);
   }
 }

 if (threadIdx.x == 0 && blockIdx.x == 0) {
   *num_unique_items = global_offset + num_items_prefix[gridDim.x];
 }
}

DeviceOrderedHashTable::DeviceOrderedHashTable(const Bucket *const table,
                                              const size_t size)
   : _table(table), _size(size) {}

DeviceOrderedHashTable OrderedHashTable::DeviceHandle() const {
 return DeviceOrderedHashTable(_table, _size);
}

OrderedHashTable::OrderedHashTable(const size_t size, unsigned int device, cudaStream_t stream, const size_t scale)
   : _table(nullptr), _size(TableSize(size, scale)), _device(device), _global_version(0), _global_offset(0) {
 // make sure we will at least as many buckets as items.
 CUDA_CALL(cudaMalloc(&_table, sizeof(Bucket) * _size));
 CUDA_CALL(cudaMalloc(&_local_mapping, sizeof(unsigned int) * size));

 printf("fuck size %lu\n", size);

 CUDA_CALL(cudaMemsetAsync(_table, (int)kEmptyKey,
                           sizeof(Bucket) * _size, stream));
 CUDA_CALL(cudaMemsetAsync(_local_mapping, (int)kEmptyKey,
                           sizeof(unsigned int) * size, stream));
}

OrderedHashTable::~OrderedHashTable() { 
    CUDA_CALL(cudaFree(_table));
    CUDA_CALL(cudaFree(_local_mapping));
}

void OrderedHashTable::FillWithDuplicates(const unsigned int *const input,
                                         const size_t num_input,
                                         unsigned int *const unique,
                                         size_t &num_unique,
                                         cudaStream_t stream) {
 const size_t num_tiles = (num_input + kCudaTileSize - 1) / kCudaTileSize;

 const dim3 grid(num_tiles);
 const dim3 block(kCudaBlockSize);

 auto device_table = MutableDeviceOrderedHashTable(this);

 generate_hashmap_duplicates<kCudaBlockSize, kCudaTileSize>
     <<<grid, block, 0, stream>>>(input, num_input, _global_version, device_table);
 CUDA_CALL(cudaGetLastError());

 unsigned int *item_prefix;
 CUDA_CALL(cudaMalloc(&item_prefix, sizeof(unsigned int) * (num_input + 1)));

 count_hashmap<kCudaBlockSize, kCudaTileSize>
     <<<grid, block, 0, stream>>>(input, num_input, device_table, _global_version, item_prefix);
 CUDA_CALL(cudaGetLastError());

 size_t workspace_bytes;
 CUDA_CALL(cub::DeviceScan::ExclusiveSum(
     nullptr, workspace_bytes, static_cast<unsigned int *>(nullptr),
     static_cast<unsigned int *>(nullptr), grid.x + 1, stream));
 void *workspace;
 CUDA_CALL(cudaMalloc(&workspace, workspace_bytes));

 CUDA_CALL(cub::DeviceScan::ExclusiveSum(
     workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1, stream));

 CUDA_CALL(cudaFree(workspace));

 size_t *d_num_unique;
 CUDA_CALL(cudaMalloc(&d_num_unique, sizeof(size_t)));
 compact_hashmap<kCudaBlockSize, kCudaTileSize><<<grid, block, 0, stream>>>(
     input, num_input, device_table, item_prefix, _local_mapping, d_num_unique, _global_offset, _global_version);
 CUDA_CALL(cudaGetLastError());
 CUDA_CALL(cudaFree(item_prefix));
 CUDA_CALL(cudaMemcpyAsync(&num_unique, d_num_unique, sizeof(size_t), cudaMemcpyDeviceToHost));
 CUDA_CALL(cudaStreamSynchronize(stream));
 _global_version++;
 _global_offset = num_unique;

 CUDA_CALL(cudaFree(d_num_unique));
 std::cout << num_unique << std::endl;
 CUDA_CALL(cudaMemcpyAsync(unique, _local_mapping, num_unique * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
}

void OrderedHashTable::FillWithUnique(const unsigned int *const input,
                                     const size_t num_input,
                                     cudaStream_t stream) {
 const size_t num_tiles = (num_input + kCudaTileSize - 1) / kCudaTileSize;

 const dim3 grid(num_tiles);
 const dim3 block(kCudaBlockSize);

 auto device_table = MutableDeviceOrderedHashTable(this);

 generate_hashmap_unique<kCudaBlockSize, kCudaTileSize>
     <<<grid, block, 0, stream>>>(input, num_input, device_table, _global_version, _global_offset);

 CUDA_CALL(cudaGetLastError());

 _global_version++;
 _global_offset += num_input;
}

int main() {
    constexpr int device = 0;

    constexpr size_t num_input = 10;
    unsigned int input [num_input] = {0, 2, 4, 4, 2, 6, 101, 5, 6, 7};
    unsigned int output [num_input];
    size_t num_output;

    constexpr size_t num_input2 = 15;
    unsigned int input2 [num_input2] = {0, 1, 2, 4, 4, 2, 6, 101, 5, 6, 7, 13, 14, 15, 10};
    unsigned int output2 [num_input2];
    size_t num_output2;

    unsigned int *d_input;
    unsigned int *d_input2;

    CUDA_CALL(cudaSetDevice(device));

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CALL(cudaMalloc(&d_input, num_input * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(&d_input2, num_input2 * sizeof(unsigned int)));

    CUDA_CALL(cudaMemcpyAsync(d_input, input, num_input * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(d_input2, input2, num_input2 * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

    OrderedHashTable table(num_input2, device, stream);
    table.FillWithDuplicates(d_input, num_input, output, num_output, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::cout << num_output << std::endl;
    for (size_t i = 0; i < num_output; i++) {
        std::cout << output[i] << ' ';
    }

    std::cout << std::endl << std::endl;

    table.FillWithDuplicates(d_input2, num_input2, output2, num_output2, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::cout << num_output2 << std::endl;
    for (size_t i = 0; i < num_output2; i++) {
        std::cout << output2[i] << ' ';
    }

    std::cout << std::endl;

    CUDA_CALL(cudaFree(d_input));
    CUDA_CALL(cudaFree(d_input2));

    CUDA_CALL(cudaStreamDestroy(stream));
}
