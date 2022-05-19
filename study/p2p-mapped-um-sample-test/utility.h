#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

#define CUDA_CALL(func)                         \
 {                                              \
    cudaError_t err = func;                     \
    if(err != cudaSuccess) {                    \
        cout << __FILE__ << ":" << __LINE__     \
             << " " << #func << " "             \
             << cudaGetErrorString(err)         \
             << " errnum " << err;              \
        exit(EXIT_FAILURE);                     \
    }                                           \
 }

__global__ void delay(volatile int* flag);
__global__ void read(int* arr, int len, int* result, int result_len);
__global__ void random_rand(int* __restrict__ arr, int len, int* result, int result_len, int seed);
__global__ void random_read_overhead(int* __restrict__ arr, int len, int* result, int result_len, int seed);

template<bool same_lkbehind>
__inline__ __device__ size_t get_lookbehind(uint32_t rand, size_t lkbehind) {
    if (same_lkbehind) {
        return lkbehind;
    } else {
        return rand % (lkbehind + 1);
    }
}


template<size_t lkbehind, bool same_lkbehind>
__global__ void random_off_sequentail_lookbehind(int* __restrict__ arr, int len, int* result, int result_len, int seed) {
    // constexpr size_t warp_size = 32;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t wtid = idx % warp_size;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
#pragma unroll(5)
    for (size_t i = 0; i < len; i += grid_size) {
        size_t off = curand(&state) % len;
        // size_t lk = curand(&state) % (lkbehind + 1);
        // size_t lk = lkbehind;
        size_t lk = get_lookbehind<same_lkbehind>(curand(&state), lkbehind);
        for (size_t j = 0; j < lk; j++) {
            result[(rid + j) % result_len] += arr[(off + j) % len];
        }
    }
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t grid_size = blockDim.x * gridDim.x;
//     size_t rid = idx % result_len;
//     curandState state;
//     curand_init(seed + idx, 0, 0, &state);
// #pragma unroll(5)
//     for(size_t i = 0; i < len; i += grid_size) {
//         int x = curand(&state) % len;
//         for (size_t j = 0; j < lkbehind; j++) {
//             result[(rid + j) % result_len] += arr[(x + j) % len];
//         }
//     }
}


template<size_t _page_size, size_t lkbehind, bool same_lkbehind>
__global__ void random_off_random_lookbehind(int* __restrict__ arr, int len, int* result, int result_len, int seed) {
    constexpr int warp_size = 32;
    constexpr size_t page_size = _page_size / sizeof(int);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t wtid = idx % warp_size;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
    curandState t_state;
    curand_init(seed + idx, 0, 0, &t_state);
#pragma unroll(5)
    for (size_t i = 0; i < len; i += grid_size) {
        size_t off = curand(&t_state) % len;
        // size_t lk = curand(&t_state) % (lkbehind + 1);
        // size_t lk = lkbehind;
        size_t lk = get_lookbehind<same_lkbehind>(curand(&t_state), lkbehind);
        for (size_t j = 0; j < lk; j++) {
            result[(rid + j) % result_len] += arr[(off + (curand(&t_state) % page_size)) % len];
        }

        // size_t rand = curand(&t_state) % warp_size;
        // if (wtid < rand) {
        //     for (size_t j = 0; j < rand; j++) {
        //         result[(rid + j) % result_len] += arr[(off + j) % len];
        //     }
        // } else {
        //     rand = curand(&t_state) % warp_size;
        //     for (size_t j = 0; j < wtid; j++) {
        //         // result[(rid + j) % result_len] += arr[curand(&t_state) % len];
        //         // result[(rid + j) % result_len] += arr[(rand + j) % len];
        //         result[(rid + j) % result_len] += arr[(off + (curand(&t_state) % page_size)) % len];
        //     }
        // }
    }
}

template<size_t _page_size, size_t lkbehind, bool same_lkbehind>
__global__ void random_off_divergence_lookbehind(int* __restrict__ arr, int len, int* result, int result_len, int seed) {
    constexpr int warp_size = 32;
    constexpr size_t page_size = _page_size / sizeof(int);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t wtid = idx % warp_size;
    size_t grid_size = blockDim.x * gridDim.x;
    size_t rid = idx % result_len;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
#pragma unroll(5)
    for (size_t i = 0; i < len; i += grid_size) {
        size_t off = curand(&state) % len;
        // size_t lk = curand(&state) % (lkbehind + 1);
        size_t lk = get_lookbehind<same_lkbehind>(curand(&state), lkbehind);
        // size_t lk = lkbehind;
        size_t rand = curand(&state) % warp_size;
        if (wtid < rand) {
            for (size_t j = 0; j < lk; j++) {
                result[(rid + j) % result_len] += arr[(off + j) % len];
            }
        } else {
            for (size_t j = 0; j < lk; j++) {
                result[(rid + j) % result_len] += arr[(off + (curand(&state) % page_size)) % len];
            }
        }
    }
}

// ---------------------------------------------------------------------------------

template<auto kernel>
void perform_kernel_with_seed(
    int grid_size, int block_size, cudaStream_t stream,
    int* arr, int len, int* result, int result_len
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    kernel<<<grid_size, block_size, 0, stream>>>(arr, len, result, result_len, gen());
}

template<auto kernel>
void perform_kernel(
    int grid_size, int block_size, cudaStream_t stream,
    int* arr, int len, int* result, int result_len
) {
    kernel<<<grid_size, block_size, 0, stream>>>(arr, len, result, result_len);
}

void perform_random_read_int32(int grid_size, int block_size, cudaStream_t stream, 
    int* arr, int len, int* result, int result_len);

constexpr auto perform_sequential_read = \
    perform_kernel<read>;
constexpr auto perform_random_read = \
    perform_kernel_with_seed<random_rand>;
constexpr auto perform_random_read_overhead = \
    perform_kernel_with_seed<random_read_overhead>;

template<size_t lkbehind> constexpr auto perform_random_off_sequentail_lookbehind = \
    perform_kernel_with_seed<random_off_sequentail_lookbehind<lkbehind, false>>;
template<size_t page_size, size_t lkbehind> constexpr auto perform_random_off_random_lookbehind = \
    perform_kernel_with_seed<random_off_random_lookbehind<page_size, lkbehind, false>>;
template<size_t page_size, size_t lkbehind> constexpr auto perform_random_off_divergence_lookbehind = \
    perform_kernel_with_seed<random_off_divergence_lookbehind<page_size, lkbehind, false>>;

template<size_t lkbehind> constexpr auto perform_random_off_sequentail_same_lookbehind = \
    perform_kernel_with_seed<random_off_sequentail_lookbehind<lkbehind, true>>;
template<size_t page_size, size_t lkbehind> constexpr auto perform_random_off_random_same_lookbehind = \
    perform_kernel_with_seed<random_off_random_lookbehind<page_size, lkbehind, true>>;
template<size_t page_size, size_t lkbehind> constexpr auto perform_random_off_divergence_same_lookbehind = \
    perform_kernel_with_seed<random_off_divergence_lookbehind<page_size, lkbehind, true>>;


tuple<double, double, double> sum_avg_std(const vector<size_t> &vec);

enum class MemoryType {
    CPU,
    Local,
    P2P,
    HostAllocMapped,
    UM_CUDA_CUDA,
    UM_CUDA_CPU
};

ostream& operator<<(ostream& os, const MemoryType &memory_type);

class Dataset {
public:
    Dataset(string name);
    ~Dataset();
    void cpu();
    void p2p(int local_device, int remote_device);
    void hostAllocMapped(int device);
    void um_cuda_cuda(int local_device, int remote_device);
    void um_cuda_cpu(int device);
    pair<unique_ptr<uint32_t[]>, unique_ptr<uint32_t[]>> get_cpu_graph();

    static string root_path;
    
    string name;
    size_t node_num;
    size_t edge_num;
    size_t train_num;
    uint32_t* indptr;
    uint32_t* indices;
    uint32_t* trainset;
private:
    int local_device, remote_device;
    MemoryType mm_type;
    void _free_graph();
};