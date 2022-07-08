#include "utility.h"
#include <regex>
#include <curand.h>
#include <curand_kernel.h>
#include <cassert>
#include <fstream>
#include <numeric>
#include <iomanip>
using namespace std;

enum class TestKernel {
    PointerChasing,
    Random
};

ostream& operator<<(ostream& os, const TestKernel &kernel) {
    switch (kernel)
    {
    case TestKernel::PointerChasing:
        os << "PointerChasing";
        return os;
    case TestKernel::Random:
        os << "Random";
        return os;
    default:
        assert(false);
    }
}

using elem_t = uint64_t;

// 12G buf, bigger than l2 tlb cache coverage(8G)
constexpr size_t array_sz = 1024ULL * 1024ULL * 1024ULL * 12ULL / sizeof(elem_t);
constexpr int local_gpu = 0;
constexpr int remote_gpu = 1; 
size_t clk_rate; // MHz

__global__ void pchasing(elem_t* arr, uint32_t* times, uint32_t* idx_res, uint32_t n) {
    extern __shared__ uint32_t shm[];
    uint32_t* s_times = (uint32_t*)shm;
    uint32_t* s_idx = (uint32_t*)shm + n;

    uint32_t idx = 0;

    for(uint32_t i = 0; i < n; i++) {
        auto start = clock();
        idx = arr[idx];
        s_idx[i] = idx;
        auto end = clock();
        s_times[i] = end - start;
    }

    for(uint32_t i = 0; i < n; i++) {
        times[i] = s_times[i];
        idx_res[i] = s_idx[i];
    }
}

__global__ void random_access(elem_t* arr, uint32_t* times, uint32_t* idx_res, uint32_t n, uint32_t range) {
    extern __shared__ uint32_t shm[];
    uint32_t *s_times  = (uint32_t*)shm;
    uint32_t *s_idx = (uint32_t*)shm + n;

    curandState state;
    auto seed = clock();
    curand_init(seed, 0, 0, &state);
    auto idx = curand(&state) % range;

    for(uint32_t i = 0; i < n; i++) {
        idx = curand(&state) % range;
        auto start = clock();
        idx = arr[idx];
        s_idx[i] = idx;
        auto end = clock();
        s_times[i] = end - start;
    }

    for(uint32_t i = 0; i < n; i++) {
        times[i] = s_times[i];
        idx_res[i] = s_idx[i];
    }
}   

// overhead testing
__global__ void curand_latency(uint32_t* times, uint32_t* idx_res, uint32_t n, uint32_t range) {
    extern __shared__ uint32_t shm[];
    uint32_t *s_times  = (uint32_t*)shm;
    uint32_t *s_idx = (uint32_t*)shm + n;

    curandState state;
    auto seed = clock();
    curand_init(seed, 0, 0, &state);
    auto rand = curand(&state) % range;

    for(uint32_t i = 0; i < n; i++) {
        auto start = clock();
        s_idx[i] = curand(&state) % range;
        auto end = clock();
        s_times[i] = end - start;
    }

    for(uint32_t i = 0; i < n; i++) {
        times[i] = s_times[i];
        idx_res[i] = s_idx[i];
    }
}

__global__ void shared_memory_latency(uint32_t* times, uint32_t* idx_res, uint32_t n) {
    extern __shared__ uint32_t shm[];
    uint32_t *s_times  = (uint32_t*)shm;
    uint32_t *s_idx = (uint32_t*)shm + n;
    
    for(uint32_t i = 0; i < n; i++) {
        auto start = clock();
        s_idx[i] = i;
        auto end = clock();
        s_times[i] = end - start;
    }

    for(uint32_t i = 0; i < n; i++) {
        times[i] = s_times[i];
        idx_res[i] = s_idx[i];
    }
}   

void test_overhead(uint32_t n, uint32_t range) {
    uint32_t *times, *cpu_times, *index;
    cpu_times = new uint32_t[n];
    CUDA_CALL(cudaMalloc(&times, sizeof(uint32_t) * n));
    CUDA_CALL(cudaMalloc(&index, sizeof(uint32_t) * n));

    double shm_latency = 0;
    double rand_latency = 0;

    shared_memory_latency<<<1, 1, sizeof(uint32_t) * n * 2, 0>>>(times, index, n);
    CUDA_CALL(cudaMemcpy(cpu_times, times, sizeof(uint32_t) * n, cudaMemcpyDefault));
    for(uint32_t i = 0; i < n; i++) {
        shm_latency += cpu_times[i];
    }
    shm_latency /= n;
    curand_latency<<<1, 1, sizeof(uint32_t) * n * 2, 0>>>(times, index, n, range);
    CUDA_CALL(cudaMemcpy(cpu_times, times, sizeof(uint32_t) * n, cudaMemcpyDefault));
    for(uint32_t i = 0; i < n; i++) {
        rand_latency += cpu_times[i];
    }
    rand_latency /= n;
    rand_latency -= shm_latency;

    cout << "share memory lantency | " << shm_latency << " clk, " << shm_latency / clk_rate << " us\n";
    cout << "      curand lantency | " << rand_latency << " clk, "  << rand_latency / clk_rate << " us\n";

    delete[] cpu_times;
    CUDA_CALL(cudaFree(times));
    CUDA_CALL(cudaFree(index));
}

int main(int arg, char **argv) {
    MemoryType memory_type = MemoryType::Local;
    TestKernel kernel = TestKernel::PointerChasing;
    size_t stride = 1;
    size_t n = 128;
    size_t range = 1024 * 1024 * 256 / sizeof(elem_t);
    bool overhead_testing = false;
    for (int i = 0; i < arg; i++) {
        if (!strcmp(argv[i], "-r")) {
            if (!strcmp(argv[i+1], "gpu")) {
                memory_type = MemoryType::UM_CUDA_CUDA;
            } else if (!strcmp(argv[i+1], "local")) {
                memory_type = MemoryType::Local;
            } else if (!strcmp(argv[i+1], "cpu")) {
                memory_type = MemoryType::UM_CUDA_CPU;
            } else {
                cout << "bad arg: " << argv[i+1] << "\n";
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (!strcmp(argv[i], "-k")) {
            if (!strcmp(argv[i+1], "pchasing")) {
                kernel = TestKernel::PointerChasing;
            } else if (!strcmp(argv[i+1], "random")) {
                kernel = TestKernel::Random;
            } else {
                cout << "bad arg: " << argv[i+1] << "\n";
                exit(EXIT_FAILURE);
            }
            i++;
        } else if (!strcmp(argv[i], "-s")) {
            regex reg("([0-9]+)(B|b|M|m)");
            smatch match;
            string str(argv[i+1]);
            if (!regex_match(str, match, reg)) {
                cout << "bad arg: " << argv[i+1] << "\n";
                exit(EXIT_FAILURE);
            }
            stride = std::stoull(match[1]);
            if (match[2] == "M" || match[2] == "m") {
                stride *= 1024 * 1024;
            }
            if (stride % sizeof(elem_t) != 0) {
                cout << "bad arg: " << argv[i+1] << "\n";
                exit(EXIT_FAILURE);
            }
            stride /= sizeof(elem_t);
            i++;
        } else if (!strcmp(argv[i], "-n")) {
            n = std::stoull(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i], "-overhead")) {
            overhead_testing = true;
        } else if (!strcmp(argv[i], "-range")) {
            regex reg("([0-9]+)(M|m|g|G)");
            smatch match;
            string str(argv[i+1]);
            if (!regex_match(str, match, reg)) {
                cout << "bad arg: " << argv[i+1] << "\n";
                exit(EXIT_FAILURE);
            }
            range = std::stoul(match[1]);
            if (match[2] == "M" || match[2] == 'm') {
                range *= 1024 * 1024;
            } else if (match[2] == "G" || match[2] == "g") {
                range *= 1024ULL * 1024ULL * 1024ULL;
            }
            assert(range % sizeof(elem_t) == 0);
            range /= sizeof(elem_t); 
            i++;
        }
    }
    
    cudaDeviceProp device_prob;
    cudaGetDeviceProperties(&device_prob, 0);
    clk_rate = device_prob.clockRate / 1000;

    if (overhead_testing) {
        test_overhead(n, range);
        return 0;
    }

    elem_t *arr, *cpu_arr;
    uint32_t *times, *cpu_times, *index;
    cpu_arr = new elem_t[array_sz];
    cpu_times = new uint32_t[n];
    CUDA_CALL(cudaSetDevice(local_gpu));
    CUDA_CALL(cudaMalloc(&times, n * sizeof(uint32_t)));
    CUDA_CALL(cudaMalloc(&index, n * sizeof(uint32_t)));
    switch (memory_type)
    {
    case MemoryType::UM_CUDA_CPU:
        CUDA_CALL(cudaMallocManaged(&arr, array_sz * sizeof(elem_t)));
        CUDA_CALL(cudaMemAdvise(arr, array_sz * sizeof(elem_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_CALL(cudaMemAdvise(arr, array_sz * sizeof(elem_t), cudaMemAdviseSetAccessedBy, local_gpu));
        break;
    case MemoryType::UM_CUDA_CUDA:
        CUDA_CALL(cudaMallocManaged(&arr, array_sz * sizeof(elem_t)));
        CUDA_CALL(cudaMemAdvise(arr, array_sz * sizeof(elem_t), cudaMemAdviseSetPreferredLocation, remote_gpu));
        CUDA_CALL(cudaMemAdvise(arr, array_sz * sizeof(elem_t), cudaMemAdviseSetAccessedBy, local_gpu));
        break;
    case MemoryType::Local:
        CUDA_CALL(cudaMalloc(&arr, array_sz * sizeof(elem_t)));
        break;
    default:
        assert(false);
        break;
    }
    for (size_t i = 0; i < stride * n; i += stride) {
        cpu_arr[i] = (i + stride) % (n * stride);
    }
    CUDA_CALL(cudaMemcpy(arr, cpu_arr, array_sz * sizeof(elem_t), cudaMemcpyDefault));

    {
        switch (kernel)
        {
        case TestKernel::PointerChasing:
            pchasing<<<1, 1, sizeof(uint32_t) * n * 2, 0>>>(arr, times, index, n);
            break;
        case TestKernel::Random:
            random_access<<<1, 1, sizeof(uint32_t) * n * 2, 0>>>(arr, times, index, n, range);
            break;
        default:
            assert(false);
        }
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy(cpu_times, times, n * sizeof(uint32_t), cudaMemcpyDefault));
        stringstream ss;
        auto size_str = [](size_t sz) -> string {
            if (sz < 1024) {
                return std::to_string(sz) + "B";
            } else if (sz < 1024 * 1024) {
                return std::to_string(sz / 1024) + "K";
            } else if (sz < 1024 * 1024 * 1024) {
                return std::to_string(sz / 1024 / 1024) + "M";
            } else {
                return std::to_string(sz / 1024 / 1024 / 1024) + "G";
            }
        };
        ss << memory_type << "-" << kernel 
           << "-stride:" << size_str(stride * sizeof(elem_t)) << "-n:" << n 
           << "-range:" << size_str(range * sizeof(elem_t)) << ".bin";
        ofstream ofs(ss.str(), ios::binary);
        assert(ofs);
        ofs.write((char*)cpu_times, sizeof(uint32_t) * n);
        ofs.close();
        assert(ofs.good());

        auto clk_avg = 1.0 * std::accumulate(cpu_times, cpu_times + n, 0) / n;
        auto clk_min = std::accumulate(cpu_times, cpu_times + n, INT32_MAX, [](uint32_t init, uint32_t first) {
            return min(init, first);
        });
        auto clk_max = std::accumulate(cpu_times, cpu_times + n, 0, [](uint32_t init, uint32_t first) {
            return max(init, first);
        });
        cout << "write test result to: " << ss.str() << '\n';
        cout << " statistic result | " << setw(10) << " clk " << setw(5) << "|" << setw(10) << " us " << "\n";
        cout << "       avg        | " << setw(10) << clk_avg << setw(5) << "|" << setw(10) << 1.0 * clk_avg / clk_rate << "\n";
        cout << "       min        | " << setw(10) << clk_min << setw(5) << "|" << setw(10) << 1.0 * clk_min / clk_rate << "\n";
        cout << "       max        | " << setw(10) << clk_max << setw(5) << "|" << setw(10) << 1.0 * clk_max / clk_rate << "\n";
    }

    delete[] cpu_arr;
    delete[] cpu_times;
    CUDA_CALL(cudaFree(times));
    CUDA_CALL(cudaFree(index));
    switch (memory_type)
    {
    case MemoryType::UM_CUDA_CPU:
    case MemoryType::UM_CUDA_CUDA:
    case MemoryType::Local:
        CUDA_CALL(cudaFree(arr));
        break;
    default:
        assert(false);
    }

}