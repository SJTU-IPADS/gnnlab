#include <cuda/atomic>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
using namespace std;

#define CUDA_CALL(func)                         \
 {                                              \
    cudaError_t err = func;                     \
    if(err != cudaSuccess) {                    \
        std::cout << __FILE__ << ":" << __LINE__\
             << " " << #func << " "             \
             << cudaGetErrorString(err)         \
             << " errnum " << err;              \
        exit(EXIT_FAILURE);                     \
    }                                           \
 }


__global__ void delay(volatile int* flag) {
    while(!*flag) {
    }
} 

__device__ void spinlock_increase(int* arr, int idx, cuda::atomic<int, cuda::thread_scope_system> &mux) {
    bool leave = false;
    while (!leave) {
        if (!mux.exchange(1, cuda::std::memory_order_acquire)) {
            arr[idx]++;
            leave = true;
            mux.store(0, cuda::std::memory_order_release);
        }
    }
}

__device__ void atomic_wait_increase(int* arr, int idx, cuda::atomic<int, cuda::thread_scope_system> &mux) {
    bool leave = false;
    while (!leave) {
        if (!mux.exchange(1, cuda::std::memory_order_acquire)) {
            arr[idx]++;
            leave = true;
            mux.store(0, cuda::std::memory_order_release);
            mux.notify_all();
        } else {
            mux.wait(1, cuda::std::memory_order_acquire);
        }
    }
}

__device__ void race_increase(int* arr, int idx, cuda::atomic<int, cuda::thread_scope_system> &mux) {
    arr[idx]++;
}

template<auto sync_increase>
__global__ void thread_race(
    int *arr, 
    int len, 
    int times, 
    cuda::atomic<int, cuda::thread_scope_system>* mux,
    int elem_per_mux,
    int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wtid = idx  % 32;
    int stride = blockDim.x * gridDim.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    for (int t = 0; t < times; t++) {
        auto index = curand(&state) % len;
        bool leave = false;
        auto mux_idx = index / elem_per_mux;
        sync_increase(arr, index, mux[mux_idx]);
    }
}

template<auto sync_increase>
__global__ void device_race(
    int *arr, 
    int len, 
    int times, 
    cuda::atomic<int, cuda::thread_scope_system>* mux,
    int elem_per_mux,
    int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < len; i += stride) {
        auto mux_idx = i / elem_per_mux;
        sync_increase(arr, i, mux[mux_idx]);
    }
}

template<auto sync_increase>
__global__ void sparse_device_race(
    int* arr,
    int len,
    int times,
    cuda::atomic<int, cuda::thread_scope_system>* mux,
    int elem_per_mux,
    int margin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx * margin; i < len; i += stride * margin) {
        auto mux_idx = i / elem_per_mux;
        sync_increase(arr, i, mux[mux_idx]);
    }
}

enum class KernelType {
    Race_thread,
    AtomicWait_thread,
    SpinLock_thread,

    Race_device,
    AtomicWait_device,
    SpinLock_device,

    SparseRace_device,
    SparseAtomicWait_device,
    SparseSpinLock_device
};

template<typename Fn>
struct Kernel {
    constexpr Kernel(KernelType type, Fn value) : type(type), value(value), ok(false) {}
    KernelType type;
    Fn value;
    bool ok;
    float ms;
    friend ostream& operator<<(ostream& os, const Kernel & kernel) {
        switch (kernel.type)
        {
        case KernelType::Race_thread:
            os << "Race_thread";
            break;;
        case KernelType::AtomicWait_thread:
            os << "AtomicWait_thread";
            break;
        case KernelType::SpinLock_thread:
            os << "SpinLock_thread";
            break;
        case KernelType::Race_device:
            os << "Race_device";
            break;
        case KernelType::AtomicWait_device:
            os << "AtomicWait_device";
            break;
        case KernelType::SpinLock_device:
            os << "SpinLock_device";
            break;
        case KernelType::SparseRace_device:
            os << "SparseRace_device";
            break;
        case KernelType::SparseAtomicWait_device:
            os << "SparseAtomicWait_device";
            break;
        case KernelType::SparseSpinLock_device:
            os << "SparseSpinLock_device";
            break;
        default:
            assert(false);
        }
        return os;
    }
};

void test(int len, int elem_per_mux, int repeat = 10) {
    assert(len % elem_per_mux == 0);
    int mux_num = len / elem_per_mux;
    int* arr;
    int* flag;
    cuda::atomic<int, cuda::thread_scope_system>* mut;
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMallocManaged(&mut, sizeof(mut) * mux_num));
    CUDA_CALL(cudaMemAdvise(mut, sizeof(mut) * mux_num, cudaMemAdviseSetPreferredLocation, 0));
    CUDA_CALL(cudaMemAdvise(mut, sizeof(mut) * mux_num, cudaMemAdviseSetAccessedBy, 0));
    CUDA_CALL(cudaMemAdvise(mut, sizeof(mut) * mux_num, cudaMemAdviseSetAccessedBy, 1));
    CUDA_CALL(cudaMallocManaged(&arr, sizeof(int) * len));
    CUDA_CALL(cudaMemAdvise(arr, sizeof(int) * len / 2, cudaMemAdviseSetPreferredLocation, 0));
    CUDA_CALL(cudaMemAdvise(arr + len / 2, sizeof(int) * len / 2, cudaMemAdviseSetPreferredLocation, 1));
    CUDA_CALL(cudaMemAdvise(arr, sizeof(int) * len, cudaMemAdviseSetAccessedBy, 0));
    CUDA_CALL(cudaMemAdvise(arr, sizeof(int) * len, cudaMemAdviseSetAccessedBy, 1));
    
    CUDA_CALL(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocPortable));

    cudaStream_t s[2];
    cudaEvent_t start[2], end[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamCreate(&s[i]));
        CUDA_CALL(cudaEventCreate(&start[i]));
        CUDA_CALL(cudaEventCreate(&end[i]));
    }

    int grid = 4, block = 1024;
    int grid_size = grid * block;
    random_device rd;
    mt19937 gen(rd());
    int margin[2] = {3, 5};

    std::vector kernels{
        Kernel{KernelType::Race_thread, thread_race<race_increase>},
        Kernel{KernelType::AtomicWait_thread, thread_race<atomic_wait_increase>},
        Kernel{KernelType::SpinLock_thread, thread_race<spinlock_increase>},
        Kernel{KernelType::Race_device, device_race<race_increase>},
        Kernel{KernelType::AtomicWait_device, device_race<atomic_wait_increase>},
        Kernel{KernelType::SpinLock_device, device_race<spinlock_increase>},
        Kernel{KernelType::SparseRace_device, sparse_device_race<race_increase>},
        Kernel{KernelType::SparseAtomicWait_device, sparse_device_race<atomic_wait_increase>},
        Kernel{KernelType::SparseSpinLock_device, sparse_device_race<spinlock_increase>}
    };

    for (int k = 0; k < kernels.size(); k++) {
        CUDA_CALL(cudaMemset(mut, 0, sizeof(mut) * mux_num));
        for (int i = 0; i < 2; i++) {
            CUDA_CALL(cudaSetDevice(i));
            CUDA_CALL(cudaMemset(arr, 0, sizeof(int) * len));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        for (int i = 0; i < 2; i++) {
            CUDA_CALL(cudaSetDevice(i));
            delay<<<1, 1, 0, s[i]>>>(flag);
            CUDA_CALL(cudaEventRecord(start[i], s[i]));
            switch (kernels[k].type)
            {
            case KernelType::SparseAtomicWait_device:
            case KernelType::SparseRace_device:
            case KernelType::SparseSpinLock_device:
                kernels[k].value<<<grid, block>>>(arr, len, repeat, mut, elem_per_mux, margin[i]);
                break;
            default:
                kernels[k].value<<<grid, block>>>(arr, len, repeat, mut, elem_per_mux, gen());
            }
            CUDA_CALL(cudaEventRecord(end[i], s[i]));
        }
        *flag = 1;
        for (int i = 0; i < 2; i++) {
            CUDA_CALL(cudaSetDevice(i));
            CUDA_CALL(cudaDeviceSynchronize());
        }
        
        for (int i = 0; i < 2; i++) {
            float ms;
            CUDA_CALL(cudaSetDevice(i));
            CUDA_CALL(cudaEventElapsedTime(&ms, start[i], end[i]));
            cout << "gpu:" << i << " " << ms << "ms\n"; 
            // time_ms[k] += ms;
            kernels[k].ms += ms;
        }

        // check result
        size_t sum = 0;
        for (int i = 0; i < len; i++) {
            sum += arr[i];
        }
        size_t exp;
        switch (kernels[k].type)
        {
        case KernelType::Race_thread:
        case KernelType::SpinLock_thread:
        case KernelType::AtomicWait_thread:
            exp = (size_t)grid_size * repeat * 2;
            break;
        case KernelType::Race_device:
        case KernelType::SpinLock_device:
        case KernelType::AtomicWait_device:
            exp = (size_t)len * 2;
            break;
        case KernelType::SparseRace_device:
        case KernelType::SparseAtomicWait_device:
        case KernelType::SparseSpinLock_device:
            exp = (size_t)(len + margin[0]) / margin[0] + (len + margin[1]) / margin[1];
            break;
        default:
            assert(false);
        }
        if (sum != exp) {
            cout << "should be " << exp << " but find " << sum << "\n";
            kernels[k].ok = false;
        } else {
            cout << "ok, sum=" << sum << "\n";
            kernels[k].ok = true;
        }
    }

    cout << "test: size=" << len 
         << " elem_per_mux=" << elem_per_mux 
         << " repeat=" << repeat << "\n";
    for (int k = 0; k < kernels.size(); k++) {
        cout << std::setw(20) << kernels[k] << "\t" << std::setw(10) << (kernels[k].ok ? "✓" : "𐄂")<< "\t"; 
        cout << std::setw(10) << std::fixed << std::setprecision(2) << kernels[k].ms << "ms\t";
        switch (kernels[k].type)
        {
        case KernelType::Race_thread:
        case KernelType::SpinLock_thread:
        case KernelType::AtomicWait_thread:
            cout << kernels[k].ms / std::find_if(kernels.begin(), kernels.end(), [](const auto &kernel){
                return kernel.type == KernelType::Race_thread;
            })->ms << "x\n"; 
            break;
        case KernelType::Race_device:
        case KernelType::SpinLock_device:
        case KernelType::AtomicWait_device:
            cout << kernels[k].ms / std::find_if(kernels.begin(), kernels.end(), [](const auto &kernel) {
                return kernel.type == KernelType::Race_device;
            })->ms << "x\n";
            break;
        case KernelType::SparseRace_device:
        case KernelType::SparseSpinLock_device:
        case KernelType::SparseAtomicWait_device:
            cout << kernels[k].ms / std::find_if(kernels.begin(), kernels.end(), [](const auto &kernel) {
                return kernel.type == KernelType::SparseRace_device;
            })->ms << "x\n";
            break;
        default:
            assert(false);
        }
    }

    // free resource
    cudaFreeHost(flag);
    CUDA_CALL(cudaFree(mut));
    CUDA_CALL(cudaFree(arr));
    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamDestroy(s[i]));
        CUDA_CALL(cudaEventDestroy(start[i]));
        CUDA_CALL(cudaEventDestroy(end[i]));
    }
}

int main() {
    test(128, 1);
    test(128, 128);

    test(1024, 1);
    test(1024, 1024);

    test(1024 * 1024, 1);
    test(1024 * 1024, 1024);

    test(1024 * 1024 * 1024, 1);
}

