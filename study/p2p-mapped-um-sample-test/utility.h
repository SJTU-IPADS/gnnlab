#include <iostream>
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

void perform_sequential_read(int grid_size, int block_size, cudaStream_t stream, 
    int* arr, int len, int* result, int result_len);
void perform_random_read_int32(int grid_size, int block_size, cudaStream_t stream, 
    int* arr, int len, int* result, int result_len);

enum class MemoryType {
    CPU,
    P2P,
    HostAllocMapped,
    UM_CUDA_CUDA,
    UM_CUDA_PU
};

class Dataset {
public:
    Dataset(string name);
    ~Dataset();
    void cpu();
    void p2p(int local_device, int remote_device);
    void hostAllocMapped(int device);
    void um_cuda_cuda(int local_device, int remote_device);
    void um_cuda_cpu(int device);

    static string root_path;
    
    string name;
    size_t node_num;
    size_t edge_num;
    uint32_t* indptr;
    uint32_t* indices;
private:
    int local_device, remote_device;
    MemoryType mm_type;
    void _free();
};