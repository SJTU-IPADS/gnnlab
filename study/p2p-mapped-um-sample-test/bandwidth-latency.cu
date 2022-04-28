#include <bits/stdc++.h>

#include "utility.h"
using namespace std;

struct Array
{
    int* data;
    size_t size;
};

class MemoryTestCase {
public:
    string name;
    int device;
    Array read_buf;
    Array read_result;
    cudaStream_t stream;
    MemoryTestCase(string name, int device, size_t read_buf_size, size_t read_result_size = 10) 
        : device(device), name(name), 
        read_buf_size(read_buf_size), read_result_size(read_result_size), 
        finished(false), time(-1) {}
    virtual ~MemoryTestCase() {}

    virtual void Init() {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CALL(cudaMalloc(&read_result.data, read_result_size * sizeof(int)));
        read_result.size = read_result_size;
    }
    virtual void Clear(float time) {
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaStreamDestroy(stream));
        CUDA_CALL(cudaFree(read_result.data));
        read_result.data = nullptr;
        read_result.size = 0;
        finished = true;
        this->time = time;
    }

    size_t ReadSize() const {
        if (!finished) {
            cout << "Warning: " << name << " Get ReadSize before test finish\n";
        }
        return sizeof(int) * read_buf_size;
    }
    double Time() const {
        if (!finished) {
            cout << "Warning: " << name << " Get time before test finish\n";
        }
        return time;
    }
protected:
    bool finished;
    const size_t read_buf_size;
    const size_t read_result_size;
    double time; // ms
};

class Local : public MemoryTestCase {
public:
    Local(int device, size_t elem_num)
        : MemoryTestCase("Local", device, elem_num) {}

    virtual void Init() override {
        MemoryTestCase::Init();
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMalloc(&read_buf.data, read_buf_size * sizeof(int)));
        read_buf.size = read_buf_size;
        CUDA_CALL(cudaMemset(read_buf.data, 0x0f, sizeof(int) * read_buf.size));
    }
    virtual void Clear(float time) override {
        MemoryTestCase::Clear(time);
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaFree(read_buf.data));
        read_buf.data = nullptr;
        read_buf.size = 0;
    }
};

class P2P : public MemoryTestCase {
public:
    int remote_device;
    P2P(int local_device, int remote_device, size_t elem_num) 
        : MemoryTestCase("P2P", local_device, elem_num), remote_device(remote_device) {
    }
    
    virtual void Init() override {
        MemoryTestCase::Init();
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaMalloc(&read_buf.data, read_buf_size * sizeof(int)));
        read_buf.size = read_buf_size;
        CUDA_CALL(cudaMemset(read_buf.data, 0x0f, sizeof(int) * read_buf.size));
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaDeviceEnablePeerAccess(remote_device, 0));
    }
    virtual void Clear(float time) override {
        MemoryTestCase::Clear(time);
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaDeviceDisablePeerAccess(remote_device));
        CUDA_CALL(cudaSetDevice(remote_device));
        CUDA_CALL(cudaFree(read_buf.data));
        read_buf.data = nullptr;
        read_buf.size = 0;
    }
};

class HostMapped : public MemoryTestCase {
public:
    HostMapped(int device, size_t elem_num)
        : MemoryTestCase("cudaHostAllocMapped", device, elem_num) {}
    
    virtual void Init() override {
        MemoryTestCase::Init();
        CUDA_CALL(cudaHostAlloc(&read_buf.data, sizeof(int) * read_buf_size, cudaHostAllocMapped));
        read_buf.size = read_buf_size;
        memset(read_buf.data, 0x0f, sizeof(int) * read_buf.size);
    }
    virtual void Clear(float time) override {
        MemoryTestCase::Clear(time);
        CUDA_CALL(cudaFreeHost(read_buf.data));
        read_buf.data = nullptr;
        read_buf.size = 0;
    }
};

class UM_CUDA_CUDA : public MemoryTestCase {
public:
    int remote_device;
    UM_CUDA_CUDA(int local_device, int remote_device, size_t elem_num)
        : MemoryTestCase("UM-cuda+cuda", local_device, elem_num), remote_device(remote_device) {}

    virtual void Init() override {
        MemoryTestCase::Init();
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMallocManaged(&read_buf.data, sizeof(int) * read_buf_size));
        read_buf.size = read_buf_size;
        CUDA_CALL(cudaMemAdvise(read_buf.data, sizeof(int) * read_buf.size, 
            cudaMemAdviseSetPreferredLocation, remote_device));
        CUDA_CALL(cudaMemAdvise(read_buf.data, sizeof(int) * read_buf.size,
            cudaMemAdviseSetAccessedBy, device));

        auto ptr = make_unique<int[]>(read_buf.size);
        memset(ptr.get(), 0x0f, sizeof(int) * read_buf.size);
        CUDA_CALL(cudaMemcpy(read_buf.data, ptr.get(), sizeof(int) * read_buf.size, cudaMemcpyHostToDevice));
    }
    virtual void Clear(float time) override {
        MemoryTestCase::Clear(time);
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaFree(read_buf.data));
        read_buf.data = nullptr;
        read_buf.size = 0;
    }
};

class UM_CUDA_CPU : public MemoryTestCase {
public:
    UM_CUDA_CPU(int device, size_t elem_num)
        : MemoryTestCase("UM-cuda+cpu", device, elem_num) {}
    
    virtual void Init() override {
        MemoryTestCase::Init();
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaMallocManaged(&read_buf.data, sizeof(int) * read_buf_size));
        read_buf.size = read_buf_size;
        CUDA_CALL(cudaMemAdvise(read_buf.data, sizeof(int) * read_buf.size,
            cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_CALL(cudaMemAdvise(read_buf.data, sizeof(int) * read_buf.size,
            cudaMemAdviseSetAccessedBy, device));

        auto ptr = make_unique<int[]>(read_buf.size);
        memset(ptr.get(), 0x0f, sizeof(int) * read_buf.size);
        CUDA_CALL(cudaMemcpy(read_buf.data, ptr.get(), sizeof(int) * read_buf_size, cudaMemcpyHostToDevice));
    }
    virtual void Clear(float time) override {
        MemoryTestCase::Clear(time);
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaFree(read_buf.data));
        read_buf.data = nullptr;
        read_buf.size = 0;
    }
}; 

template<size_t elem_num, auto perform_read>
class SMReadTest {
public:
    SMReadTest(int local_device, int remote_deivce, int repeat = 5) 
    : repeat(repeat) {
        // check p2p is enble, which will have effect on um-cuda+cuda and p2p test case
        int access = 0;
        CUDA_CALL(cudaDeviceCanAccessPeer(&access, local_device, remote_deivce));
        if (!access) {
            cout << "device " << local_device << " " << remote_deivce
                 << " do not support p2p access, abort testing\n";
        }
        env.push_back(make_unique<Local>(local_device, elem_num));
        env.push_back(make_unique<HostMapped>(local_device, elem_num));
        env.push_back(make_unique<P2P>(local_device, remote_deivce, elem_num));
        env.push_back(make_unique<UM_CUDA_CUDA>(local_device, remote_deivce, elem_num));
        env.push_back(make_unique<UM_CUDA_CPU>(local_device, elem_num));
    }
    void Run() {
        volatile int* start_flag;
        cudaHostAlloc(&start_flag, sizeof(int), cudaHostAllocPortable);
        vector<cudaEvent_t> start(env.size()), end(env.size());
        for(int i = 0; i < env.size(); i++) {
            // cout << env[i]->name << " " << env[i]->device << "\n";
            CUDA_CALL(cudaSetDevice(env[i]->device));
            CUDA_CALL(cudaEventCreate(&start[i]));
            CUDA_CALL(cudaEventCreate(&end[i]));
        }
        int block_size = 0, grid_size = 0;
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&block_size, &grid_size, read));
        for(int i = 0; i < env.size(); i++) {
            report_process(i);
            *start_flag = 0;
            env[i]->Init();
            CUDA_CALL(cudaSetDevice(env[i]->device));
            CUDA_CALL(cudaStreamSynchronize(env[i]->stream));
            delay<<<1, 1, 0, env[i]->stream>>>(start_flag);
            CUDA_CALL(cudaEventRecord(start[i], env[i]->stream));
            for(int r = 0; r < repeat; r++) {
                // read<<<grid_size, block_size, 0, env[i]->stream>>>(
                //     env[i]->read_buf.data, env[i]->read_buf.size, 
                //     env[i]->read_result.data, env[i]->read_result.size);
                perform_read(grid_size, block_size, env[i]->stream, 
                    env[i]->read_buf.data, env[i]->read_buf.size, 
                    env[i]->read_result.data, env[i]->read_result.size);
            }
            CUDA_CALL(cudaEventRecord(end[i], env[i]->stream));
            
            *start_flag = 1;
            CUDA_CALL(cudaStreamSynchronize(env[i]->stream));
            float ms;
            CUDA_CALL(cudaEventElapsedTime(&ms, start[i], end[i]));
            env[i]->Clear(ms);
        }
        report_process(env.size());
        cudaFree((void*)start_flag);
        for(int i = 0; i < env.size(); i++) {
            CUDA_CALL(cudaSetDevice(env[i]->device));
            CUDA_CALL(cudaEventDestroy(start[i]));
            CUDA_CALL(cudaEventDestroy(end[i]));
        }
    }
    virtual void Statistic() = 0;
protected:
    vector<unique_ptr<MemoryTestCase>> env;
    int repeat;
    void report_process(int i) {
        int done = 10.0 * i / env.size();
        int todo = 10 - done;
        cout << "\rSMTesetRead<elem_num=" << elem_num << ">: ";
        cout << "[";
        cout << string(done, '#') << string(todo, '.') << "] ";
        cout << i << "/" << env.size() << std::flush;
        if(i == env.size()) {
            cout << "\n";
        }
    }
};

template<size_t elem_num = (1ULL << 28)>
class BandWitdhTest : public SMReadTest<elem_num, perform_sequential_read> {
public:
    BandWitdhTest(int local_device, int remote_device, int repeat = 5)
        : SMReadTest<elem_num, perform_sequential_read>(local_device, remote_device, repeat) {}
    virtual void Statistic() override {
        cout << "BandWidth Test Result:\n";
        cout << "----------------------\n";
        cout.setf(ios::left, ios::adjustfield);
        size_t fwid1 = string{"MemoryType"}.size();
        for(auto &e : this->env) {
            fwid1 = std::max(fwid1, e->name.size());
        }
        size_t fwid2 = string{"BandWidth(GB/s)"}.size();
        fwid1 += 4;
        fwid2 += 4;
        cout.width(fwid1); cout << "MemoryType" << "|  ";
        cout.width(fwid2); cout << "BandWidth(GB/s)" << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto bandwidth = 1.0 * e->ReadSize() / 1024 / 1024 / 1024 / time * 1000; // gb/s
            bandwidth *= this->repeat;
            cout.width(fwid1); cout << e->name << "|  ";
            cout.width(fwid2); cout << bandwidth << "\n";
        }
        cout << "\n";
    }
};

template<size_t elem_num = (1ULL << 28)>
class LatencyTest : public SMReadTest<elem_num, perform_random_read_int32> {
public: 
    LatencyTest(int local_device, int remote_device, int repeat = 5)
        : SMReadTest<elem_num, perform_random_read_int32>(local_device, remote_device, repeat) {};
    virtual void Statistic() override {
        cout << "Latency Test Result:\n";
        cout << "--------------------\n";
        cout.setf(ios::left, ios::adjustfield);
        size_t fwid1 = string{"MemoryType"}.size();
        for(auto &e : this->env) {
            fwid1 = std::max(fwid1, e->name.size());
        }
        size_t fwid2 = string{"Latency(us)"}.size();
        fwid1 += 4;
        fwid2 += 4;
        cout.width(fwid1); cout << "MemoryType" << "|  ";
        cout.width(fwid2); cout << "Latency(us)" << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto latency = time * 1e3; // us
            latency /= this->repeat;
            cout.width(fwid1);
            cout << e->name << "|  ";
            cout.width(fwid2);
            cout << latency << "\n";
            // cout << "\t" << e->name << "\t|"
            //      << "\t" << latency << "\t\n";
        }
        cout << "\n";
    }
};

template<size_t elem_num = (1ULL << 28)>
class RandomBandwidth : public SMReadTest<elem_num, perform_random_read> {
public:
    RandomBandwidth(int local_device, int remote_device, int repeat = 5)
        : SMReadTest<elem_num, perform_random_read>(local_device, remote_device, repeat) {}
    virtual void Statistic() override {
        cout << "RandomBandwidth Test Result:\n";
        cout << "----------------------------\n";
        cout.setf(ios::left, ios::adjustfield);
        string l = "MemoryType", r = "RandomBandwidth(GB/s)";
        size_t fw1 = l.size();
        size_t fw2 = r.size();
        for(auto &e : this->env) {
            fw1 = std::max(fw1, e->name.size());
        }
        fw1 += 4, fw2 += 4;
        cout.width(fw1); cout << l << "| ";
        cout.widen(fw2); cout << r << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto bandwidth = 1.0 * e->ReadSize() / 1024 / 1024 / 1024 / time * 1000; // gb/s
            bandwidth *= this->repeat;
            cout.width(fw1); cout << e->name << "|  ";
            cout.width(fw2); cout << bandwidth << "\n";
        }
        cout << "\n";
    }
};

int main() {
    BandWitdhTest bandwidth_test(0, 1, 1);
    bandwidth_test.Run();
    bandwidth_test.Statistic();

    LatencyTest latency_test(0, 1, 100);
    latency_test.Run();
    latency_test.Statistic();

    RandomBandwidth random_bandwidth_test(0, 1, 1);
    random_bandwidth_test.Run();
    random_bandwidth_test.Statistic();
}