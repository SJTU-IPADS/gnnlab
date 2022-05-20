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

template<size_t elem_num, auto perform_read, decltype(perform_read) overhead = nullptr>
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
        vector<cudaEvent_t> start_oh(env.size()), end_oh(env.size());
        for(int i = 0; i < env.size(); i++) {
            // cout << env[i]->name << " " << env[i]->device << "\n";
            CUDA_CALL(cudaSetDevice(env[i]->device));
            CUDA_CALL(cudaEventCreate(&start[i]));
            CUDA_CALL(cudaEventCreate(&end[i]));
            CUDA_CALL(cudaEventCreate(&start_oh[i]));
            CUDA_CALL(cudaEventCreate(&end_oh[i]));
        }
        int block_size = 0, grid_size = 0;
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&block_size, &grid_size, read));

        auto measure_kernel = [&](
            MemoryTestCase* env, cudaEvent_t start, cudaEvent_t end, decltype(perform_read) kernel
        ) -> float {
            *start_flag = 0;
            CUDA_CALL(cudaSetDevice(env->device));
            CUDA_CALL(cudaStreamSynchronize(env->stream));
            delay<<<1, 1, 0, env->stream>>>(start_flag);
            CUDA_CALL(cudaEventRecord(start, env->stream));
            for(int r = 0; r < this->repeat; r++) {
                kernel(grid_size, block_size, env->stream, 
                    env->read_buf.data, env->read_buf.size, 
                    env->read_result.data, env->read_result.size);
            }
            CUDA_CALL(cudaEventRecord(end, env->stream));

            *start_flag = 1;
            CUDA_CALL(cudaStreamSynchronize(env->stream));
            float ms;
            CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
            return ms;
        };

        for(int i = 0; i < env.size(); i++) {
            report_process(i);
            env[i]->Init();
            float kernel_ms = measure_kernel(env[i].get(), start[i], end[i], perform_read);
            if (overhead != nullptr) {
                float overhead_ms = measure_kernel(env[i].get(), start_oh[i], end_oh[i], overhead);
                // cout << "\n" << env[i]->name << " overhead_ms " << overhead_ms << " kernel_ms " << kernel_ms << "\n";
                kernel_ms -= overhead_ms;
            }
            env[i]->Clear(kernel_ms);
        }
        report_process(env.size());
        cudaFree((void*)start_flag);
        for(int i = 0; i < env.size(); i++) {
            CUDA_CALL(cudaSetDevice(env[i]->device));
            CUDA_CALL(cudaEventDestroy(start[i]));
            CUDA_CALL(cudaEventDestroy(end[i]));
            CUDA_CALL(cudaEventDestroy(start_oh[i]));
            CUDA_CALL(cudaEventDestroy(end_oh[i]))
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
        cout << "Sequential Thpt Test Result:\n";
        cout << "----------------------------\n";
        cout.setf(ios::left, ios::adjustfield);
        size_t fwid1 = string{"MemoryType"}.size();
        for(auto &e : this->env) {
            fwid1 = std::max(fwid1, e->name.size());
        }
        size_t fwid2 = string{"Throughput(GB/s)"}.size();
        size_t fwid3 = string{"Time(ms)"}.size();
        fwid1 += 4;
        fwid2 += 4;
        fwid3 += 4;
        cout.width(fwid1); cout << "MemoryType" << "|  ";
        cout.width(fwid2); cout << "Throughput(GB/s)" << "| ";
        cout.width(fwid3); cout << "Time(ms)" << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto bandwidth = 1.0 * e->ReadSize() / 1024 / 1024 / 1024 / time * 1000; // gb/s
            bandwidth *= this->repeat;
            cout.width(fwid1); cout << e->name << "|  ";
            cout.width(fwid2); cout << bandwidth << "| ";
            cout.widen(fwid3); cout << time / this->repeat << "\n";
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
        size_t fwid3 = string{"Time(ms)"}.size();
        fwid1 += 4;
        fwid2 += 4;
        fwid3 += 4;
        cout.width(fwid1); cout << "MemoryType" << "|  ";
        cout.width(fwid2); cout << "Latency(us)" << "| ";
        cout.width(fwid3); cout << "Time(ms)" << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto latency = time * 1e3; // us
            latency /= this->repeat;
            cout.width(fwid1);
            cout << e->name << "|  ";
            cout.width(fwid2);
            cout << latency << "| ";
            cout.width(fwid3);
            cout << time / this->repeat << "\n";
            // cout << "\t" << e->name << "\t|"
            //      << "\t" << latency << "\t\n";
        }
        cout << "\n";
    }
};

template<size_t elem_num = (1ULL << 28)>
class RandomBandwidth : public SMReadTest<elem_num, perform_random_read, perform_random_read_overhead> {
public:
    RandomBandwidth(int local_device, int remote_device, int repeat = 5)
        : SMReadTest<elem_num, perform_random_read, perform_random_read_overhead>(local_device, remote_device, repeat) {}
    virtual void Statistic() override {
        cout << "Random Thpt Test Result:\n";
        cout << "----------------------------\n";
        cout.setf(ios::left, ios::adjustfield);
        string l = "MemoryType", r = "Throughput(GB/s)", t = "Time(ms)";
        size_t fw1 = l.size();
        size_t fw2 = r.size();
        size_t fw3 = t.size();
        for(auto &e : this->env) {
            fw1 = std::max(fw1, e->name.size());
        }
        fw1 += 4, fw2 += 4, fw3 += 4;
        cout.width(fw1); cout << l << "| ";
        cout.width(fw2); cout << r << "| ";
        cout.width(fw3); cout << t << "\n";
        for(auto &e : this->env) {
            auto time = e->Time();
            auto bandwidth = 1.0 * e->ReadSize() / 1024 / 1024 / 1024 / time * 1000; // gb/s
            bandwidth *= this->repeat;
            cout.width(fw1); cout << e->name << "|  ";
            cout.width(fw2); cout << bandwidth << "| ";
            cout.width(fw3); cout << time / this->repeat << "\n";
        }
        cout << "\n";
    }
};

template<auto perform_kerenl, size_t elem_num = (1ULL << 28), decltype(perform_kerenl) overhead = nullptr> 
class KernelTimeTest : public SMReadTest<elem_num, perform_kerenl, overhead> {
public:
    KernelTimeTest(string test_name, int local_device, int remote_device, int repeat = 5) 
        : test_name(test_name), SMReadTest<elem_num, perform_kerenl, overhead>(local_device, remote_device, repeat) {}
    virtual void Statistic() override {
        std::string title = test_name + " Time Test:";
        cout << title << "\n";
        cout << string(title.size(), '-') << "\n";
        cout.setf(ios::left, ios::adjustfield);
        string l = "MemoryType", r = "Time(ms)";
        size_t fw1 = l.size();
        size_t fw2 = r.size();
        for (auto &e : this->env) {
            fw1 = std::max(fw1, e->name.size());
        }
        fw1 += 4, fw2 += 4;
        cout.width(fw1); cout << l << "| ";
        cout.width(fw2); cout << r << "\n";
        for (auto &e: this->env) {
            auto time = e->Time();
            time /= this->repeat;
            cout.width(fw1); cout << e->name << "| ";
            cout.width(fw2); cout << time << "\n";
        }
        cout << "\n";
    }
    string test_name;
};

// using RandomWithLocaloverheadTime = \
//     KernelTimeTest<perform_random_read_with_local_overhead>;
using RandomCoalescedTime = \
    KernelTimeTest<perform_random_read_coalesced>;

template<size_t lkbehind> using RoffSlookbehindTime = \
    KernelTimeTest<perform_random_off_sequentail_lookbehind<lkbehind>>;
template<size_t page_size, size_t lkbehind> using RoffRlookbehindTime = \
    KernelTimeTest<perform_random_off_random_lookbehind<page_size, lkbehind>>;
template<size_t page_size, size_t lkbehind> using RoffDlookbehindTime = \
    KernelTimeTest<perform_random_off_divergence_lookbehind<page_size, lkbehind>>;

template<size_t lkbehind> using RoffSsmlookbehindTime = \
    KernelTimeTest<perform_random_off_sequentail_same_lookbehind<lkbehind>>;
template<size_t page_size, size_t lkbehind> using RoffRsmlookbehindTime = \
    KernelTimeTest<perform_random_off_random_same_lookbehind<page_size, lkbehind>>;
template<size_t page_size, size_t lkbehind> using RoffDsmlookbehindTime = \
    KernelTimeTest<perform_random_off_divergence_same_lookbehind<page_size, lkbehind>>;


std::string sz2str(size_t sz) {
    stringstream ss;
    if (sz < 1024ULL) {
        return std::to_string(sz);
    } else if (sz < 1024ULL * 1024ULL) {
        ss << std::fixed << std::setprecision(1) << 1.0 * sz / 1024 << "K";
        return ss.str();
    } else if (sz < 1024ULL * 1024ULL * 1024ULL) {
        ss << std::fixed << std::setprecision(1) << 1.0 * sz / 1024 / 1024 << "M";
        return ss.str();
    } else {
        ss << std::fixed << std::setprecision(1) << 1.0 * sz / 1024 / 1024 / 1024 << "G";
        return ss.str();
    }
}

template<size_t N>
struct iter { static constexpr auto value = N; };

template<typename Fn, size_t... Is>
void static_for(Fn fn, std::index_sequence<Is...>) {
    (fn(iter<Is>{}), ...);
}

template<size_t N, typename Fn>
void static_for(Fn fn) {
    static_for(fn, std::make_index_sequence<N>());
}

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

    // RandomWithLocaloverheadTime random_with_local_overhead_test("random_with_local_overhead", 0, 1, 1);
    // random_with_local_overhead_test.Run();
    // random_with_local_overhead_test.Statistic();

    RandomCoalescedTime random_coalesced_test("random_coalesced", 0, 1, 1);
    random_coalesced_test.Run();
    random_coalesced_test.Statistic();

    {
        constexpr std::array<size_t, 3> vec = {{
            1, 8, 32
        }};
        static_for<vec.size()>([&](auto i) {
            constexpr size_t lkbehind = vec[i.value];
            RoffSlookbehindTime<lkbehind> random_off_sequential_lookbehind_test(
                "random_off_sequential_lookbehind<"+to_string(lkbehind)+">", 0, 1, 1);
            random_off_sequential_lookbehind_test.Run();
            random_off_sequential_lookbehind_test.Statistic();
        });
        static_for<vec.size()>([&](auto i) {
            constexpr size_t lkbehind = vec[i.value];
            RoffSsmlookbehindTime<lkbehind> random_off_sequentail_same_lookbehind_test(
                "random_off_sequentail_same_lookbehind<"+to_string(lkbehind)+">", 0, 1, 1);
            random_off_sequentail_same_lookbehind_test.Run();
            random_off_sequentail_same_lookbehind_test.Statistic();
        });
    }
    {
        constexpr std::array<std::pair<size_t, size_t>, 3> vec = {{
            {128, 32}, {1024, 32}, {1024 * 1024 * 2, 32}
            // {128, 16}
        }};
        static_for<vec.size()>([&](auto i) {
            // cout << vec[i.value].first << " " << vec[i.value].second << "\n";
            // constexpr auto [page_sz, lkbehind] = vec[i.value];
            constexpr size_t page_sz = vec[i.value].first;
            constexpr size_t lkbehind = vec[i.value].second;
            RoffRlookbehindTime<page_sz, lkbehind> random_off_random_lookbehind_test(
                "random_off_random_lookbehind<"+sz2str(page_sz)+", "+to_string(lkbehind)+">", 0, 1, 1);
            random_off_random_lookbehind_test.Run();
            random_off_random_lookbehind_test.Statistic();
        });
        static_for<vec.size()>([&](auto i) {
            constexpr size_t page_sz = vec[i.value].first;
            constexpr size_t lkbehind = vec[i.value].second;
            RoffRsmlookbehindTime<page_sz, lkbehind> random_off_random_same_lookbehind_test(
                "random_off_random_same_lookbehind<"+sz2str(page_sz)+", "+to_string(lkbehind)+">", 0, 1, 1);
            random_off_random_same_lookbehind_test.Run();
            random_off_random_same_lookbehind_test.Statistic();
        });
    }
    {
        constexpr std::array<std::pair<size_t, size_t>, 3> vec = {{
            {128, 32}, {1024, 32}, {1024 * 1024 * 2, 32}
        }};
        static_for<vec.size()>([&](auto i) {
            constexpr size_t page_sz = vec[i.value].first;
            constexpr size_t lkbehind = vec[i.value].second;
            RoffDlookbehindTime<page_sz, lkbehind> random_off_divergence_lookbehind_test(
                "random_off_divergence_lookbehind<"+sz2str(page_sz)+", "+to_string(lkbehind)+">", 0, 1, 1);
            random_off_divergence_lookbehind_test.Run();
            random_off_divergence_lookbehind_test.Statistic();
        });
        static_for<vec.size()>([&](auto i) {
            constexpr size_t page_sz = vec[i.value].first;
            constexpr size_t lkbehind = vec[i.value].second;
            RoffDsmlookbehindTime<page_sz, lkbehind> random_off_divergence_same_lookbehind_test(
                "random_off_divergence_same_lookbehind<"+sz2str(page_sz)+", "+to_string(lkbehind)+">", 0, 1, 1);
            random_off_divergence_same_lookbehind_test.Run();
            random_off_divergence_same_lookbehind_test.Statistic();
        });
    }
}
