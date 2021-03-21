#include "../logging.h"
#include "../config.h"
#include "cpu_engine.h"
#include "cpu_loops.h"

namespace samgraph {
namespace common {
namespace cpu {

SamGraphCpuEngine::SamGraphCpuEngine() {
    _initialize = false;
    _should_shutdown = false;
}

void SamGraphCpuEngine::Init(std::string dataset_path, int sample_device, int train_device,
                             size_t batch_size, std::vector<int> fanout, int num_epoch) {
    if (_initialize) {
        return;
    }

    SAM_CHECK_EQ(sample_device, CPU_DEVICE_ID);
    SAM_CHECK_GT(train_device, CPU_DEVICE_ID);

    _engine_type = kCpuEngine;
    _sample_device = sample_device;
    _train_device = train_device;
    _dataset_path = dataset_path;
    _batch_size = batch_size;
    _fanout = fanout;
    _num_epoch = num_epoch;
    _joined_thread_cnt = 0;

    // Load the target graph data
    LoadGraphDataset();

    // Create CUDA streams
    CUDA_CALL(cudaSetDevice(_train_device));
    _work_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    CUDA_CALL(cudaStreamCreateWithFlags(_work_stream, cudaStreamNonBlocking));

    CUDA_CALL(cudaStreamSynchronize(*_work_stream));

    _extractor = new Extractor();
    _permutation = new RandomPermutation(_dataset->train_set, _num_epoch, _batch_size, false);
    _num_step = _permutation->num_step();
    _graph_pool = new GraphPool(Config::kGraphPoolThreshold);

    _initialize = true;
}

void SamGraphCpuEngine::Start() {
    std::vector<LoopFunction> func;
    
    // func.push_back(CpuSampleLoop);

    // Start background threads
    for (size_t i = 0; i < func.size(); i++) {
        _threads.push_back(new std::thread(func[i]));
    }
    SAM_LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void SamGraphCpuEngine::Shutdown() {
    if (_should_shutdown) {
        return;
    }

    _should_shutdown = true;
    int total_thread_num = _threads.size();

    while (!IsAllThreadFinish(total_thread_num)) {
        // wait until all threads joined
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }

    for (size_t i = 0; i < _threads.size(); i++) {
        _threads[i]->join();
        delete _threads[i];
        _threads[i] = nullptr;
    }

    if (_extractor) {
        delete _extractor;
        _extractor = nullptr;
    }

    delete _dataset;

    if (_work_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_work_stream));
        CUDA_CALL(cudaStreamDestroy(*_work_stream));
        free(_work_stream);
        _work_stream = nullptr;
    }

    if (_permutation) {
        delete _permutation;
        _permutation = nullptr;
    }

    if (_graph_pool) {
        delete _graph_pool;
        _graph_pool = nullptr;
    }

    _threads.clear();
    _joined_thread_cnt = 0;
    _initialize = false;
    _should_shutdown = false;
}

} // namespace cpu
} // namespace common
} // namespace samgraph
