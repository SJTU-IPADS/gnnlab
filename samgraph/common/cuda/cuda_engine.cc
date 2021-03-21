#include <cstdlib>

#include "../common.h"
#include "../config.h"
#include "../logging.h"
#include "cuda_loops.h"
#include "cuda_engine.h"

namespace samgraph{
namespace common {
namespace cuda {

SamGraphCudaEngine::SamGraphCudaEngine() {
    _initialize = false;
    _should_shutdown = false;
}

void SamGraphCudaEngine::Init(std::string dataset_path, int sample_device, int train_device,
                          size_t batch_size, std::vector<int> fanout, int num_epoch) {
    if (_initialize) {
        return;
    }

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
    _sample_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _id_copy_host2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _graph_copy_device2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _id_copy_device2host_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _feat_copy_host2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));

    CUDA_CALL(cudaSetDevice(_sample_device));
    CUDA_CALL(cudaStreamCreateWithFlags(_sample_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_id_copy_host2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_graph_copy_device2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_id_copy_device2host_stream, cudaStreamNonBlocking));

    CUDA_CALL(cudaSetDevice(_train_device));
    CUDA_CALL(cudaStreamCreateWithFlags(_feat_copy_host2device_stream, cudaStreamNonBlocking));

    CUDA_CALL(cudaStreamSynchronize(*_sample_stream));
    CUDA_CALL(cudaStreamSynchronize(*_id_copy_host2device_stream));
    CUDA_CALL(cudaStreamSynchronize(*_graph_copy_device2device_stream));
    CUDA_CALL(cudaStreamSynchronize(*_id_copy_device2host_stream));

    CUDA_CALL(cudaStreamSynchronize(*_feat_copy_host2device_stream));

    _submit_table = new ReadyTable(Config::kSubmitTableCount, "CUDA_SUBMIT");
    _cpu_extractor = new CpuExtractor();
    _permutation = new RandomPermutation(_dataset->train_set, _num_epoch, _batch_size, false);
    _num_step = _permutation->num_step();
    _graph_pool = new GraphPool(Config::kGraphPoolThreshold);

    // Create queues
    for (int i = 0; i < CudaQueueNum; i++) {
        SAM_LOG(DEBUG) << "Create task queue" << i;
        auto type = static_cast<CudaQueueType>(i);
        if (type == CUDA_SUBMIT) {
            _queues.push_back(new SamGraphTaskQueue(type, Config::kQueueThreshold.at(type), _submit_table));
        } else {
            _queues.push_back(new SamGraphTaskQueue(type, Config::kQueueThreshold.at(type), nullptr));
        }
    }

    _initialize = true;
}

void SamGraphCudaEngine::Start() {
    std::vector<LoopFunction> func;
    
    func.push_back(HostPermutateLoop);
    func.push_back(IdCopyHost2DeviceLoop);
    func.push_back(DeviceSampleLoop);
    func.push_back(GraphCopyDevice2DeviceLoop);
    func.push_back(IdCopyDevice2HostLoop);
    func.push_back(HostFeatureExtractLoop);
    func.push_back(FeatureCopyHost2DeviceLoop);
    func.push_back(SubmitLoop);

    // func.push_back(SingleLoop);

    // Start background threads
    for (size_t i = 0; i < func.size(); i++) {
        _threads.push_back(new std::thread(func[i]));
    }
    SAM_LOG(DEBUG) << "Started " << func.size() << " background threads.";
}

void SamGraphCudaEngine::Shutdown() {
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

    if (_submit_table) {
        delete _submit_table;
        _submit_table = nullptr;
    }

    if (_cpu_extractor) {
        delete _cpu_extractor;
        _cpu_extractor = nullptr;
    }

    delete _dataset;

    // free queue
    for (size_t i = 0; i < CudaQueueNum; i++) {
        if (_queues[i]) {
            delete _queues[i];
            _queues[i] = nullptr;
        }
    }

    if (_sample_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_sample_stream));
        CUDA_CALL(cudaStreamDestroy(*_sample_stream));
        free(_sample_stream);
        _sample_stream = nullptr;
    }

    if (_id_copy_host2device_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_id_copy_host2device_stream));
        CUDA_CALL(cudaStreamDestroy(*_id_copy_host2device_stream));
        free(_id_copy_host2device_stream);
        _id_copy_host2device_stream = nullptr;
    }

    if (_graph_copy_device2device_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_graph_copy_device2device_stream));
        CUDA_CALL(cudaStreamDestroy(*_graph_copy_device2device_stream));
        free(_graph_copy_device2device_stream);
        _graph_copy_device2device_stream = nullptr;
    }

    if (_id_copy_device2host_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_id_copy_device2host_stream));
        CUDA_CALL(cudaStreamDestroy(*_id_copy_device2host_stream));
        free(_id_copy_device2host_stream);
        _id_copy_device2host_stream = nullptr;
    }

    if (_feat_copy_host2device_stream) {
        CUDA_CALL(cudaStreamSynchronize(*_feat_copy_host2device_stream));
        CUDA_CALL(cudaStreamDestroy(*_feat_copy_host2device_stream));
        free(_feat_copy_host2device_stream);
        _feat_copy_host2device_stream = nullptr;
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

bool SamGraphCudaEngine::IsAllThreadFinish(int total_thread_num) {
  int k = _joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

} // namespace cuda
} // namespace common
} // namespace samgraph