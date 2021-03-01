#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "types.h"
#include "config.h"
#include "logging.h"
#include "engine.h"

namespace samgraph{
namespace common {

namespace {

std::tuple<void*, size_t> MmapData(std::string filepath) {
    // Fetch the size of the file
    struct stat st;
    stat(filepath.c_str(), &st);
    size_t size = st.st_size;

    // Map the content of the file into memory
    int fd = open(filepath.c_str(), O_RDONLY, 0);
    void *data = mmap(NULL, size, PROT_READ, MAP_SHARED|MAP_FILE, fd, 0);
    mlock(data, size);
    close(fd);

    return {data, size};
}

void FreeOrMunmapData(void *data, size_t size, int dev, bool inheap) {
    if (!data) {
        return;
    }

    if (dev == CPU_DEVICE_ID && inheap) {
        free(data);
    } else if (dev == CPU_DEVICE_ID && !inheap) {
        munmap(data, size);
    } else {
        SAM_CHECK_GT(dev, 0);
        CUDA_CALL(cudaFree(data));
    }
}

inline void *GetDataPtr(std::tuple<void*, size_t> &data) {
    return std::get<0>(data);
}

inline size_t GetDataSize(std::tuple<void*, size_t> &data) {
    return std::get<1>(data);
}

}

bool SamGraphEngine::_initialize = false;
bool SamGraphEngine::_should_shutdown = false;

int SamGraphEngine::_sample_device = 0;
int SamGraphEngine::_train_device = 0;
std::string SamGraphEngine::_dataset_path = "";
SamGraphDataset* SamGraphEngine::_dataset = nullptr;
int SamGraphEngine::_batch_size = 0;
std::vector<int> SamGraphEngine::_fanout;
int SamGraphEngine::_num_epoch = 0;

cudaStream_t* SamGraphEngine::_sample_stream = nullptr;
cudaStream_t* SamGraphEngine::_id_copy_host2device_stream = nullptr;
cudaStream_t* SamGraphEngine::_graph_copy_device2device_stream = nullptr;
cudaStream_t* SamGraphEngine::_id_copy_device2host_stream = nullptr;
cudaStream_t* SamGraphEngine::_feat_copy_host2device_stream = nullptr;

void SamGraphEngine::Init(std::string dataset_path, int sample_device, int train_device,
                          int batch_size, std::vector<int> fanout, int num_epoch) {
    if (_initialize) {
        return;
    }

    _sample_device = sample_device;
    _train_device = train_device;
    _dataset_path = dataset_path;
    _batch_size = batch_size;
    _fanout = fanout;
    _num_epoch = num_epoch;

    // Load the target graph data
    LoadGraphDataset();

    // Create CUDA streams
    _sample_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _id_copy_host2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _graph_copy_device2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _id_copy_device2host_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    _feat_copy_host2device_stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));

    CUDA_CALL(cudaStreamCreateWithFlags(_sample_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_id_copy_host2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_graph_copy_device2device_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_id_copy_device2host_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_id_copy_device2host_stream, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(_feat_copy_host2device_stream, cudaStreamNonBlocking));

    CUDA_CALL(cudaStreamSynchronize(*_sample_stream));
    CUDA_CALL(cudaStreamSynchronize(*_id_copy_host2device_stream));
    CUDA_CALL(cudaStreamSynchronize(*_graph_copy_device2device_stream));
    CUDA_CALL(cudaStreamSynchronize(*_id_copy_device2host_stream));
    CUDA_CALL(cudaStreamSynchronize(*_feat_copy_host2device_stream));

    _initialize = true;
}

void SamGraphEngine::Shutdown() {
    _should_shutdown = true;

    RemoveGraphDataset();

    if (_sample_stream) {
        CUDA_CALL(cudaStreamDestroy(*_sample_stream));
        free(_sample_stream);
        _sample_stream = nullptr;
    }

    if (_id_copy_host2device_stream) {
        CUDA_CALL(cudaStreamDestroy(*_id_copy_host2device_stream));
        free(_id_copy_host2device_stream);
        _id_copy_host2device_stream = nullptr;
    }

    if (_graph_copy_device2device_stream) {
        CUDA_CALL(cudaStreamDestroy(*_graph_copy_device2device_stream));
        free(_graph_copy_device2device_stream);
        _graph_copy_device2device_stream = nullptr;
    }

    if (_id_copy_device2host_stream) {
        CUDA_CALL(cudaStreamDestroy(*_id_copy_device2host_stream));
        free(_id_copy_device2host_stream);
        _id_copy_device2host_stream = nullptr;
    }

    if (_feat_copy_host2device_stream) {
        CUDA_CALL(cudaStreamDestroy(*_feat_copy_host2device_stream));
        free(_feat_copy_host2device_stream);
        _feat_copy_host2device_stream = nullptr;
    }
}

void SamGraphEngine::LoadGraphDataset() {
    _dataset = new SamGraphDataset();
    std::unordered_map<std::string, size_t> meta;

    if (_dataset_path.back() != '/') {
        _dataset_path.push_back('/');
    }

    // Parse the meta data
    std::ifstream meta_file(_dataset_path + SAMGRAPH_META_FILE);
    std::string line;
    while(std::getline(meta_file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> kv {std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

        if (kv.size() < 2) {
            break;
        }

        meta[kv[0]] = std::stoull(kv[1]);
    }

    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_NODE) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_EDGE) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_FEAT_DIM) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_CLASS) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_TRAIN_SET) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_TEST_SET) > 0);
    SAM_CHECK(meta.count(SAMGRAPH_META_NUM_VALID_SET) > 0);

    _dataset->numNode = meta[SAMGRAPH_META_NUM_NODE];
    _dataset->numEdge = meta[SAMGRAPH_META_NUM_EDGE];
    _dataset->numClass = meta[SAMGRAPH_META_NUM_CLASS];

    // Mmap the data
    auto feat_data = MmapData(_dataset_path + SAMGRAPH_FEAT_FILE);
    auto label_data = MmapData(_dataset_path + SAMGRAPH_LABEL_FILE);
    auto indptr_data = MmapData(_dataset_path + SAMGRAPH_INDPTR_FILE);
    auto indice_data = MmapData(_dataset_path + SAMGRAPH_INDICE_FILE);
    auto train_set_data = MmapData(_dataset_path + SAMGRAPH_TRAIN_SET_FILE);
    auto test_set_data = MmapData(_dataset_path + SAMGRAPH_TEST_SET_FILE);
    auto valid_set_data = MmapData(_dataset_path + SAMGRAPH_VALID_SET_FILE);

    int onedim = 1; 

    SAM_CHECK_EQ(GetDataSize(feat_data), _dataset->numNode * meta[SAMGRAPH_META_FEAT_DIM] * sizeof(feat_t));
    SAM_CHECK_EQ(GetDataSize(label_data), _dataset->numNode * onedim * sizeof(label_t));
    SAM_CHECK_EQ(GetDataSize(indptr_data), (_dataset->numNode + 1) * sizeof(id_t));
    SAM_CHECK_EQ(GetDataSize(indice_data), (_dataset->numEdge) * sizeof(id_t));
    SAM_CHECK_EQ(GetDataSize(train_set_data), meta[SAMGRAPH_META_NUM_TRAIN_SET] * sizeof(id_t));
    SAM_CHECK_EQ(GetDataSize(test_set_data), meta[SAMGRAPH_META_NUM_TEST_SET] * sizeof(id_t));
    SAM_CHECK_EQ(GetDataSize(valid_set_data), meta[SAMGRAPH_META_NUM_VALID_SET] * sizeof(id_t));

    // Copy graph topology data into the target device and unmap the addr
    CUDA_CALL(cudaSetDevice(_sample_device));
    void *dev_base_addr;
    CUDA_CALL(cudaMalloc(&dev_base_addr, (_dataset->numNode + 1 + _dataset->numEdge) * sizeof(id_t)));
    CUDA_CALL(cudaMemcpy(dev_base_addr, GetDataPtr(indptr_data),
                         GetDataSize(indptr_data), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy((char *)dev_base_addr + GetDataSize(indptr_data),
                         GetDataPtr(indice_data), GetDataSize(indice_data), cudaMemcpyHostToDevice));

    FreeOrMunmapData(GetDataPtr(indptr_data), GetDataSize(indptr_data), CPU_DEVICE_ID, false);
    FreeOrMunmapData(GetDataPtr(indice_data), GetDataSize(indice_data), CPU_DEVICE_ID, false);


    // These data is in device
    _dataset->indptr.SetData(dev_base_addr,
                             {_dataset->numNode + 1},
                             DataType::SAM_INT32,
                             _sample_device,
                             GetDataSize(indptr_data));
    _dataset->indice.SetData((char *)dev_base_addr + (_dataset->numNode + 1),
                             {_dataset->numEdge},
                             DataType::SAM_INT32,
                             _sample_device,
                             GetDataSize(indice_data));

    // These data is in cpu but not in heap
    _dataset->feat.SetData(GetDataPtr(feat_data),
                           {_dataset->numNode, meta[SAMGRAPH_META_FEAT_DIM]},
                           DataType::SAM_FLOAT32,
                           CPU_DEVICE_ID,
                           GetDataSize(feat_data),
                           false);
    _dataset->label.SetData(GetDataPtr(label_data),
                            {_dataset->numNode},
                            DataType::SAM_INT32,
                            CPU_DEVICE_ID,
                            GetDataSize(label_data),
                            false);
    _dataset->trainSet.SetData(GetDataPtr(train_set_data),
                               {meta[SAMGRAPH_META_NUM_TRAIN_SET]},
                               DataType::SAM_INT32,
                               CPU_DEVICE_ID,
                               GetDataSize(train_set_data),
                               false);
    _dataset->testSet.SetData(GetDataPtr(test_set_data),
                              {meta[SAMGRAPH_META_NUM_TEST_SET]},
                              DataType::SAM_INT32,
                              CPU_DEVICE_ID,
                              GetDataSize(test_set_data),
                              false);
    _dataset->validSet.SetData(GetDataPtr(valid_set_data),
                               {meta[SAMGRAPH_META_NUM_VALID_SET]},
                               DataType::SAM_INT32,
                               CPU_DEVICE_ID,
                               GetDataSize(test_set_data),
                               false);
}

void SamGraphEngine::RemoveGraphDataset() {
    if (!_dataset) {
        return;
    }

    SamTensor &feat = _dataset->feat;
    FreeOrMunmapData(feat.data, feat.bytesize, feat.device, feat.inheap);
    feat.ResetData();


    SamTensor &label = _dataset->label;
    FreeOrMunmapData(label.data, label.bytesize, label.device, label.inheap);
    label.ResetData();

    SamTensor &indptr = _dataset->indptr;
    FreeOrMunmapData(indptr.data, indptr.bytesize, indptr.device, indptr.inheap);
    indptr.ResetData();

    SamTensor &indice = _dataset->indice;
    FreeOrMunmapData(indice.data, indice.bytesize, indice.device, indice.inheap);
    indice.ResetData();

    SamTensor &trainSet = _dataset->trainSet;
    FreeOrMunmapData(trainSet.data, trainSet.bytesize, trainSet.device, trainSet.inheap);
    trainSet.ResetData();

    SamTensor &testSet = _dataset->testSet;
    FreeOrMunmapData(testSet.data, testSet.bytesize, testSet.device, testSet.inheap);
    testSet.ResetData();
    
    SamTensor &validSet = _dataset->validSet;
    FreeOrMunmapData(validSet.data, validSet.bytesize, validSet.device, validSet.inheap);
    validSet.ResetData();

    delete _dataset;
    _dataset = nullptr;
}

} // namespace common
} // namespace samgraph