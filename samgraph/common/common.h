#ifndef SAMGRAPH_COMMON_H
#define SAMGRAPH_COMMON_H

#include <string>

namespace samgraph {
namespace common {

// Keep the order consistent with DMLC/mshadow
// https://github.com/dmlc/mshadow/blob/master/mshadow/base.h
enum DataType {
  SAM_FLOAT32 = 0,
  SAM_FLOAT64 = 1,
  SAM_FLOAT16 = 2,
  SAM_UINT8 = 3,
  SAM_INT32 = 4,
  SAM_INT8 = 5,
  SAM_INT64 = 6,
};

#define CPU_DEVICE_ID (-1)

struct SamTensor {
    void* data;
    std::vector<size_t> size;
    DataType dtype;
    int device;

    size_t bytesize;
    bool inheap; // whether the data is allocated in heap or mmapped

    SamTensor() : data(nullptr), dtype(0), device(0), bytesize(0), inheap(true) {}
    void SetData(void *data, std::vector<size_t> size, DataType dtype,
                 int device, size_t bytesize, bool inheap=true) {
        this->data = data;
        this->size = size;
        this->dtype = dtype;
        this->device = device;

        this->bytesize = bytesize;
        this->inheap = inheap;
    }
    void ResetData() {
        data = nullptr;
        size.reset();
        dtype = 0;
        device = 0;
        bytesize = 0;
        inheap = true;
    }
};

enum QueueType {
    SHUFFLE = 0,
    ID_COPYD2H,
    DEV_SAMPLE,
    GRAPH_COPYD2D,
    ID_COPYD2H,
    FEAT_INDEX_SELECT,
    FEAT_COPYD2H,
    QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST
}

const int QueueNum =
    (int)QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST;

// Graph dataset that should be loaded from the .
struct SamGraphDataset {
    // Graph topology data
    SamTensor indptr;
    SamTensor indice;
    size_t numNode;
    size_t numEdge;
    
    // Node feature and label
    size_t numClass;
    SamTensor feat;
    SamTensor label;
    
    // Node group for machine learning
    SamTensor trainSet;
    SamTensor testSet;
    SamTensor validSet;
};

struct TaskEntry {

};

} // namespace common
} // namespace samgraph

#endif
