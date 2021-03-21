#ifndef SAMGRAPH_CPU_ENGINE_H
#define SAMGRAPH_CPU_ENGINE_H

#include <thread>

#include <cuda_runtime.h>

#include "../engine.h"
#include "../random_permutation.h"
#include "../extractor.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cpu {

class SamGraphCpuEngine : public SamGraphEngine {
 public:
  SamGraphCpuEngine();

  void Init(std::string dataset_path, int sample_device, int train_device,
            size_t batch_size, std::vector<int> fanout, int num_epoch) override;
  void Start() override;
  void Shutdown() override;

  RandomPermutation* GetRandomPermutation() { return _permutation; }
  Extractor* GetExtractor() { return _extractor; }
  cudaStream_t* GetWorkStream() {  return _work_stream; }

  static inline SamGraphCpuEngine *GetEngine() {
    SAM_CHECK_EQ(_engine_type, kCpuEngine);
    return dynamic_cast<SamGraphCpuEngine *>(SamGraphEngine::_engine);;
  }

 private:
  // Task queue
  std::vector<std::thread*> _threads;

  cudaStream_t* _work_stream;
  
  // Random node batch genrator
  RandomPermutation* _permutation;
  // CPU Extractor
  Extractor* _extractor;
};

} // namespace cpu
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_CPU_ENGINE_H