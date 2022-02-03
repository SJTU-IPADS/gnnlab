#ifndef SAMGRAPH_UM_CHECKER
#define SAMGRAPH_UM_CHECKER

#include "../common.h"
#include "cuda_random_states.h"

namespace samgraph {
namespace common {
namespace cuda {

class SamplingChecker {
public:
  SamplingChecker(const Dataset &dataset, Context sampler_ctx);
  virtual void Check(IdType* src, IdType* dst, size_t* num_out, 
                    IdType* src_chk,IdType* dst_chk, size_t* num_out_chk, 
                    Context ctx) const;
  const IdType* GetRawIndptr() const;
  const IdType* GetRawIndices() const;
  virtual ~SamplingChecker(){};
protected:
  Dataset _raw_dataset;
};

class UMChecker : public SamplingChecker {
public:
  UMChecker(const Dataset &dataset, TensorPtr order, Context sampler_ctx);
  void CvtInputNodeId(const IdType *input, IdType* input_chk, const size_t num_input, Context ctx) const;
  void Check(IdType* src, IdType* dst, size_t* num_out, 
             IdType* src_chk,IdType* dst_chk, size_t* num_out_chk, 
             Context ctx) const override;
private:
  TensorPtr _nodeIdnew2old;
  TensorPtr _nodeIdold2new;
};

} // namespace cuda
} // namespace common
} // namespace samgraph


#endif