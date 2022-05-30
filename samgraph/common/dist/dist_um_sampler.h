#ifndef SAMGRAPH_DIST_UM_SAMPLER_H
#define SAMGRAPH_DIST_UM_SAMPLER_H

#include "../common.h"
#include "../logging.h"
#include "../engine.h"
#include "../cuda/cuda_engine.h"
#include "../cuda/cuda_hashtable.h"

// tiny engine for um sampler, maintain sampler ctx, etc

namespace samgraph {
namespace common {
namespace dist {

class DistUMSampler {
public:
    DistUMSampler(IdType sampler_id);
    ~DistUMSampler();
private:
    IdType _sampler_id;

    Context _sampler_ctx;
    StreamHandle _sample_stream;

    Shuffler* _shuffler;
    cuda::OrderedHashTable* _hashtable;
};
    
} // namespace dist
} // namespace common
} // namespace samgraph



#endif