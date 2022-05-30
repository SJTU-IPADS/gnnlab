#include "dist_um_sampler.h"
#include "../run_config.h"
#include "../device.h"

namespace samgraph {
namespace common {
namespace dist {

DistUMSampler::DistUMSampler(IdType sampler_id) 
    : _sampler_id(sampler_id), _sampler_ctx(RunConfig::unified_memory_ctxes[sampler_id]) 
{
        
} 

} // namespace dist
} // namespace common
} // namespace samgraph
