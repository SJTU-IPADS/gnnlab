#ifndef SAMGRAPH_COO2CSR_H
#define SAMGRAPH_COO2CSR_H

namespace samgraph {
namespace common {
namespace cuda {

void ConvertCoo2Csr(void *csr_);

} // namespace cuda
} // namespace common
} // namespace samgraph

#endif // SAMGRAPH_COO2CSR_H
