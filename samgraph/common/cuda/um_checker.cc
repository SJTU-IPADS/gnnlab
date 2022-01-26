#include "um_checker.h"
#include "../logging.h"
#include "../constant.h"
#include "../device.h"
#include "../run_config.h"

namespace samgraph {
namespace common {
namespace cuda {

UMChecker::UMChecker(const Dataset &dataset, TensorPtr order) : 
    _nodeIdnew2old(order) {
  auto sampler_ctx = dataset.indptr->Ctx();
  CHECK(sampler_ctx.device_type == DeviceType::kGPU_UM);
  sampler_ctx.device_type = DeviceType::kGPU;
  _raw_dataset.indptr = Tensor::CopyTo(dataset.indptr, sampler_ctx);
  _raw_dataset.indices = Tensor::CopyTo(dataset.indices, sampler_ctx);

  CHECK(dataset.alias_table == nullptr || dataset.alias_table->Data() == nullptr);
  CHECK(dataset.prob_table == nullptr || dataset.prob_table->Data() == nullptr);
  CHECK(dataset.prob_prefix_table == nullptr || dataset.prob_prefix_table->Data() == nullptr);

  _nodeIdold2new = Tensor::CopyTo(order, order->Ctx());
  auto _nodeIdold2new_ptr = static_cast<IdType*>(_nodeIdold2new->MutableData());
  auto _nodeIdnew2old_ptr = static_cast<const IdType*>(_nodeIdnew2old->Data());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(size_t i = 0; i < _nodeIdold2new->Shape()[0]; i++) {
    _nodeIdold2new_ptr[_nodeIdnew2old_ptr[i]] = i;
  }
}


void UMChecker::Check(IdType* src, IdType* dst, size_t* num_out,
                      IdType* src_chk, IdType* dst_chk, size_t* num_out_chk,
                      Context ctx) const {
  size_t _lenth, lenth;
  Device::Get(ctx)->CopyDataFromTo(num_out, 0, &_lenth, 0, sizeof(size_t), ctx, CPU());
  Device::Get(ctx)->CopyDataFromTo(num_out_chk, 0, &lenth, 0, sizeof(size_t), ctx, CPU());
  CHECK(_lenth == lenth);
  auto _src = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), lenth * sizeof(IdType)));
  auto _dst = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), lenth * sizeof(IdType)));
  auto _src_chk = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), lenth * sizeof(IdType)));
  auto _dst_chk = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), lenth * sizeof(IdType)));
  Device::Get(ctx)->CopyDataFromTo(src, 0, _src, 0, lenth * sizeof(IdType), ctx, CPU());
  Device::Get(ctx)->CopyDataFromTo(dst, 0, _dst, 0, lenth * sizeof(IdType), ctx, CPU());
  Device::Get(ctx)->CopyDataFromTo(src_chk, 0, _src_chk, 0, lenth * sizeof(IdType), ctx, CPU());
  Device::Get(ctx)->CopyDataFromTo(dst_chk, 0, _dst_chk, 0, lenth * sizeof(IdType), ctx, CPU());
  auto _nodeIdnew2old_ptr = static_cast<const IdType*>(_nodeIdnew2old->Data());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(size_t i = 0; i < lenth; i++) {
    if(_src[i] == Constant::kEmptyKey) {
      CHECK(_dst[i] == Constant::kEmptyKey);
      CHECK(_src_chk[i] == Constant::kEmptyKey);
      CHECK(_dst_chk[i] == Constant::kEmptyKey);
    } else {
      CHECK(_src_chk[i] == _nodeIdnew2old_ptr[_src[i]]);
      CHECK(_dst_chk[i] == _nodeIdnew2old_ptr[_dst[i]]);
    }
  }
  for(auto ptr : {_src, _dst, _src_chk, _dst_chk}) {
    Device::Get(CPU())->FreeWorkspace(CPU(), ptr);
  }
}

void UMChecker::CvtInputNodeId(const IdType* input, IdType* input_chk,
                               const size_t num_input,
                               Context ctx) const {
  auto _nodeIdnew2old_ptr = static_cast<const IdType*>(_nodeIdnew2old->Data());
  auto _input_chk = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), num_input * sizeof(IdType)));
  auto _input = static_cast<IdType*>(Device::Get(CPU())->AllocWorkspace(
    CPU(), num_input * sizeof(IdType)));
  Device::Get(ctx)->CopyDataFromTo(
    input, 0, _input, 0, 
    num_input * sizeof(IdType), 
    ctx, CPU());
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for(size_t i = 0; i < num_input; i++) {
    CHECK(_input[i] >= 0 && _input[i] < _nodeIdnew2old->Shape()[0]);
    _input_chk[i] = _nodeIdnew2old_ptr[_input[i]];
  }
  Device::Get(ctx)->CopyDataFromTo(
    _input_chk, 0, input_chk, 0, 
    num_input * sizeof(IdType), 
    CPU(), ctx);
  Device::Get(CPU())->FreeWorkspace(CPU(), _input);
  Device::Get(CPU())->FreeWorkspace(CPU(), _input_chk);
}

const IdType* UMChecker::GetRawIndptr() const {
  return static_cast<const IdType*>(_raw_dataset.indptr->Data());
}

const IdType* UMChecker::GetRawIndices() const {
  return static_cast<const IdType*>(_raw_dataset.indices->Data());
}

} // namespace cuda
} // namespace common
} // namespace samgraph