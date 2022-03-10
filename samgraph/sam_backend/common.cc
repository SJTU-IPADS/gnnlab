#include "common.h"
#include <iostream>
#include <iomanip>

namespace samgraph {
namespace sam_backend {

namespace {

constexpr size_t row_limit = 3;
constexpr size_t col_limit = 3;
std::unordered_map<common::DeviceType, std::string> dev_name = {
    {common::DeviceType::kCPU, "cpu"},
    {common::DeviceType::kGPU, "gpu"},
    {common::DeviceType::kMMAP, "mmap"}};
template <typename T>
void print_1d(const T *ptr, size_t col) {
  std::cout << "  (" << col << ")[";
  for (size_t j = 0; j < col; j++) {
    if (col > col_limit * 2 && j == col_limit) {
      std::cout << " ...,";
      j = col - col_limit - 1;
      continue;
    }
    std::cout << ptr[j] << ",";
  }
  std::cout << "]\n";
}
template <typename T>
void print_2d(const T *ptr, size_t row, size_t col) {
  std::cout << "  [ (" << row << "," << col << ")\n";
  for (size_t i = 0; i < row; i++) {
    if (row > row_limit * 2 && i == row_limit) {
      std::cout << "    ...,\n";
      i = row - row_limit - 1;
      continue;
    }
    std::cout << "    [";
    for (size_t j = 0; j < col; j++) {
      if (col > col_limit * 2 && j == col_limit) {
        std::cout << "..., ";
        j = col - col_limit - 1;
        continue;
      }
      std::cout << ptr[i * col + j] << ", ";
    }
    std::cout << "],\n";
  }
  std::cout << "  ]\n";
}
template <typename T>
void tensor_printer(TensorPtr ptr) {
  TensorPtr cpup = Tensor::CopyTo(ptr, common::CPU());
  if (cpup->Shape().size() == 1) {
    print_1d(tensor_cast_const<T>(cpup), cpup->Shape()[0]);
  } else if (cpup->Shape().size() == 2) {
    print_2d(tensor_cast_const<T>(cpup), cpup->Shape()[0], cpup->Shape()[1]);
  }
}
} // namespace

void PrintTensor(TensorPtr ptr, std::string prefix) {
  // std::cout << prefix << " tensor \"" << ptr->Name() << "\" : ";
  // if (ptr->Data() == nullptr) {
  //   std::cout << " nullptr\n";
  //   return;
  // } else {
  //   std::cout << "\n";
  // }
  // std::cout << std::scientific << std::setprecision(4);
  // switch (ptr->Type()) {
  //   case common::kF32: tensor_printer<float>(ptr); break;
  //   case common::kI32: tensor_printer<uint32_t>(ptr); break;
  //   case common::kI64: tensor_printer<uint64_t>(ptr); break;
  //   default: CHECK(false);
  // }
}

GradTensor::GradTensor(bool require_grad) : _require_grad(require_grad) {
  _data = Tensor::Null();
  _grad = Tensor::Null();
}
GradTensor::GradTensor(TensorPtr d, bool require_grad) : _data(d), _require_grad(require_grad) {
  _data_actual_size = Device::Get(_data->Ctx())->WorkspaceActualSize(_data->Ctx(), _data->MutableData());
  if (require_grad) {
    _grad = Tensor::Empty(d->Type(), d->Shape(), d->Ctx(), "");
    _grad_actual_size = Device::Get(_grad->Ctx())->WorkspaceActualSize(_grad->Ctx(), _grad->MutableData());
  } else {
    _grad = Tensor::Null();
  }
}
GradTensor::GradTensor(common::DataType dt, std::vector<size_t> shape, Context ctx,
                       std::string name, bool require_grad)
    : _require_grad(require_grad) {
  _data = Tensor::Empty(dt, shape, ctx, name);
  _data_actual_size = Device::Get(_data->Ctx())->WorkspaceActualSize(_data->Ctx(), _data->MutableData());
  if (require_grad) {
    _grad = Tensor::Empty(dt, shape, ctx, name);
    _grad_actual_size = Device::Get(_grad->Ctx())->WorkspaceActualSize(_grad->Ctx(), _grad->MutableData());
  } else {
    _grad = Tensor::Null();
  }
}

void GradTensor::Resize(common::DataType dtype, std::vector<size_t> shape, Context ctx,
                        std::string name) {
  if (_data_actual_size >= common::GetTensorBytes(dtype, shape)) {
    _data->ForceChangeShape(dtype, shape, ctx, name);
  } else {
    _data->ChangeShape(dtype, shape, ctx, name);
    _data_actual_size = Device::Get(_data->Ctx())->WorkspaceActualSize(_data->Ctx(), _data->MutableData());
  }

  if (_require_grad || _grad->Defined()) {
    if (_grad_actual_size >= common::GetTensorBytes(dtype, shape)) {
      _grad->ForceChangeShape(dtype, shape, ctx, name);
    } else {
      _grad->ChangeShape(dtype, shape, ctx, name);
      _grad_actual_size = Device::Get(_grad->Ctx())->WorkspaceActualSize(_grad->Ctx(), _grad->MutableData());
    }
  }
  
}

GradTensorPtr GradTensor::Null(bool require_grad) {
  return std::make_shared<GradTensor>(require_grad);
}
GradTensorPtr GradTensor::FromTensor(TensorPtr d, bool require_grad) {
  return std::make_shared<GradTensor>(d, require_grad);
}
GradTensorPtr GradTensor::Empty(common::DataType dt, std::vector<size_t> shape, Context ctx,
                                std::string name, bool require_grad) {
  return std::make_shared<GradTensor>(dt, shape, ctx, name, require_grad);
}

} // namespace sam_backend
} // namespace samgraph