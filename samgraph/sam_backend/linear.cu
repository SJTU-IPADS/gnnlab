#include "model.h"
#include "utils.h"
#include <cublas_v2.h>
#include <iostream>
namespace samgraph {
namespace sam_backend {

/**
 * cublas sgemm use col-major matrix
 * our input/output is stored with continious per-vertex feature, so in cublas,
 * the input/output matrix should be in_dim*len / out_dim*len
 *
 * for debug purpose, the weight is in shape [out_dim, in_dim] and cublas treats
 * it as in_dim * out_dim
 */
namespace {
// weight is in_dim * out_dim
// out_dim * in_dim  x  in_dim * len  ==>  out_dim*len
//        W^T               input             output
template <typename T>
__host__ void forward_task(cublasHandle_t blas, const T *weight, const T *input, T *output,
                           const size_t in_dim, const size_t len, const size_t out_dim,
                           cudaStream_t stream) {
  // Weight matches outDim
  T alpha = 1.0f, beta = 0.0f;
  // 1 2 3: 1x3 * 3x2 -> 1x2
  CUBLAS_CALL(cublasSgemm_v2(blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, len, in_dim,
                             &alpha, weight, in_dim,   // weight is in_dim * out_dim
                             input, in_dim,            // input is in_dim * len
                             &beta, output, out_dim)); // output is out_dim * len
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

template <typename T>
__host__ void backward_task(cublasHandle_t blas, const T *weight, const T *input, const T *output_grad,
                            T *weight_grad, T *input_grad, const size_t in_dim, const size_t len,
                            const size_t out_dim, cudaStream_t stream, bool accumulate_input_grad) {
  T alpha = 1.0f, beta = 0.0f;
  if (accumulate_input_grad) beta = 1.0f;
  // Compute weight_grad
  // Note that we use alpha = 1.0 to accumulate weight gradients)
  // weight is in_dim * out_dim
  // in_dim * out_dim  x  out_dim * len  ==>  in_dim * len
  //        W                  o_g                i_g
  if (input_grad) {
    CUBLAS_CALL(cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_N, in_dim, len, out_dim,
                               &alpha, weight, in_dim,      // weight      is  in_dim * out_dim
                               output_grad, out_dim,        // output_grad is out_dim * len
                               &beta, input_grad, in_dim)); // input_grad  is  in_dim * len
  }
  // Compute input_grad
  // Note that we use alpha = 1.0 to accumulate input gradients
  // weight is in_dim * out_dim
  // in_dim * len  x  len * out_dim  ==>  in_dim * out_dim
  //      input           o_g^T                  w_g
  beta = 0.0f;
  if (weight_grad) {
    CUBLAS_CALL(cublasSgemm_v2(blas, CUBLAS_OP_N, CUBLAS_OP_T, in_dim, out_dim, len,
                               &alpha, input, in_dim,        // input       is  in_dim * len
                               output_grad, out_dim,         // output_grad is out_dim * len
                               &beta, weight_grad, in_dim)); // weight_grad is  in_dim * out_dim
  }
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

} // namespace

Linear::Linear(Model *model, const GradTensorPtr input, IdType in_dim, IdType out_dim,
               Initializer *initializer)
    : GnnOp(model, input), in_dim(in_dim), out_dim(out_dim) {
  // weight
  weight = GradTensor::Empty(to_data_type<TrainType>(), {out_dim, in_dim}, model->ctx,
                             "linear_weight", true);
  output = GradTensor::Null(true);
  if (initializer == nullptr) {
    initializer = new GlorotUniform();
    initializer->init(model, weight->data());
    delete initializer;
  } else {
    initializer->init(model, weight->data());
  }
}

void Linear::forward() {
  if (check_nan_exist(inputs[0]->data()))
    std::cout << "linear founds nan\n";
  size_t num_input = inputs[0]->Shape()[0];
  CHECK(inputs[0]->Shape()[1] == in_dim);
  CHECK(output->Shape()[1] == out_dim);
  CHECK(inputs[0]->Shape()[0] == output->Shape()[0]);
  const TrainType *input_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *weight_ptr = tensor_cast_const<TrainType>(this->weight->data());
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  forward_task<TrainType>(_model->blas, weight_ptr, input_ptr, output_ptr, in_dim, num_input, out_dim,
                          _model->stream);
  if (check_nan_exist(output->data()))
    std::cout << "linear founds nan in output\n";
}
void Linear::backward() {
  size_t num_input = inputs[0]->Shape()[0];
  const TrainType *input = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *weight_ptr = tensor_cast_const<TrainType>(this->weight->data());
  const TrainType *output_grad = tensor_cast_const<TrainType>(output->grad());
  TrainType *input_grad = nullptr;
  if (inputs[0]->RequireGrad()) {
    input_grad = tensor_cast<TrainType>(inputs[0]->grad());
  }
  TrainType *weight_grad = tensor_cast<TrainType>(this->weight->grad());
  // forward & backward do the same thing
  backward_task<TrainType>(_model->blas, weight_ptr, input, output_grad, weight_grad, input_grad, in_dim,
                           num_input, out_dim, _model->stream, accumulate_input_grad[0]);
}

void Linear::prepare() {
  output->Resize(inputs[0]->Type(), {inputs[0]->Shape()[0], out_dim}, _model->ctx, "");
}
bool Linear::generate_grad()   const {return true;}
// bool Linear::accumulate_grad() const {return true;}
bool Linear::accumulate_grad() const {return false;}
std::string Linear::name() const {return "linear";}

PartialLinear::PartialLinear(Model *model, const GradTensorPtr input, IdType in_dim, IdType out_dim,
                             std::function<IdType(void)> source_of_num, Initializer *initializer)
    : GnnOp(model, input), in_dim(in_dim), out_dim(out_dim), source_of_num(source_of_num) {
  // weight
  weight = GradTensor::Empty(to_data_type<TrainType>(), {out_dim, in_dim}, model->ctx,
                             "linear_weight", true);
  output = GradTensor::Null(true);

  if (initializer == nullptr) {
    initializer = new GlorotUniform();
    initializer->init(model, weight->data());
    delete initializer;
  } else {
    initializer->init(model, weight->data());
  }
}

void PartialLinear::forward() {
  if (check_nan_exist(inputs[0]->data()))
    std::cout << "linear founds nan\n";
  IdType num_input = output->Shape()[0];
  CHECK(inputs[0]->Shape()[1] == in_dim);
  CHECK(output->Shape()[1] == out_dim);
  CHECK(num_input == output->Shape()[0]);
  CHECK(num_input <= inputs[0]->Shape()[0]);
  const TrainType *input_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *weight_ptr = tensor_cast_const<TrainType>(this->weight->data());
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  forward_task<TrainType>(_model->blas, weight_ptr, input_ptr, output_ptr, in_dim, num_input, out_dim,
                          _model->stream);
  if (check_nan_exist(output->data()))
    std::cout << "linear founds nan in output\n";
}
void PartialLinear::backward() {
  IdType num_input = output->Shape()[0];
  if (inputs[0]->RequireGrad() && num_input != inputs[0]->Shape()[0] && accumulate_input_grad[0] == false) {
    CHECK(false) << "partial linear cannot reset full grad of input now";
  }
  const TrainType *input = tensor_cast_const<TrainType>(inputs[0]->data());
  const TrainType *weight_ptr = tensor_cast_const<TrainType>(this->weight->data());
  const TrainType *output_grad = tensor_cast_const<TrainType>(output->grad());
  TrainType *input_grad = nullptr;
  if (inputs[0]->RequireGrad()) {
    input_grad = tensor_cast<TrainType>(inputs[0]->grad());
  }
  TrainType *weight_grad = tensor_cast<TrainType>(this->weight->grad());
  // forward & backward do the same thing
  backward_task<TrainType>(_model->blas, weight_ptr, input, output_grad, weight_grad, input_grad, in_dim,
                           num_input, out_dim, _model->stream, accumulate_input_grad[0]);
}

void PartialLinear::prepare() {
  IdType num_input = source_of_num();
  output->Resize(inputs[0]->Type(), {num_input, out_dim}, _model->ctx, "");
}
bool PartialLinear::generate_grad()   const {return true;}
// bool PartialLinear::accumulate_grad() const {return true;}
bool PartialLinear::accumulate_grad() const {return true;}
std::string PartialLinear::name() const {return "partial linear";}

} // namespace sam_backend
} // namespace samgraph