#include "model.h"
#include "optimizer.h"
#include "../common/profiler.h"
#include <iostream>

namespace samgraph {
namespace sam_backend {

GnnOp::GnnOp(Model* model, GradTensorPtr input) {
  _model = model;
  numInputs = 1;
  inputs[0] = input;
  accumulate_input_grad[0] = false;
}

GnnOp::GnnOp(Model* model, GradTensorPtr input0, GradTensorPtr input1) {
  _model = model;
  numInputs = 2;
  inputs[0] = input0;
  inputs[1] = input1;
  accumulate_input_grad[0] = false;
  accumulate_input_grad[1] = false;
}
GnnOp::~GnnOp() {}

void Model::forward(TaskPtr task) {
  // we must make sure that input_feat.data points to the same Tensor
  CUDA_CALL(cudaSetDevice(ctx.device_id));
  cur_task = task;
  input_feat->ChangeData(task->input_feat);
  for (size_t l = 0; l < ops.size(); l++) {
    ops[l]->prepare(); // resizing output
  }
  for (size_t l = 0; l < ops.size(); l++) {
    ops[l]->forward();
  }
}

void Model::backward(void) {
  for (int l = ops.size() - 1; l >= 0; l--) {
    ops[l]->backward();
  }
}

void Model::update(void) {
  optimizer->next();
  optimizer->update();
}
// void Model::zero_gradients(void) {
//   IndexLauncher launcher(ZERO_GRAD_TASK_ID, taskIS,
//                          TaskArgument(NULL, 0), taskArgs);
//   for (size_t p = 0; p < parameters.size(); p++) {
//     // regions[p]: region_grad
//     launcher.add_region_requirement(
//         RegionRequirement(parameters[p].part_grad, 0 /*projection*/,
//                           WRITE_ONLY, EXCLUSIVE, parameters[p].region_grad,
//                           MAP_TO_FB_MEMORY));
//     launcher.add_field(p, FID_DATA);
//   }
//   runtime->execute_index_space(ctx, launcher);
// }

namespace {

void _assign_reset_task(GnnOp *op, size_t input_idx) {
  GradTensorPtr input = op->inputs[input_idx];
  if (input->RequireGrad() == false) return;
  if (input->GetConsumer() == 0) {
    // currently we assert all gnn op can reset grad
    CHECK(op->generate_grad());
    CHECK(op->reset_grad());
    op->accumulate_input_grad[input_idx] = false;
    input->AddConsumer();
  } else {
    if (op->generate_grad()) {
      CHECK(op->accumulate_grad());
      op->accumulate_input_grad[input_idx] = true;
      input->AddConsumer();
    }
  }
}

}

Model* Model::_singleton = nullptr;

void Model::assign_reset_task() {
  for (int layer_idx = ops.size() - 1; layer_idx >= 0; layer_idx --) {
    for (size_t input_idx = 0; input_idx < ops[layer_idx]->numInputs; input_idx++) {
      _assign_reset_task(ops[layer_idx], input_idx);
    }
  }
}

GradTensorPtr Model::relu(const GradTensorPtr _input) {
  Activation * op = new Activation(this, _input, KAcModeRelu);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::sigmoid(const GradTensorPtr _input) {
  Activation * op = new Activation(this, _input, KAcModeSigmoid);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::dropout(const GradTensorPtr _input, float rate, int seed) {
  Dropout * op = new Dropout(this, _input, rate, seed);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::add(const GradTensorPtr _input1, const GradTensorPtr _input2) {
  Element * op = new Element(this, _input1, _input2, kEwTypeAdd);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::mul(const GradTensorPtr _input1, const GradTensorPtr _input2) {
  Element * op = new Element(this, _input1, _input2, kEwTypeMul);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::indegree_norm(const GradTensorPtr _input, size_t layer_idx, NormMode norm_mode) {
  InDegreeNorm * op = new InDegreeNorm(this, _input, layer_idx, norm_mode);
  ops.push_back(op);
  return op->output;
}
GradTensorPtr Model::linear(const GradTensorPtr _input, int in_dim, int out_dim, Initializer* init) {
  Linear * op = new Linear(this, _input, in_dim, out_dim, init);
  ops.push_back(op);
  parameters.push_back(op->weight);
  return op->output;
}
GradTensorPtr Model::partial_lienar(const GradTensorPtr _input, int in_dim, int out_dim, std::function<IdType(void)> source_of_num, Initializer* init) {
  PartialLinear * op = new PartialLinear(this, _input, in_dim, out_dim, source_of_num, init);
  ops.push_back(op);
  parameters.push_back(op->weight);
  return op->output;
}
GradTensorPtr Model::bias(const GradTensorPtr _input, int dim) {
  Bias * op = new Bias(this, _input, dim);
  ops.push_back(op);
  parameters.push_back(op->_bias);
  return op->output;
}
GradTensorPtr Model::scatter_gather(const GradTensorPtr _input, size_t layer_idx) {
  ScatterGather * op = new ScatterGather(this, _input, layer_idx);
  ops.push_back(op);
  return op->output;
}
void Model::softmax_cross_entropy(const GradTensorPtr logits) {
  SoftmaxCrossEntropy * op = new SoftmaxCrossEntropy(this, logits);
  ops.push_back(op);
  return;
}

Model::Model(Context ctx) {
  mode = KModeTrain;
  this->ctx = ctx;
  stream = static_cast<cudaStream_t>(Device::Get(ctx)->CreateStream(ctx));
  input_feat = GradTensor::Null(false);
  CUDNN_CALL(cudnnCreate(&this->dnn));
  CUDNN_CALL(cudnnSetStream(this->dnn, this->stream));
  CUBLAS_CALL(cublasCreate_v2(&this->blas));
  CUBLAS_CALL(cublasSetStream_v2(this->blas, this->stream));
  CUSPARSE_CALL(cusparseCreate(&this->sparse));
  CUSPARSE_CALL(cusparseSetStream(this->sparse, this->stream));
  ones = Tensor::Null();
}
Model::~Model() {
  CUDNN_CALL(cudnnDestroy(this->dnn));
  CUBLAS_CALL(cublasDestroy_v2(this->blas));
  CUSPARSE_CALL(cusparseDestroy(this->sparse));
}

void Model::adam_optimize(TrainType lr, TrainType weight_decay) {
  AdamOptimizer * op = new AdamOptimizer(this, lr, weight_decay);
  this->optimizer = op;
}

void Model::naive_optimize(TrainType lr) {
  NaiveOptimizer * op = new NaiveOptimizer(this, lr);
  this->optimizer = op;
}

void Model::loss(float &loss, float &accuracy) {
  SoftmaxCrossEntropy* op = dynamic_cast<SoftmaxCrossEntropy*>(ops.back());
  if (op == nullptr) CHECK(false);
  op->loss(loss, accuracy);
  std::string modeInfo = (mode == KModeTrain) ? "[TRAIN]" : "\t[INFER]";
  // if (_model->mode == KModeInfer) {
  printf("%s loss: %.4lf  accuracy: %.2lf%%\n", modeInfo.c_str(), loss, accuracy);
}

} // namespace sam_backend
} // namespace samgraph