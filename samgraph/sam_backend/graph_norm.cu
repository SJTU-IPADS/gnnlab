#include "model.h"
#include "utils.h"
#include <cassert>
#include <iostream>

namespace samgraph {
namespace sam_backend {

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void coo_to_degree(IdType *degree, const IdType *id_list, size_t num_e) {
  assert(BLOCK_SIZE == blockDim.x);
  // fixme: roc use flatterned parallel, maybe we can do the same? rather than assign one node to
  // one thread?
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
  size_t e_id = threadIdx.x + block_start;
  IdType curr_id = Constant::kEmptyKey;
  IdType curr_cnt = 0;
  if (e_id < num_e)
    curr_id = id_list[e_id];
  for (; e_id < block_end; e_id += BLOCK_SIZE) {
    if (e_id < num_e) {
      if (curr_id == id_list[e_id]) {
        curr_cnt++;
        continue;
      } else {
        assert(curr_id != Constant::kEmptyKey);
        atomicAdd(degree + curr_id, curr_cnt);
        curr_id = id_list[e_id];
        curr_cnt = 1;
      }
    }
  }
  if (curr_id != Constant::kEmptyKey) {
    atomicAdd(degree + curr_id, curr_cnt);
  }
}

template <typename T, NormMode norm_mode>
__global__ void norm_coop_kernel(const IdType *degree, const T *input, T *output,
                                 const size_t num_input, const size_t hiddenDim) {

  const size_t stride = blockDim.y * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.y + threadIdx.y; i < num_input; i += stride) {
    {
      T factor = 1.0 / (T)degree[i];
      if (norm_mode == kNormModeSqrt) {
        factor = 1.0 / sqrt((T)(degree[i]));
      }
      if (degree[i] == 0)
        factor = 0;
      const T *local_input = input + i * hiddenDim;
      T *local_output = output + i * hiddenDim;
      for (size_t h = threadIdx.x; h < hiddenDim; h += blockDim.x) {
        local_output[h] = local_input[h] * factor;
      }
    }
  }
}

template <typename T>
__host__ void forward_task(InDegreeNorm *op, const IdType *degree, const T *input, T *output,
                           const size_t num_input, const size_t hiddenDim) {
  dim3 block(256, 1);
  while (static_cast<size_t>(block.x) >= 2 * hiddenDim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_input, static_cast<size_t>(block.y)));

  switch (op->_norm_mode) {
  case kNormModeDirect: {
    norm_coop_kernel<T, kNormModeDirect>
        <<<grid, block, 0, op->_model->stream>>>(degree, input, output, num_input, hiddenDim);
    break;
  }
  case kNormModeSqrt: {
    norm_coop_kernel<T, kNormModeSqrt>
        <<<grid, block, 0, op->_model->stream>>>(degree, input, output, num_input, hiddenDim);
    break;
  }
  }
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

} // namespace

InDegreeNorm::InDegreeNorm(Model *model, const GradTensorPtr input, size_t layer_idx,
                           NormMode norm_mode)
    : _layer_idx(layer_idx), _norm_mode(norm_mode), GnnOp(model, input) {
  output = GradTensor::Null(input->RequireGrad());
}

void InDegreeNorm::forward() {
  if (check_nan_exist(inputs[0]->data()))
    std::cout << "norm founds nan\n";
  size_t num_input = inputs[0]->Shape()[0];
  size_t hidden_dim = inputs[0]->Shape()[1];
  const TrainType *input_ptr = tensor_cast_const<TrainType>(inputs[0]->data());
  const IdType *degree_ptr = tensor_cast_const<IdType>(_model->cur_task->graphs[_layer_idx]->dst_degree);
  TrainType *output_ptr = tensor_cast<TrainType>(output->data());
  PrintTensor(_model->cur_task->graphs[_layer_idx]->dst_degree, "degree list");
  forward_task<TrainType>(this, degree_ptr, input_ptr, output_ptr, num_input, hidden_dim);
  if (check_nan_exist(output->data()))
    std::cout << "norm founds nan in output\n";
}
void InDegreeNorm::backward() {
  if (inputs[0]->RequireGrad() == false)
    return;
  size_t num_input = inputs[0]->Shape()[0];
  size_t hidden_dim = inputs[0]->Shape()[1];
  const TrainType *output_grad = tensor_cast_const<TrainType>(output->grad());
  const IdType *degree = tensor_cast_const<IdType>(_model->cur_task->graphs[_layer_idx]->dst_degree);
  TrainType *input_grad = tensor_cast<TrainType>(inputs[0]->grad());
  // forward & backward do the same thing
  forward_task<TrainType>(this, degree, output_grad, input_grad, num_input, hidden_dim);
}

void InDegreeNorm::prepare() {

  TaskPtr cur_task = _model->cur_task;

  TrainGraphPtr graph = cur_task->graphs[this->_layer_idx];

  CHECK(inputs[0]->data()->Shape()[0] == graph->num_dst);
  if (graph->dst_degree == nullptr) {
    size_t num_v = graph->num_dst;
    size_t num_e = graph->num_edge;
    const size_t num_tiles = RoundUpDiv(num_e, Constant::kCudaTileSize);
    const dim3 grid(num_tiles);
    const dim3 block(Constant::kCudaBlockSize);
    graph->dst_degree = Tensor::Empty(common::kI32, {num_v}, _model->ctx, "");
    ZerosInitializer * init = new ZerosInitializer();
    init->init(this->_model, graph->dst_degree);
    delete init;
    coo_to_degree<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, _model->stream>>>(tensor_cast<IdType>(graph->dst_degree),
                                             tensor_cast_const<IdType>(graph->col), num_e);
    CUDA_CALL(cudaStreamSynchronize(_model->stream));
  }

  output->Resize(inputs[0]->Type(), inputs[0]->Shape(), _model->ctx, "");
}

bool InDegreeNorm::generate_grad()   const {return true;}
bool InDegreeNorm::accumulate_grad() const {return false;}
std::string InDegreeNorm::name() const {return "degree norm";}

} // namespace sam_backend
} // namespace samgraph