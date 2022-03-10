#include "model.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cusparse.h>

namespace samgraph {
namespace sam_backend {

namespace {
#ifdef DEAD_CODE
template <typename T, size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void scatter_gather_kernel(ScatterGather *op, const IdType *o_v_id, const IdType *i_v_id,
                                      size_t num_e, const T *i_v_h, T *o_v_h, const IdType hidden_dim) {

  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
#pragma unroll
  for (size_t i = threadIdx.x + block_start; i < block_end; i += BLOCK_SIZE) {
    if (i < num_e) {
      for (size_t h = 0; h < hidden_dim; h++) {
        atomicAdd(&o_v_h[o_v_id[i] * hidden_dim + h], i_v_h[i_v_id[i] * hidden_dim + h]);
      }
    }
  }
}
void scatter_gather(ScatterGather *op, TensorPtr o_v_id, TensorPtr i_v_id, TensorPtr i_v_h,
                    TensorPtr o_v_h) {
  size_t num_e = o_v_id->Shape()[0];
  const IdType hidden_dim = i_v_h->Shape()[1];
  const size_t num_tiles = RoundUpDiv(num_e, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  ZerosInitializer *init = new ZerosInitializer();
  init->init(op->_model, o_v_h);
  delete init;
  scatter_gather_kernel<TrainType, Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, op->_model->stream>>>(
          op, tensor_cast_const<IdType>(o_v_id), tensor_cast_const<IdType>(i_v_id), num_e,
          tensor_cast_const<TrainType>(i_v_h), tensor_cast<TrainType>(o_v_h), hidden_dim);
  CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
}
#endif

/** m x k * k x n => m x n */
template <typename T>
void spmm(ScatterGather *op, cusparseSpMatDescr_t A_desc, T *B_vals, T *C_vals, const IdType m,
          const IdType n, const IdType k, bool trans_A = false) {
  // {
  //   int64_t rows, cols, nnz;
  //   cusparseSpMatGetSize(A_desc, &rows, &cols, &nnz);
  //   if (trans_A == false) {
  //     CHECK(rows == m);
  //     CHECK(cols == k);
  //   } else {
  //     CHECK(rows == k);
  //     CHECK(cols == m);
  //   }
  // }
  // {
  //   TensorPtr B_ten = Tensor::CopyBlob(B_vals, to_data_type<T>(), {k, n}, op->_model->ctx,
  //                                      common::CPU(), "spmm_B_cpu");
  //   T *B_ptr = tensor_cast<T>(B_ten);
  //   for (size_t i = 0; i < B_ten->NumItem(); i++)
  //     CHECK(B_ptr[i] <= std::numeric_limits<T>::max());
  // }
  // {
  //   TensorPtr C_ten = Tensor::CopyBlob(C_vals, to_data_type<T>(), {m, n}, op->_model->ctx,
  //                                      common::CPU(), "spmm_C_cpu");
  //   T *C_ptr = tensor_cast<T>(C_ten);
  //   for (size_t i = 0; i < C_ten->NumItem(); i++)
  //     CHECK(C_ptr[i] <= std::numeric_limits<T>::max());
  // }

  cusparseHandle_t sparse = op->_model->sparse;

  cudaDataType dt = to_cuda_data_type<T>();

  cusparseDnMatDescr_t B_desc;
  CUSPARSE_CALL(
      cusparseCreateDnMat(&B_desc, k, n, n, B_vals, dt, cusparseOrder_t::CUSPARSE_ORDER_ROW));

  cusparseDnMatDescr_t C_desc;
  CUSPARSE_CALL(
      cusparseCreateDnMat(&C_desc, m, n, n, C_vals, dt, cusparseOrder_t::CUSPARSE_ORDER_ROW));

  T alpha = 1, beta = 0;

  size_t buffer_size;

  cusparseOperation_t op_A_ =
      trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMMAlg_t alg_ = CUSPARSE_SPMM_ALG_DEFAULT;

  CUSPARSE_CALL(cusparseSpMM_bufferSize(sparse, op_A_, op_, &alpha, A_desc, B_desc, &beta, C_desc,
                                        dt, alg_, &buffer_size));
  void *&spmm_buffer = op->buffer;
  if (buffer_size > op->buffer_size) {
    if (spmm_buffer) {
      GPUDevice::Get(op->_model->ctx)->FreeWorkspace(op->_model->ctx, spmm_buffer);
    }
    spmm_buffer = GPUDevice::Get(op->_model->ctx)->AllocWorkspace(op->_model->ctx, buffer_size);
    op->buffer_size = GPUDevice::Get(op->_model->ctx)->WorkspaceActualSize(op->_model->ctx, spmm_buffer);
  }

  CUSPARSE_CALL(cusparseSpMM(sparse, op_A_, op_, &alpha, A_desc, B_desc, &beta, C_desc, dt, alg_,
                             spmm_buffer));
  // fixme: is it safe to destroy desc here?
  CUSPARSE_CALL(cusparseDestroyDnMat(B_desc));
  CUSPARSE_CALL(cusparseDestroyDnMat(C_desc));
  if (common::RunConfig::option_samback_cuda_launch_blocking) {
    CUDA_CALL(cudaStreamSynchronize(op->_model->stream));
  }
}

#ifdef DEAD_CODE
template <typename T>
__global__ void aggre_coop_kernel(IdType num_output_vertex, int hiddenDim, const IdType *indptr,
                                  const IdType *col_idxs, const T *input, T *output) {
  // assert(blockDim.x % hiddenDim == 0);
  // assert(aggrType == AGGR_SUM || aggrType == AGGR_AVG);
  int vtxPerBlock = CUDA_NUM_THREADS / hiddenDim;
  typedef cub::BlockScan<IdType, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ IdType blkColStart;
  __shared__ T acc_h[CUDA_NUM_THREADS];
  int tidDiv = threadIdx.x / hiddenDim;
  int tidMod = threadIdx.x % hiddenDim;
  for (IdType cur_iter_output_v_base = blockIdx.x * vtxPerBlock + 0;
       cur_iter_output_v_base <= num_output_vertex;
       cur_iter_output_v_base += vtxPerBlock * gridDim.x) {
    IdType myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    if (threadIdx.x + cur_iter_output_v_base <= num_output_vertex && threadIdx.x < vtxPerBlock) {
      IdType curVtx = threadIdx.x + cur_iter_output_v_base;
      IdType startColIdx = indptr[curVtx], endColIdx = indptr[curVtx + 1];
      myNumEdges = endColIdx - startColIdx;
      if (threadIdx.x == 0)
        blkColStart = startColIdx;
    }
    acc_h[threadIdx.x] = 0.0f;
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    IdType done = 0;
    while (totalNumEdges > 0) {
      if (tidDiv < totalNumEdges && tidDiv < vtxPerBlock) {
        EdgeStruct es = col_idxs[blkColStart + done + tidDiv - 0];
        T val = input[es.src * hiddenDim + tidMod];
        assert(es.dst >= cur_iter_output_v_base && es.dst < cur_iter_output_v_base + vtxPerBlock);
        int offset = (es.dst - cur_iter_output_v_base) * hiddenDim + tidMod;
        atomicAdd(&acc_h[offset], val);
      }
      done += vtxPerBlock;
      totalNumEdges -= (totalNumEdges > vtxPerBlock) ? vtxPerBlock : totalNumEdges;
    }
    __syncthreads();
    if (tidDiv < vtxPerBlock && tidDiv + cur_iter_output_v_base <= num_output_vertex) {
      output[(cur_iter_output_v_base - 0) * hiddenDim + threadIdx.x] = acc_h[threadIdx.x];
    }
  }
}
#endif
} // namespace

ScatterGather::ScatterGather(Model *model, const GradTensorPtr input, size_t layer_idx)
    : GnnOp(model, input), layer_idx(layer_idx) {
  output = GradTensor::Null(input->RequireGrad());
}
ScatterGather::~ScatterGather() {
  if (buffer) {
    Device::Get(_model->ctx)->FreeWorkspace(_model->ctx, buffer);
  }
}

void ScatterGather::forward() {
  if (check_nan_exist(inputs[0]->data())) std::cout << "sg founds nan\n";
  TrainType *input_h = tensor_cast<TrainType>(this->inputs[0]->data());
  TrainType *output_h = tensor_cast<TrainType>(this->output->data());
  IdType num_input_v = inputs[0]->Shape()[0];
  IdType num_output_v = output->Shape()[0];
  IdType hidden_dim = output->Shape()[1];
  IdType num_e = _model->cur_task->graphs[layer_idx]->num_edge;
  CUSPARSE_CALL(cusparseCreateCoo(&A_T, num_output_v, num_input_v, num_e, tensor_cast<IdType>(col),
                                  tensor_cast<IdType>(row), tensor_cast<TrainType>(_model->ones),
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                  to_cuda_data_type<TrainType>()));
  // we should be passing in A^T !!
  spmm<TrainType>(this, A_T, input_h, output_h, num_output_v, hidden_dim, num_input_v);
  // scatter_gather(this, col, row, inputs[0]->data(), output->data());
}

void ScatterGather::backward() {
  if (inputs[0]->RequireGrad() == false) return;
  TrainType *input_grad = tensor_cast<TrainType>(this->inputs[0]->grad());
  TrainType *output_grad = tensor_cast<TrainType>(this->output->grad());
  IdType num_input_v = inputs[0]->Shape()[0];
  IdType num_output_v = output->Shape()[0];
  IdType hidden_dim = output->Shape()[1];
  spmm<TrainType>(this, A_T, output_grad, input_grad, num_input_v, hidden_dim, num_output_v, true);
  // scatter_gather(this, row, col, output->grad(), inputs[0]->grad());
  CUSPARSE_CALL(cusparseDestroySpMat(A_T));
}


void ScatterGather::prepare() {
  TaskPtr cur_task = _model->cur_task;

  TrainGraphPtr graph = cur_task->graphs[layer_idx];

  IdType hidden_dim = inputs[0]->Shape()[1];
  row = graph->row;
  col = graph->col;

  IdType num_input_v = graph->num_src;
  IdType num_output_v = graph->num_dst;
  IdType num_e = graph->num_edge;
  CHECK(num_e == row->Shape()[0]);
  CHECK(num_e == col->Shape()[0]);
  CHECK(num_input_v == inputs[0]->Shape()[0]);
  CHECK(inputs[0]->Ctx() == _model->ctx);
  CHECK(row->Ctx() == _model->ctx);
  CHECK(col->Ctx() == _model->ctx);
  // {
  //   TensorPtr cpu_row = Tensor::CopyTo(row, common::CPU());
  //   IdType *row_ptr = tensor_cast<IdType>(cpu_row);
  //   for (size_t i = 0; i < cpu_row->Shape()[0]; i++)
  //     CHECK(row_ptr[i] < num_input_v);
  // }
  // {
  //   TensorPtr cpu_col = Tensor::CopyTo(col, common::CPU());
  //   IdType *col_ptr = tensor_cast<IdType>(cpu_col);
  //   for (size_t i = 0; i < cpu_col->Shape()[0]; i++)
  //     CHECK(col_ptr[i] < num_output_v);
  // }

  if (!_model->ones->Defined() || _model->ones->Shape()[0] < graph->num_edge) {
    _model->ones->ChangeShape(inputs[0]->Type(), {graph->num_edge}, _model->ctx, "_model->ones");
    ValInitializer *init = new ValInitializer(1.0);
    init->init(_model, _model->ones);
    delete init;
  }
  // CUSPARSE_CALL(cusparseCreateCoo(&A_T, num_output_v, num_input_v, num_e, tensor_cast<IdType>(col),
  //                                 tensor_cast<IdType>(row), tensor_cast<TrainType>(_model->ones),
  //                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
  //                                 to_cuda_data_type<TrainType>()));

  output->Resize(inputs[0]->Type(), {num_output_v, hidden_dim}, _model->ctx, "");
  // sort_coo(col, row, num_output_v, num_input_v, _model->sparse, _model->stream);
  // {
  //   TensorPtr row_ten = Tensor::CopyTo(row, common::CPU());
  //   IdType *row_ptr = tensor_cast<IdType>(row_ten);
  //   CHECK(row_ptr[0] < num_input_v);
  //   TensorPtr col_ten = Tensor::CopyTo(col, common::CPU());
  //   IdType *col_ptr = tensor_cast<IdType>(col_ten);
  //   CHECK(col_ptr[0] < num_output_v);
  //   for (size_t i = 1; i < col_ten->Shape()[0]; i++) {
  //     CHECK(col_ptr[i] < num_output_v);
  //     CHECK(row_ptr[i] < num_input_v);
  //     CHECK(col_ptr[i - 1] <= col_ptr[i]);
  //     if (col_ptr[i - 1] == col_ptr[i])
  //       CHECK(row_ptr[i - 1] <= row_ptr[i]);
  //   }
  // }
  // {
  //   TensorPtr A_ten = Tensor::CopyTo(ones, common::CPU());
  //   TrainType *A_ptr = tensor_cast<TrainType>(A_ten);
  //   for (size_t i = 0; i < num_e; i++)
  //     CHECK(A_ptr[i] == (TrainType)1.0);
  // }
}

bool ScatterGather::generate_grad()   const {return true;}
bool ScatterGather::accumulate_grad() const {return false;}
std::string ScatterGather::name() const {return "scatter gather";}

} // namespace sam_backend
} // namespace samgraph