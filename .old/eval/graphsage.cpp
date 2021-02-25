#include <cstdio>
#include <memory>
#include <vector>

#include <cusparse.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <gflags/gflags.h>

#include "data/dataset.hpp"
#include "sampling/cpu/sampler.hpp"
#include "sampling/cpu/block.hpp"
#include "sampling/cpu/shuffler.hpp"
#include "train/torch_block.hpp"
#include "util/cuda.hpp"
#include "util/performance.hpp"
#include "util/tictoc.hpp"

DEFINE_string(dataset_key, "papers100M","");
DEFINE_string(dataset_folder, "/graph-learning/preprocess/papers100M", "");

DEFINE_int32(n_hidden, 256, "");
DEFINE_int32(n_layers, 3, "");
DEFINE_double(dropout, 0.5f, "");

DEFINE_string(fanout, "15,10,5", "");
DEFINE_int32(num_epoch, 200, "");
DEFINE_uint64(batch_size, 8192, "");
DEFINE_double(lr, 0.003, "");
DEFINE_bool(bias, true, "");

struct SPMM : torch::autograd::Function<SPMM> {
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx,
                                                  torch::autograd::Variable x,
                                                  torch::autograd::Variable csr_indptr,
                                                  torch::autograd::Variable csr_indices,
                                                  torch::autograd::Variable csr_val,
                                                  int csr_m,
                                                  int csr_n,
                                                  int csr_k,
                                                  int csr_nnz,
                                                  torch::autograd::Variable csc_indptr,
                                                  torch::autograd::Variable csc_indices,
                                                  torch::autograd::Variable csc_val,
                                                  int csc_m,
                                                  int csc_n,
                                                  int csc_k,
                                                  int csc_nnz) {
        ctx->save_for_backward({csc_indptr, csc_indices, csc_val});
        ctx->saved_data["csc_m"] = csc_m;
        ctx->saved_data["csc_n"] = csc_n;
        ctx->saved_data["csc_k"] = csc_k;
        ctx->saved_data["csc_nnz"] = csc_nnz;

        const float alpha = 1.0;
        cusparseMatDescr_t descrA;
        int ldb = csr_k;
        const float beta = 0.0;
        int ldc = csr_m;
        float *C;
        
        CUDA_CALL(cudaMalloc((void **)&C, csr_m * csr_n *sizeof(float)))

        CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
        CUSPARSE_CALL(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CALL(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
        CUSPARSE_CALL(cusparseScsrmm(*cusparse_handle(),
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       csr_m, csr_n, csr_k, csr_nnz, &alpha,
                       descrA,
                       csr_val.data_ptr<float>(),
                       csr_indptr.data_ptr<int>(),
                       csr_indices.data_ptr<int>(),
                       x.data_ptr<float>(),
                       ldb,
                       &beta,
                       C,
                       ldc
                       ));

        torch::Tensor output = torch::from_blob(
            C,
            {(long long) csr_m, (long long) csr_n},
            [](void *data) {
                CUDA_CALL(cudaFree(data));
            },
            torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

        return {output};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
        auto saved = ctx->get_saved_variables();
        torch::autograd::Variable csc_indptr = saved[0];
        torch::autograd::Variable csc_indices = saved[1];
        torch::autograd::Variable csc_val = saved[2];

        int csc_m = ctx->saved_data["csc_m"].toInt();
        int csc_n = ctx->saved_data["csc_n"].toInt();
        int csc_k = ctx->saved_data["csc_k"].toInt();
        int csc_nnz = ctx->saved_data["csc_nnz"].toInt();

        const float alpha = 1.0;
        cusparseMatDescr_t descrA;
        const float *B;
        int ldb = csc_k;
        const float beta = 0.0;
        int ldc = csc_m;
        float *C;
        
        CUDA_CALL(cudaMalloc((void **)&C, csc_m * csc_n *sizeof(float)))

        CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
        CUSPARSE_CALL(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CALL(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
        CUSPARSE_CALL(cusparseScsrmm(*cusparse_handle(),
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       csc_m, csc_n, csc_k, csc_nnz, &alpha,
                       descrA,
                       csc_val.data_ptr<float>(),
                       csc_indptr.data_ptr<int>(),
                       csc_indices.data_ptr<int>(),
                       grad_output[0].data_ptr<float>(),
                       ldb,
                       &beta,
                       C,
                       ldc
                       ));

        torch::Tensor grad_input = torch::from_blob(
            C,
            {(long long) csc_m, (long long) csc_n},
            [](void *data) {
                CUDA_CALL(cudaFree(data));
            },
            torch::TensorOptions().dtype(torch::kFloat32).device(grad_output[0].device()));

        return {grad_input, torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

struct SAGEConv : torch::nn::Module {
    SAGEConv(int in_feats, int out_feats, bool bias) : in_feats(in_feats), out_feats(out_feats) {
        fc_self = register_module("fc_self", torch::nn::Linear(torch::nn::LinearOptions(in_feats, out_feats).bias(bias)));
        fc_neigh = register_module("fc_neigh", torch::nn::Linear(torch::nn::LinearOptions(in_feats, out_feats).bias(bias)));
        ResetParam();
    }

    torch::Tensor forward(TorchCSR &csr, TorchCSR &csc, const torch::Tensor &x) {
        torch::Tensor feat_src = x;
        torch::Tensor feat_dst = x.slice(0, 0, csr.m).clone();

        torch::Tensor h_self = feat_dst;
        int n = in_feats;
        torch::Tensor h_neigh = SPMM::apply(feat_src, csr.indptr, csr.indices, csr.val, csr.m, n, csr.k, csr.nnz, 
                                             csc.indptr, csc.indices, csc.val, csc.m, n, csc.k, csc.nnz)[0];
        
        std::cout << h_self.sizes() << '\t' << h_self.device() << "\t" << h_neigh.sizes() << '\t' << h_neigh.device() << std::endl;
        torch::Tensor rst = fc_self->forward(h_self) + fc_neigh->forward(h_neigh);
        
        return rst;
    }

    void ResetParam() {
        double gain = torch::nn::init::calculate_gain(torch::kReLU);
        torch::nn::init::xavier_uniform_(fc_self->weight, gain);
        torch::nn::init::xavier_uniform_(fc_neigh->weight, gain);
    }

    torch::nn::Linear fc_self = nullptr;
    torch::nn::Linear fc_neigh = nullptr;
    int in_feats;
    int out_feats;
};

struct SAGE : torch::nn::Module {
    SAGE(int in_feats,
         int n_hidden,
         int n_classes,
         int n_layers,
         double dropout,
         bool bias)
         : dropout(dropout) {
        int layer_id = 0;
        
        layers.push_back(register_module("layer" + std::to_string(layer_id), std::make_shared<SAGEConv>(in_feats, n_hidden, bias)));
        for (layer_id = 1; layer_id <  n_layers - 1; layer_id++) {
            layers.push_back(register_module("layer" + std::to_string(layer_id), std::make_shared<SAGEConv>(n_hidden, n_hidden, bias)));
        }
        layers.push_back(register_module("layer" + std::to_string(layer_id), std::make_shared<SAGEConv>(n_hidden, n_classes, bias)));
    }

    torch::Tensor forward(TorchBlocks &blocks) {
        torch::Tensor h = blocks.feature;
        for (int i = 0; i < layers.size(); i++) {
            h = layers[i]->forward(blocks.csr[i], blocks.csc[i], h);
            if (i != layers.size() -  1) {
                h = torch::relu(h); // activation
                h = torch::dropout(h, dropout, is_training()); //  dropout
            }
        }
        h = torch::log_softmax(h, 1);
        return h;
    }

    std::vector<std::shared_ptr<SAGEConv>> layers;
    double dropout;
};

int main(int argc, char *argv[]) {
    gflags::SetUsageMessage("some usage message");
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(FLAGS_dataset_key, FLAGS_dataset_folder);

    SamplingTask task;
    std::stringstream ss(FLAGS_fanout);
    for (int i; ss >> i;) {
        task.fanout.push_back(i);
        if (ss.peek() == ',') {
            ss.ignore();
        }
    }
    task.num_blocks = task.fanout.size();

    Performance &p = Performance::Instance();
    p.SetMeta(task.num_blocks);

    torch::Device device("cuda:0");
    auto model = std::make_shared<SAGE>(dataset->GetFeature().dim, FLAGS_n_hidden, dataset->GetLabel().num_classes, FLAGS_n_layers, FLAGS_dropout, FLAGS_bias);
    model->to(device);
    torch::optim::AdamOptions lr(FLAGS_lr);
    torch::optim::Adam optimizer(model->parameters(), lr);

    Shuffler shuffler(dataset->GetTrainSet().ids, dataset->GetTrainSet().len, FLAGS_batch_size);
    for (int epoch = 0; epoch < FLAGS_num_epoch; epoch++) {
        // model->train();
        shuffler.Shuffle();

        for (int step = 0; shuffler.HasNext(); step++) {
            NodesBatch batch = shuffler.GetNextBatch();
            std::vector<std::shared_ptr<Block>> blocks = SampleMultiHops(dataset, batch, task);
            TorchBlocks thblocks = CPUBlocksToDevice(blocks, "cuda:0");

            torch::Tensor output = model->forward(thblocks);
            torch::Tensor loss = torch::nll_loss(output, thblocks.label);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            printf("epoch %d  step %d \n", epoch, step);
        }
    }

    gflags::ShutDownCommandLineFlags();
    return 0;
}
