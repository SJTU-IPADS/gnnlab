#include <cstdio>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cusparse.h>
#include <torch/torch.h>
#include <gflags/gflags.h>

#include "data/dataset.hpp"
#include "sampling/cpu/sampler.hpp"
#include "sampling/cpu/block.hpp"
#include "sampling/cpu/shuffler.hpp"
#include "train/torch_block.hpp"
#include "util/macros.hpp"
#include "util/cusparse.hpp"
#include "util/performance.hpp"
#include "util/tictoc.hpp"

DEFINE_string(dataset_key, "papers100M","");
DEFINE_string(dataset_folder, "/graph-learning/preprocess/papers100M", "");
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
                                                  torch::autograd::Variable csr_meta,
                                                  torch::autograd::Variable csc_indptr,
                                                  torch::autograd::Variable csc_indices,
                                                  torch::autograd::Variable csc_val,
                                                  torch::autograd::Variable csc_meta) {
        ctx->save_for_backward({csc_indptr, csc_indices, csc_meta});
        int m = 0;
        int n = 0;
        int k = 0;
        int nnz = 0;
        const double alpha = 0;
        cusparseMatDescr_t descrA;
        const float *csrValA;
        const int *csrRowPtrA;
        const int  *csrColIndA;
        const float *B;
        int ldb;
        const float *beta;
        float *C;
        int ldc;
        
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseScsrmm(,
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       );
        return {};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
        auto saved = ctx->get_saved_variables();
        torch::autograd::Variable csc_indptr = saved[0];
        torch::autograd::Variable csc_indices = saved[1];
        torch::autograd::Variable csc_meta = saved[2];

        torch::autograd::Variable grad_input;

        return {grad_input, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

struct SAGEConv : torch::nn::Module {
    SAGEConv(int in_feats, int out_feats, bool bias) : in_feats(in_feats), out_feats(out_feats) {
        fc_self = register_module("fc_self", std::make_shared<torch::nn::Linear>(torch::nn::LinearOptions(in_feats, out_feats).bias(bias)));
        fc_neigh = register_module("fc_neigh", std::make_shared<torch::nn::Linear>(torch::nn::LinearOptions(in_feats, out_feats).bias(bias)));
        ResetParam();
    }

    torch::Tensor forward(TorchCSR &csr, TorchCSR &csc, const torch::Tensor &x) {
        torch::Tensor feat_src = x;
        torch::Tensor feat_dst = x.slice(in_feats, 0, csr.meta[1].item().toInt(), 1);
        
        torch::Tensor h_self = feat_dst;
        torch::Tensor h_neigh = SPMM::apply(feat_src, csr.indptr, csr.indices, csr.val, csr.meta, csc.indptr, csc.indices, csc.val, csc.meta)[0];
        
        torch::Tensor rst = fc_self->get()->forward(h_self) + fc_neigh->get()->forward(h_neigh);
        
        return rst;
    }

    void ResetParam() {
        double gain = torch::nn::init::calculate_gain(torch::kReLU);
        torch::nn::init::xavier_uniform_(fc_self->get()->weight, gain);
        torch::nn::init::xavier_uniform_(fc_neigh->get()->weight, gain);
    }

    std::shared_ptr<torch::nn::Linear> fc_self;
    std::shared_ptr<torch::nn::Linear> fc_neigh;
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

    torch::Device device("cuda:0");
    auto model = std::make_shared<SAGE>(0, 0, 0, 0, 0.0f, FLAGS_bias);
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
        }
    }

    gflags::ShutDownCommandLineFlags();
    return 0;
}
