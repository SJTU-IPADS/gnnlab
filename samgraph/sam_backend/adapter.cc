#include "../common/engine.h"
#include "model.h"
#include "utils.h"
namespace samgraph {
namespace sam_backend {

extern "C" {
void samgraph_backend_init_model() {
  using common::Engine;
  using common::RunConfig;
  sam_backend::TrainType lr = RunConfig::lr;
  sam_backend::TrainType dropout = RunConfig::dropout;
  size_t num_hidden = RunConfig::hiddem_dim;

  LOG(INFO) << "lr is " << lr;
  LOG(INFO) << "dropout is " << dropout;
  LOG(INFO) << "hidden is " << num_hidden;

  std::vector<size_t> dim_list = {Engine::Get()->GetGraphDataset()->feat->Shape()[1]};
  for (size_t idx = 0; idx < RunConfig::num_layer - 1; idx++) {
    dim_list.push_back(num_hidden);
  }
  dim_list.push_back(Engine::Get()->GetGraphDataset()->num_class);

  sam_backend::Model::Create();
  sam_backend::Model *model = sam_backend::Model::Get();
  sam_backend::GradTensorPtr cur_h = model->input_feat;

  for (size_t idx = 0; idx < RunConfig::num_layer; idx++) {
    auto input_h = cur_h;
    auto self_linear_h =
        model->partial_lienar(input_h, dim_list[idx], dim_list[idx + 1], [model, idx]() -> IdType {
          return model->cur_task->graphs[idx]->num_dst;
        });
    cur_h = model->scatter_gather(cur_h, idx);
    cur_h = model->indegree_norm(cur_h, idx);
    cur_h = model->linear(cur_h, dim_list[idx], dim_list[idx + 1]);
    cur_h = model->add(cur_h, self_linear_h);
    cur_h = model->bias(cur_h, dim_list[idx + 1]);
    if (idx != RunConfig::num_layer - 1) {
      cur_h = model->relu(cur_h);
      cur_h = model->dropout(cur_h, dropout);
    }
  }

  model->softmax_cross_entropy(cur_h);
  model->adam_optimize(lr, 0);
  model->assign_reset_task();
}

void samgraph_backend_train_current_batch() {
  using common::Engine;
  TaskPtr task = Engine::Get()->GetGraphBatch();
  // sam_backend::check_nan_exist(task->input_feat);
  sam_backend::Model *model = sam_backend::Model::Get();
  model->forward(task);
  // sam_backend::check_nan_exist(model->ops.back()->output->data());
  float loss, accuracy;
  model->loss(loss, accuracy);
  model->backward();
  model->update();
}
}
} // namespace sam_backend
} // namespace samgraph