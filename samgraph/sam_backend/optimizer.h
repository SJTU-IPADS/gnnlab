#pragma once
#include "common.h"
#include <map>
#include <vector>

namespace samgraph {
namespace sam_backend {
class Model;

class Optimizer {
 public:
  Optimizer(const Model *_model);
  virtual void next(void) = 0;
  virtual void update() = 0;
  const Model *model;
};

class AdamOptimizer : public Optimizer {
 public:
  AdamOptimizer(const Model *_model,
                TrainType lr = 0.001f, TrainType weight_decay = 0.0f, TrainType beta1 = 0.9f,
                TrainType beta2 = 0.999f, 
                TrainType epsilon = 1e-8);
  void next();
  void update();
  /** fixed: how the alpha, beta changes remains to be determined */
  const TrainType _lr;
  const TrainType _beta1, _beta2;
  const TrainType _weight_decay, _epsilon;
  TrainType _stepped_beta1, _stepped_beta2;
  std::vector<TensorPtr> v, m;
};

class NaiveOptimizer : public Optimizer {
 public:
  NaiveOptimizer(const Model *model, TrainType lr);
  void next();
  void update();
  const TrainType _lr;
};

} // namespace sam_backend
} // namespace samgraph