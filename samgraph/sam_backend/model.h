#pragma once
#include "common.h"
#include "constants.h"
#include "../common/run_config.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <cusparse.h>
#include <string>
#include <vector>
#include <functional>
namespace samgraph {
namespace sam_backend {

class Initializer;
class Model;
class Optimizer;

class GnnOp {
 public:
  GnnOp(Model *model, GradTensorPtr input);
  GnnOp(Model *model, GradTensorPtr input1, GradTensorPtr input2);
  virtual ~GnnOp();
  // GnnOp(Model *model, const GradTensorPtr input1, const GradTensorPtr input2,
  //       const GradTensorPtr input3);
  virtual void forward() = 0;
  virtual void backward() = 0;
  virtual void prepare() = 0;
  // abilities of this gnn op
  virtual std::string name() const = 0;
  virtual bool generate_grad() const = 0;
  virtual bool accumulate_grad() const = 0; // reset grad is a must-have ability;
  constexpr inline bool reset_grad() const { return true; }
  // virtual void update(const Model& model) = 0;
 public:
  int numInputs;
  GradTensorPtr inputs[MAX_NUM_INPUTS];
  bool accumulate_input_grad[MAX_NUM_INPUTS];
  GradTensorPtr output;
  Model *_model;
  // IndexLauncher *fwdLauncher, *bwdLauncher, *gradLauncher;
};
// Perform sum aggregation
class ScatterGather : public GnnOp {
 public:
  ScatterGather(Model *model, const GradTensorPtr input, size_t layer_idx);
  ~ScatterGather();
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  // void update(const Model& model);


  cusparseSpMatDescr_t A_T; // from root to fanout
  TensorPtr row, col;

  size_t buffer_size = 0;
  void * buffer = nullptr;

  IdType layer_idx;
};

class InDegreeNorm : public GnnOp {
 public:
  InDegreeNorm(Model *model, const GradTensorPtr input, size_t layer_idx, NormMode norm_mode = kNormModeDirect);
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  // void update(const Model& model);
  IdType _layer_idx;
  NormMode _norm_mode;
};

class Linear : public GnnOp {
 public:
  Linear(Model *model, const GradTensorPtr input, IdType in_dim, IdType out_dim,
         Initializer *initializer);
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  // void update(const Model& model);
 public:
  // ActiMode activation;
  IdType in_dim, out_dim;
  GradTensorPtr weight;
};
class PartialLinear : public GnnOp {
 public:
  PartialLinear(Model *model, const GradTensorPtr input, IdType in_dim, IdType out_dim, std::function<IdType(void)> source_of_num, 
         Initializer *initializer);
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  // void update(const Model& model);
 public:
  // ActiMode activation;
  IdType in_dim, out_dim;
  GradTensorPtr weight;
  std::function<IdType(void)> source_of_num;
};

class Activation : public GnnOp {
 public:
  Activation(Model *model, const GradTensorPtr input, ActiMode _actiMode);
  ~Activation();
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;

 public:
  ActiMode actiMode;
  cudnnTensorDescriptor_t tensorDesc = nullptr;
  cudnnActivationDescriptor_t actiDesc = nullptr;
};


class Bias : public GnnOp {
 public:
  Bias(Model *model, const GradTensorPtr input, IdType dim);
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;

 public:
  GradTensorPtr _bias;
};

class Element : public GnnOp {
 public:
  Element(Model *model, const GradTensorPtr input0, const GradTensorPtr input1,
          ElementType _elementType);
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;

 public:
  ElementType elementType;
};

class Dropout : public GnnOp {
 public:
  Dropout(Model *model, const GradTensorPtr input, float rate, int seed);
  ~Dropout();
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  // void update(const Model& model);

 public:
  float rate;
  int seed;

  // dropout reserved space size, cannot change between forward and backward
  size_t space_size = 0;
  void *space = nullptr;
  size_t actual_space_size = 0;
  cudnnDropoutDescriptor_t dropout_desc = nullptr;
  cudnnTensorDescriptor_t tensor_desc = nullptr;

  // dropout random states
  void *dropout_states;
  size_t dropout_states_size;
};

class SoftmaxCrossEntropy : public GnnOp {
 public:
  SoftmaxCrossEntropy(Model *model, const GradTensorPtr logits);
  ~SoftmaxCrossEntropy();
  void forward();
  void backward();
  void prepare();
  std::string name() const ;
  bool generate_grad() const ;
  bool accumulate_grad() const ;
  //  void update(const Model& model);

  void loss(float & loss, float &accuracy);

 public:
  int epoch_num;
  cudnnTensorDescriptor_t tensor_desc;
};

// class NcclTask : public IndexLauncher {
//  public:
//   NcclTask(const Graph &graph, const IndexSpaceT<1> &domain, const ArgumentMap &arg_map);
// };

class Initializer {
 public:
  Initializer(void);
  virtual ~Initializer(void);
  virtual void init(const Model *model, const TensorPtr tensor) = 0;
};
class GlorotUniform : public Initializer {
 public:
  GlorotUniform(void);
  ~GlorotUniform(void);
  void init(const Model *model, const TensorPtr tensor);
};

class ZerosInitializer : public Initializer {
 public:
  ZerosInitializer(void);
  ~ZerosInitializer(void);
  void init(const Model *model, const TensorPtr tensor);
};
class ValInitializer : public Initializer {
 public:
  ValInitializer(TrainType val);
  ~ValInitializer(void);
  void init(const Model *model, const TensorPtr tensor);
  TrainType val;
};


class Model {
 public:
  Model(Context ctx);
  ~Model();
  /** model builder */
  GradTensorPtr mul(const GradTensorPtr _input1, const GradTensorPtr _input2);
  GradTensorPtr add(const GradTensorPtr _input1, const GradTensorPtr _input2);
  GradTensorPtr dropout(const GradTensorPtr _input, float rate, int seed = 0);
  GradTensorPtr scatter_gather(const GradTensorPtr _input, size_t layer_idx);
  void softmax_cross_entropy(const GradTensorPtr logits);
  GradTensorPtr indegree_norm(const GradTensorPtr _input, size_t layer_idx, NormMode norm_mode = kNormModeDirect);
  GradTensorPtr linear(const GradTensorPtr _input, int in_dim, int out_dim,
                       Initializer *initializer = NULL);
  GradTensorPtr bias(const GradTensorPtr _input, int dim);
  GradTensorPtr partial_lienar(const GradTensorPtr _input, int in_dim, int out_dim,
                               std::function<IdType(void)> source_of_num,
                               Initializer *initializer = NULL);
  GradTensorPtr relu(const GradTensorPtr _input);
  GradTensorPtr sigmoid(const GradTensorPtr _input);
  void adam_optimize(float lr, float weight_decay);
  void naive_optimize(float lr);

  inline void train_mode(void) { this->mode = KModeTrain; }
  inline void infer_mode(void) { this->mode = KModeInfer; }
  void assign_reset_task();

  // train progress
  void forward(TaskPtr TaskPtr);
  void loss(float &loss, float & accuracy);
  void backward(void);
  void update(void);
  void zero_gradients(void);

  static inline Model* Get() { return _singleton; }
  static void Create() { _singleton = new Model(common::RunConfig::trainer_ctx); }

 public:
  ModelMode mode;
  Context ctx;

  cudnnHandle_t dnn;
  cublasHandle_t blas;
  cusparseHandle_t sparse;
  cudaStream_t stream;

  Optimizer *optimizer;
  std::vector<GradTensorPtr> parameters;

  TaskPtr cur_task = nullptr;

  GradTensorPtr input_feat;

  TensorPtr ones;

  std::vector<GnnOp *> ops;
 private:
  static Model* _singleton;
};

} // namespace sam_backend
} // namespace samgraph