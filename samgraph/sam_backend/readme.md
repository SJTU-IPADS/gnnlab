# Sam Backend

A tiny GNN trainer for samgraph.

## General Architecture

A `Model` storing `GnnOp`s in a vector. Each `GnnOp` contains one or more input tensor and single output tensor.
One `GnnOp`'s input tensor comes from another `GnnOp`'s output, which forms the computation graph.
=
Each tensor is actually a `GradTensor`, which stores the original tensor along with its corresponding gradient.

## Lifetime of Tensors

For convenient, tensors are preserved across batches, i.e. each `GnnOp` stores output tensor in the same location.
Once a batch is obtained, the size of each layer is known, and tensors are re-allocated(`GradTensor::Resize`) if current size does not fit.

### Gradient accumulation

Normally, the compuation graph should be sequential, so each operation would overwrite input's gradient by default. However, gnn like GraphSAGE would have one tensor consumed by multiple ops (self_linear and scatter_gather). In this case, the gradient of the input should be the accumulation of gradients calculated by these ops.

The general idea is to let the first op(in backward) to reset the grad and following ops to accumulate the grad.
In sam_backend, all ops has a member function `accumulate_grad()` to indicate that whether this op is able to accumulate grad. Note that not all GnnOps has this ability(to accumulate grad).
Once the model is build, one should call `model.assign_reset_task()` to check whether gradient can be correctly accumulated, and tells each op, whether to overwrite the gradient or accumulate the gradient.


## Usage

### Build Model

Call model's member function named by operations (e.g. linear, dropout) to build a model. An example of GraphSAGE:
```c++
sam_backend::Model * model = new sam_backend::Model(ctx);
sam_backend::GradTensorPtr cur_h = model->input_feat;

for (size_t idx = 0; idx < RunConfig::num_layer; idx++) {
  auto input_h = cur_h;
  auto self_linear_h = model->partial_lienar(input_h, dim_list[idx], dim_list[idx+1], [model, idx]()->IdType{return model->cur_task->graphs[idx]->num_dst;});
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
```

### Train the Model

To train the model, pass a `Task` to the model and call `forward`, `backward`, `update` sequentially. A `loss` member function can be used to optain loss and accuracy.

Note that `forward` will first prepare each op, e.g. check size of each tensor in case for larger input, then call each op's forward method.

## Notes on GnnOps

Most of ops only takes a tensor of $$num_v * h_dim$$ and produces another, like `relu` and `dropout`.
Some ops also takes the input graph, so we pass a `layer_idx` to tell it which layer to use, like `scatter_gather`, `indegree_norm`.

`partial_linear` is a special op that handles `self_linear` operation in GraphSAGE. `self_linear` requires the feature of dst nodes to multiply with weight, while features of dst nodes are always grouped at the beginning of feature tensor. To eliminate the overhead of extracting dst features, `partial_linear` enables model to do linear on first part of rows of the input feature. Since the amount of rows to use remains unknown until the batch is prepared, a lambda is requrired for `partial_linear` to know how many rows to use.
