from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary PyTorch C types.
import torch

from samgraph.torch import c_lib
from samgraph.common import SamGraphBasics as _SamGraphBasics
_basics = _SamGraphBasics(__file__, 'c_lib')

init = _basics.init
start = _basics.start
dataset_num_class = _basics.dataset_num_class
dataset_num_feat_dim = _basics.dataset_num_feat_dim
num_epoch = _basics.num_epoch
num_step_per_epoch = _basics.num_step_per_epoch
get_next_batch = _basics.get_next_batch
shutdown = _basics.shutdown

def _check_tensor(tensor):
    if not tensor.dtype == torch.float32:
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')

def _csrmm(graph_key, tensor):
    _check_tensor(tensor)
    return c_lib.samgraph_torch_csrmm(graph_key, tensor)

def _csrmm_transpose(graph_key, tensor):
    _check_tensor(tensor)
    return c_lib.samgraph_torch_csrmm_transpose(graph_key, tensor)

class SamGraphCsrmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph_key, tensor_input):
        ctx.key = graph_key
        return _csrmm(graph_key, tensor_input)

    @staticmethod
    def backward(ctx, grad_output):
        graph_key = ctx.graph_key
        return None, None, _csrmm_transpose(graph_key, grad_output)

def csrmm(graph_key, tensor):
    ouput = SamGraphCsrmm.apply(graph_key, tensor)
    return ouput

def get_graph_feat(batch_key):
    return c_lib.samgraph_torch_get_graph_feat(batch_key)
    
def get_graph_label(batch_key):
    return c_lib.samgraph_torch_get_graph_label(batch_key)