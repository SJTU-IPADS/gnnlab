from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary PyTorch C types.
import dgl
import time

from samgraph.torch import c_lib
from samgraph.common import SamGraphBasics as _SamGraphBasics
from samgraph.common import cpu, gpu, simple_hashtable, parallel_hashtable
_basics = _SamGraphBasics(__file__, 'c_lib')

config = _basics.config
init = _basics.init
start = _basics.start
num_class = _basics.num_class
feat_dim = _basics.feat_dim
num_epoch = _basics.num_epoch
steps_per_epoch = _basics.steps_per_epoch
get_next_batch = _basics.get_next_batch
get_graph_num_row = _basics.get_graph_num_row
get_graph_num_col = _basics.get_graph_num_col
shutdown = _basics.shutdown
sample = _basics.sample
report = _basics.report
report_node_access = _basics.report_node_access


def get_graph_feat(batch_key):
    return c_lib.samgraph_torch_get_graph_feat(batch_key)


def get_graph_label(batch_key):
    return c_lib.samgraph_torch_get_graph_label(batch_key)


def get_graph_row(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_row(batch_key, layer_idx)


def get_graph_col(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_col(batch_key, layer_idx)


def get_dgl_blocks(batch_key, num_layers):
    feat = get_graph_feat(batch_key)
    label = get_graph_label(batch_key)
    blocks = []
    for i in range(num_layers):
        t0 = time.time()
        row = get_graph_row(batch_key, i)
        col = get_graph_col(batch_key, i)
        num_src_nodes = get_graph_num_row(batch_key, i)
        num_dst_nodes = get_graph_num_col(batch_key, i)

        t1 = time.time()

        blocks.append(dgl.create_block({('_U', '_V', '_U'): (
            row, col)}, num_src_nodes={'_U': num_src_nodes}, num_dst_nodes={'_U': num_dst_nodes}))

        t2 = time.time()

        print("get_dgl_block {:.4f} {:.4f}".format(t1 - t0, t2 - t1))

    return blocks, feat, label
