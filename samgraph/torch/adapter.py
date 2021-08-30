from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary PyTorch C types.
import dgl
import time

from samgraph.torch import c_lib
from samgraph.common import *
_basics = SamGraphBasics(__file__, 'c_lib')

config = _basics.config
config_khop = _basics.config_khop
config_random_walk = _basics.config_random_walk
init = _basics.init
start = _basics.start
num_class = _basics.num_class
feat_dim = _basics.feat_dim
num_epoch = _basics.num_epoch
steps_per_epoch = _basics.steps_per_epoch
get_next_batch = _basics.get_next_batch
get_graph_num_src = _basics.get_graph_num_src
get_graph_num_dst = _basics.get_graph_num_dst
shutdown = _basics.shutdown
sample_once = _basics.sample_once
log_step = _basics.log_step
log_step_add = _basics.log_step_add
log_epoch_add = _basics.log_epoch_add
get_log_step_value = _basics.get_log_step_value
get_log_epoch_value = _basics.get_log_epoch_value
report_step = _basics.report_step
report_step_average = _basics.report_step_average
report_epoch = _basics.report_epoch
report_epoch_average = _basics.report_epoch_average
report_node_access = _basics.report_node_access

# mult-GPUs training APIs
data_init = _basics.data_init
sample_init = _basics.sample_init
train_init = _basics.train_init
extract_start = _basics.extract_start

def get_graph_feat(batch_key):
    return c_lib.samgraph_torch_get_graph_feat(batch_key)


def get_graph_label(batch_key):
    return c_lib.samgraph_torch_get_graph_label(batch_key)


def get_graph_row(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_row(batch_key, layer_idx)


def get_graph_col(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_col(batch_key, layer_idx)


def get_graph_data(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_data(batch_key, layer_idx)


def get_dgl_blocks(batch_key, num_layers):
    feat = get_graph_feat(batch_key)
    label = get_graph_label(batch_key)
    blocks = []
    for i in range(num_layers):
        # t0 = time.time()
        row = get_graph_row(batch_key, i)
        col = get_graph_col(batch_key, i)
        num_src_nodes = get_graph_num_src(batch_key, i)
        num_dst_nodes = get_graph_num_dst(batch_key, i)

        # t1 = time.time()

        blocks.append(dgl.create_block({('_U', '_V', '_U'): (
            row, col)}, num_src_nodes={'_U': num_src_nodes}, num_dst_nodes={'_U': num_dst_nodes}))

        # t2 = time.time()

        # print("get_dgl_block {:.4f} {:.4f}".format(t1 - t0, t2 - t1))

    return blocks, feat, label


def get_dgl_blocks_with_weights(batch_key, num_layers):
    feat = get_graph_feat(batch_key)
    label = get_graph_label(batch_key)
    blocks = []
    for i in range(num_layers):
        # t0 = time.time()
        row = get_graph_row(batch_key, i)
        col = get_graph_col(batch_key, i)
        weights = get_graph_data(batch_key, i)
        num_src_nodes = get_graph_num_src(batch_key, i)
        num_dst_nodes = get_graph_num_dst(batch_key, i)

        # t1 = time.time()
        block = dgl.create_block({('_U', '_V', '_U'): (
            row, col)}, num_src_nodes={'_U': num_src_nodes}, num_dst_nodes={'_U': num_dst_nodes})
        block.edata['weights'] = weights
        blocks.append(block)

        # t2 = time.time()

        # print("get_dgl_block {:.4f} {:.4f}".format(t1 - t0, t2 - t1))

    return blocks, feat, label
