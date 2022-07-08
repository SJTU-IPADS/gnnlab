"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary PyTorch C types.
import dgl
import torch
from dgl.heterograph import DGLBlock

from samgraph.common import *
from samgraph.common import _basics
from samgraph.torch import c_lib

config               = _basics.config
init                 = _basics.init
start                = _basics.start
num_class            = _basics.num_class
feat_dim             = _basics.feat_dim
num_epoch            = _basics.num_epoch
steps_per_epoch      = _basics.steps_per_epoch
get_next_batch       = _basics.get_next_batch
get_graph_num_src    = _basics.get_graph_num_src
get_graph_num_dst    = _basics.get_graph_num_dst
shutdown             = _basics.shutdown
sample_once          = _basics.sample_once
log_step             = _basics.log_step
log_step_add         = _basics.log_step_add
log_epoch_add        = _basics.log_epoch_add
get_log_init_value   = _basics.get_log_init_value
get_log_step_value   = _basics.get_log_step_value
get_log_epoch_value  = _basics.get_log_epoch_value
report_init          = _basics.report_init
report_step          = _basics.report_step
report_step_average  = _basics.report_step_average
report_epoch         = _basics.report_epoch
report_epoch_average = _basics.report_epoch_average
report_node_access   = _basics.report_node_access
trace_step_begin     = _basics.trace_step_begin
trace_step_end       = _basics.trace_step_end
trace_step_begin_now = _basics.trace_step_begin_now
trace_step_end_now   = _basics.trace_step_end_now
dump_trace           = _basics.dump_trace
forward_barrier      = _basics.forward_barrier
wait_one_child       = _basics.wait_one_child
log_step_by_key      = _basics.log_step_by_key
get_log_step_value_by_key   = _basics.get_log_step_value_by_key

# switch APIs
switch_init   = _basics.switch_init

# multi-GPUs training APIs
data_init      = _basics.data_init
sample_init    = _basics.sample_init
um_sample_init = _basics.um_sample_init
train_init     = _basics.train_init
extract_start  = _basics.extract_start
num_local_step = _basics.num_local_step

def get_graph_feat(batch_key):
    batch_feat = c_lib.samgraph_torch_get_graph_feat(batch_key)
    if batch_feat.dtype != torch.float32:
        batch_feat = batch_feat.float()
    return batch_feat


def get_graph_label(batch_key):
    return c_lib.samgraph_torch_get_graph_label(batch_key)


def get_graph_row(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_row(batch_key, layer_idx)


def get_graph_col(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_col(batch_key, layer_idx)


def get_graph_data(batch_key, layer_idx):
    return c_lib.samgraph_torch_get_graph_data(batch_key, layer_idx)


def _create_dgl_block(data, num_src_nodes, num_dst_nodes):
    row, col = data
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, ['coo', 'csr', 'csc'])
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])

    return g

def get_dgl_blocks(batch_key, num_layers, with_feat=True):
    if with_feat:
        feat = get_graph_feat(batch_key)
        label = get_graph_label(batch_key)
    else:
        feat = None
        label = None
    blocks = []
    for i in range(num_layers):
        row = get_graph_row(batch_key, i)
        col = get_graph_col(batch_key, i)
        num_src_nodes = get_graph_num_src(batch_key, i)
        num_dst_nodes = get_graph_num_dst(batch_key, i)

        blocks.append(_create_dgl_block((row, col), num_src_nodes, num_dst_nodes))

        # blocks.append(dgl.create_block({('_N', '_E', '_N'): (
        #     row, col)}, num_src_nodes={'_N': num_src_nodes}, num_dst_nodes={'_N': num_dst_nodes}))

    return blocks, feat, label


def get_dgl_blocks_with_weights(batch_key, num_layers, with_feat=True):
    if with_feat:
        feat = get_graph_feat(batch_key)
        label = get_graph_label(batch_key)
    else:
        feat = None
        label = None
    blocks = []
    for i in range(num_layers):
        row = get_graph_row(batch_key, i)
        col = get_graph_col(batch_key, i)
        weights = get_graph_data(batch_key, i)
        num_src_nodes = get_graph_num_src(batch_key, i)
        num_dst_nodes = get_graph_num_dst(batch_key, i)

        # block = dgl.create_block({('_N', '_E', '_N'): (
        #     row, col)}, num_src_nodes={'_N': num_src_nodes}, num_dst_nodes={'_N': num_dst_nodes})

        block = _create_dgl_block((row, col), num_src_nodes, num_dst_nodes)
        block.edata['weights'] = weights
        blocks.append(block)

    return blocks, feat, label


def notify_sampler_ready(barrier):
    barrier.wait()


def wait_for_sampler_ready(barrier):
    barrier.wait()


def get_dataset_feat():
    return c_lib.samgraph_torch_get_dataset_feat()


def get_dataset_label():
    return c_lib.samgraph_torch_get_dataset_label()


def get_graph_input_nodes(batch_key):
    return c_lib.samgraph_torch_get_graph_input_nodes(batch_key)


def get_graph_output_nodes(batch_key):
    return c_lib.samgraph_torch_get_graph_output_nodes(batch_key)


def load_subtensor(batch_key, feat, label, device):
    input_nodes = get_graph_input_nodes(batch_key).to(feat.device)
    output_nodes = get_graph_output_nodes(batch_key).to(label.device)

    batch_inputs = torch.index_select(
        feat, 0, input_nodes.long()).to(device, dtype=torch.float32)
    batch_labels = torch.index_select(
        label, 0, output_nodes.long()).to(device)

    return batch_inputs, batch_labels
