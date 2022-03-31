import argparse
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor, sum as sparsesum, mul

from common_config import *

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation, dropout, norm=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=False))
        # hidden layers
        for _ in range(1, num_layers - 1):
            self.layers.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        # output layer
        self.layers.append(
            GCNConv(hidden_channels, out_channels, add_self_loops=False, normalize=False))
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adjs):
        for i, (edge_index, _, _) in enumerate(adjs):
            if i != 0:
                x = self.dropout(x)
            if norm:
                edge_index = self.gcn_norm(edge_index)
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
        return x

    def gcn_norm(self, edge_index):
        assert(isinstance(edge_index, SparseTensor))
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))

        rowptr, col, value = adj_t.csc()
        transpose_adj_t = SparseTensor(
            rowptr=rowptr, col=col, value=value, is_sorted=True)
        tranpose_deg = sparsesum(transpose_adj_t, dim=1)
        tranpose_deg_inv_sqrt = tranpose_deg.pow_(-0.5)
        tranpose_deg_inv_sqrt.masked_fill_(
            tranpose_deg_inv_sqrt == float('inf'), 0.)

        adj_t = mul(adj_t, tranpose_deg_inv_sqrt.view(1, -1))

        return adj_t


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--device', type=str,
                           default=default_run_config['device'])
    argparser.add_argument('--dataset', type=str,
                           default=default_run_config['dataset'])
    argparser.add_argument('--root-path', type=str,
                           default='/graph-learning/samgraph/')
    argparser.add_argument('--pipelining', action='store_true',
                           default=default_run_config['pipelining'])
    argparser.add_argument(
        '--no-pipelining', dest='pipelining', action='store_false')
    argparser.add_argument('--num-sampling-worker', type=int,
                           default=default_run_config['num_sampling_worker'])

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--num-epoch', type=int,
                           default=default_run_config['num_epoch'])
    argparser.add_argument('--num-hidden', type=int,
                           default=default_run_config['num_hidden'])
    argparser.add_argument('--batch-size', type=int,
                           default=default_run_config['batch_size'])
    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])

    argparser.add_argument('--validate-configs',
                           action='store_true', default=False)

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['device'] = 'cuda:1'
    default_run_config['dataset'] = 'reddit'
    # default_run_config['dataset'] = 'products'
    # default_run_config['dataset'] = 'papers100M'
    # default_run_config['dataset'] = 'com-friendster'
    default_run_config['root_path'] = '/graph-learning/samgraph/'
    default_run_config['pipelining'] = False
    default_run_config['num_sampling_worker'] = 0

    # In PyG, the order from root to leaf is from front to end
    default_run_config['fanout'] = [15, 10, 5]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.01
    default_run_config['dropout'] = 0.5
    default_run_config['weight_decay'] = 0.0005

    run_config = parse_args(default_run_config)

    assert(run_config['num_sampling_worker'] >= 0)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'], force_load64=True)
    num_train_set = dataset.train_set.shape[0]

    # [prefetch_factor]: number of samples loaded in advance by each worker.
    # 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)
    # DGL uses a custom dataset, it makes PyTorch thinks a batch is a sample.
    if run_config['pipelining'] == False and run_config['num_sampling_worker'] > 0:
        # make it sequential. sample all the batch before training.
        # assumed that drop last = False
        num_samples_per_epoch = num_train_set
        num_batch_per_epoch = math.ceil(
            num_samples_per_epoch / run_config['batch_size'])
        run_config['num_prefetch_batch'] = num_batch_per_epoch
        run_config['prefetch_factor'] = math.ceil(
            num_batch_per_epoch / run_config['num_sampling_worker'])
    else:
        # default prefetch factor is 2
        run_config['prefetch_factor'] = 2

    run_config['seed'] = 0

    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        print('config:{:}={:}'.format(k, v))

    g = dataset.to_pyg_graph()
    run_config['dataset'] = dataset
    run_config['g'] = g

    if run_config['validate_configs']:
        sys.exit()

    return run_config

def get_data_iterator(run_config, dataloader):
    if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
        return [data for data in iter(dataloader)]
    else:
        return iter(dataloader)

def run():
    run_config = get_run_config()
    device = torch.device(run_config['device'])

    dataset = run_config['dataset']
    g = run_config['g']
    feat = dataset.feat
    label = dataset.label
    train_nids = dataset.train_set
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    dataloader = NeighborSampler(g,
                                 sizes=run_config['fanout'],
                                 batch_size=run_config['batch_size'],
                                 node_idx=train_nids,
                                 shuffle=True,
                                 drop_last=False,
                                 num_workers=run_config['num_sampling_worker'],
                                 return_e_id=False,
                                 prefetch_factor=run_config['prefetch_factor']
                                 )

    model = GCN(in_feats, run_config['num_hidden'], n_classes,
                run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])
    num_epoch = run_config['num_epoch']

    model.train()

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_train_times = []
    epoch_total_times = []

    sample_times = []
    copy_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    tic = time.time()

    for epoch in range(num_epoch):
        epoch_sample_time = 0.0
        epoch_copy_time = 0.0
        epoch_train_time = 0.0

        t0 = time.time()
        for step, (batch_size, n_id, adjs) in enumerate(get_data_iterator(run_config, dataloader)):
            t1 = time.time()
            adjs = [adj.to(device) for adj in adjs]
            batch_inputs = feat[n_id].to(device)
            batch_labels = label[n_id[:batch_size]].to(device)
            if not run_config['pipelining']:
                event_sync()
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(batch_inputs, adjs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not run_config['pipelining']:
                event_sync()

            num_samples.append(sum([adj.adj_t.nnz() for adj in adjs]))
            num_nodes.append(batch_inputs.shape[0])
            batch_inputs = None
            batch_labels = None
            adjs = None

            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            epoch_sample_time += (t1 - t0)
            epoch_copy_time += (t2 - t1)
            epoch_train_time += (t3 - t2)

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()

        event_sync()

        toc = time.time()

        epoch_sample_times.append(epoch_sample_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(toc - tic)

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]), np.mean(epoch_sample_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:])))


if __name__ == '__main__':
    run()
