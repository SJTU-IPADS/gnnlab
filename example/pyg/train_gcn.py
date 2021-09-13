import argparse
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import NeighborSampler
from torch_sparse import SparseTensor, sum as sparsesum, mul


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

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['device'] = 'cuda:1'
    default_run_config['dataset'] = 'reddit'
    # default_run_config['dataset'] = 'products'
    # default_run_config['dataset'] = 'papers100M'
    # default_run_config['dataset'] = 'com-friendster'
    default_run_config['root_path'] = '/graph-learning/samgraph/'
    default_run_config['num_sampling_worker'] = 16

    # In PyG, the order from root to leaf is from front to end
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5

    run_config = parse_args(default_run_config)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print('Evaluation time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(*run_config.items(), sep='\n')

    return run_config


def run():
    run_config = get_run_config()
    device = torch.device(run_config['device'])

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'], force_load64=True)
    g = dataset.to_pyg_graph()
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
                                 num_workers=run_config['num_sampling_worker'],
                                 return_e_id=False,
                                 #  prefetch_factor=run_config['num_epoch']
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

    for epoch in range(num_epoch):
        epoch_sample_time = 0.0
        epoch_copy_time = 0.0
        epoch_train_time = 0.0
        epoch_total_time = 0.0

        t0 = time.time()
        for step, (batch_size, n_id, adjs) in enumerate(dataloader):
            t1 = time.time()
            adjs = [adj.to(device) for adj in adjs]
            batch_inputs = feat[n_id].to(device)
            batch_labels = label[n_id[:batch_size]].to(device)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(batch_inputs, adjs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            epoch_sample_time += (t1 - t0)
            epoch_copy_time += (t2 - t1)
            epoch_train_time += (t3 - t2)
            epoch_total_time += (t3 - t0)

            num_sample = 0
            for adj in adjs:
                num_sample += adj.adj_t.nnz()
            num_samples.append(num_sample)
            num_nodes.append(batch_inputs.shape[0])

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()

        epoch_sample_times.append(epoch_sample_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(epoch_total_time)

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]), np.mean(epoch_sample_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:])))


if __name__ == '__main__':
    run()
