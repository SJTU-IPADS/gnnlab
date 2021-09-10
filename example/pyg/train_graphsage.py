import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.data import NeighborSampler


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation, dropout):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(1, num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


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
    default_run_config['pipelining'] = False  # default value must be false
    default_run_config['num_sampling_worker'] = 16

    default_run_config['fanout'] = [5, 10, 15]
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

    if run_config['pipelining'] == False:
        run_config['num_sampling_worker'] = 0

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

    dataloader = NeighborSampler(g, sizes=run_config['fanout'], batch_size=run_config['batch_size'], node_idx=train_nids,
                                 shuffle=True, num_workers=run_config['num_sampling_worker'])

    model = SAGE(in_feats, run_config['num_hidden'], n_classes,
                 run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])
    num_epoch = run_config['num_epoch']

    for epoch in range(num_epoch):
        for step, (batch_size, n_id, adjs) in enumerate(dataloader):
            adjs = [adj.to(device) for adj in adjs]
            batch_inputs = feat[n_id].to(device)
            batch_labels = label[n_id[:batch_size]].to(device)
            batch_pred = model(batch_inputs, adjs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {:05d} | Step {:05d} | Loss {:.4f} '.format(
            epoch, step, loss))


if __name__ == '__main__':
    run()
