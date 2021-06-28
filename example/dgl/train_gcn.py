"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import fastgraph
import time
import numpy as np


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h


def get_run_config():
    run_config = {}
    run_config['pipeline'] = False
    run_config['device'] = 'cuda:0'
    # run_config['dataset'] = 'reddit'
    # run_config['dataset'] = 'products'
    # run_config['dataset'] = 'papers100M'
    run_config['dataset'] = 'com-friendster'
    run_config['root_path'] = '/graph-learning/samgraph/'

    run_config['fanout'] = [5, 10, 15]
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_epoch'] = 10
    run_config['num_hidden'] = 256
    run_config['batch_size'] = 8000
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['report_per_count'] = 1

    return run_config


def run():
    run_config = get_run_config()
    device = torch.device(run_config['device'])

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'])
    g = dataset.to_dgl_graph()
    train_nids = dataset.train_set
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    sampler = dgl.dataloading.MultiLayerNeighborSampler(run_config['fanout'])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nids,
        sampler,
        batch_size=run_config['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=0)

    model = GCN(in_feats, run_config['num_hidden'],
                n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])
    num_epoch = run_config['num_epoch']

    model.train()

    epoch_avg_sample_time = 0.0
    epoch_avg_copy_time = 0.0
    epoch_avg_train_time = 0.0
    epoch_avg_total_time = 0.0

    sample_times = []
    copy_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    for epoch in range(num_epoch):
        t0 = time.time()
        for step, (_, _, blocks) in enumerate(dataloader):
            t1 = time.time()
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            epoch_avg_sample_time += (t1 - t0)
            epoch_avg_copy_time += (t2 - t1)
            epoch_avg_train_time += (t3 - t2)
            epoch_avg_total_time += (t3 - t0)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)
            num_nodes.append(blocks[0].num_src_nodes())

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times), np.mean(sample_times), np.mean(copy_times), np.mean(train_times), loss))
            t0 = time.time()

    epoch_avg_sample_time /= num_epoch
    epoch_avg_copy_time /= num_epoch
    epoch_avg_train_time /= num_epoch
    epoch_avg_total_time /= num_epoch

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        epoch_avg_total_time, epoch_avg_sample_time, epoch_avg_copy_time, epoch_avg_train_time))


if __name__ == '__main__':
    run()
