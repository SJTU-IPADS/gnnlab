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
import dgl.nn.pytorch as dglnn
import fastgraph
import time
import numpy as np

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

def run():
    device  = torch.device('cuda:0')

    dataset = fastgraph.Papers100M('/graph-learning/samgraph/papers100M')
    g = dataset.to_dgl_graph()
    train_nids = dataset.train_set
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nids,
        sampler,
        batch_size=8192,
        shuffle=True,
        drop_last=True,
        num_workers=0)

    model = GCN(g, in_feats, 256, n_classes, 3, F.relu, 0.5)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    num_epoch = 20

    model.train()

    for epoch in range(num_epoch):
        sample_times = []
        copy_times = []
        train_times = []
        total_times = []
        num_samples = []

        t0 = time.time()
        for step, (_, _, blocks) in enumerate(dataloader):
            t1 = time.time()
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            num_samples = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)

            print('Epoch {:05d} | Step {:05d} | Samples {:f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, np.mean(num_samples[1:]), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()


if __name__ == '__main__':
    run()
