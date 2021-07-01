import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
import numpy as np


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def get_run_config():
    run_config = {}
    run_config['device'] = 'cuda:0'
    # run_config['dataset'] = 'reddit'
    # run_config['dataset'] = 'products'
    # run_config['dataset'] = 'papers100M'
    run_config['dataset'] = 'com-friendster'
    run_config['root_path'] = '/graph-learning/samgraph/'

    run_config['fanout'] = [5, 10, 15]
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    # we use the average result of 10 epochs, the first epoch is used to warm up the system
    run_config['num_epoch'] = 11
    run_config['num_hidden'] = 256
    run_config['batch_size'] = 8000
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5

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

    model = SAGE(in_feats, run_config['num_hidden'], n_classes,
                 run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])
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

            epoch_sample_time += (t1 - t0)
            epoch_copy_time += (t2 - t1)
            epoch_train_time += (t3 - t2)
            epoch_total_time += (t3 - t0)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)
            num_nodes.append(blocks[0].num_src_nodes())

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
