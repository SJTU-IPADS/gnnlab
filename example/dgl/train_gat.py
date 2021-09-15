import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GATConv
import time
import fastgraph
import dgl
import torch.nn.functional as F
import numpy as np


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, blocks, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](blocks[l], h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        return logits


def get_run_config():
    run_config = {}
    run_config['device'] = 'cuda:0'
    # run_config['dataset'] = 'reddit'
    # run_config['dataset'] = 'products'
    run_config['dataset'] = 'papers100M'
    # run_config['dataset'] = 'com-friendster'
    run_config['root_path'] = '/graph-learning/samgraph/'

    run_config['fanout'] = [10, 5]
    run_config['num_fanout'] = run_config['num_layers'] = len(
        run_config['fanout'])
    run_config['num_epoch'] = 2
    run_config['num_heads'] = 8
    run_config['num_out_heads'] = 1
    run_config['num_hidden'] = 32
    run_config['residual'] = False
    run_config['in_drop'] = 0.6
    run_config['attn-drop'] = 0.6
    run_config['lr'] = 0.005
    run_config['weight_decay'] = 5e-4
    run_config['negative_slope'] = 0.2
    run_config['batch_size'] = 8000

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
        drop_last=True,
        num_workers=0)

    heads = ([run_config['num_heads']] * run_config['num_layers']) + \
        [run_config['num_out_heads']]
    model = GAT(run_config['num_layers'], in_feats, run_config['num_hidden'], n_classes, heads, F.elu,
                run_config['in_drop'], run_config['attn-drop'], run_config['negative_slope'], run_config['residual'])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(
    ), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
    num_epoch = run_config['num_epoch']

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
            print(batch_inputs.shape)
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

            num_samples.append(sum([block.num_edges() for block in blocks]))

            print('Epoch {:05d} | Step {:05d} | Samples {:f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, np.mean(num_samples[1:]), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()


if __name__ == '__main__':
    run()
