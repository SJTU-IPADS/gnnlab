import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time


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


def run():
    device = torch.device('cuda:0')

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

    model = SAGE(in_feats, 16, n_classes, 3, F.relu, 0.5)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(2):
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
            print('Epoch {:05d} | Step {:05d} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                epoch, step, t3 - t0, t1 - t0, t2 - t1, t3 - t2, loss))
            t0 = time.time()


if __name__ == '__main__':
    run()
