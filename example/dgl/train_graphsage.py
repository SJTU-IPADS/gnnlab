import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
    g =  dgl.data.CoraFullDataset()[0]
    
    train_nids = torch.arange(10000)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 4, 3])

    device = torch.device('cuda:0')
    in_feats = g.ndata['feat'].shape[1]
    n_classes = torch.max(g.ndata['label']).item() + 1

    model = SAGE(in_feats, 16, n_classes, 3, F.relu, 0.5)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nids,
        sampler,
        batch_size=1000,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    
    for _, (_, _, blocks) in enumerate(dataloader):
        blocks = [block.int().to('cuda:0') for block in blocks]
        batch_inputs = blocks[0].srcdata['feat']
        batch_labels = blocks[-1].dstdata['label']

        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    run()