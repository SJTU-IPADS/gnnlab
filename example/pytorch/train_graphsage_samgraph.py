import torch.nn as nn
import torch.nn.functional as F
import argparse

from samgraph.torch.nn import SageConv

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
        self.layers.append(SageConv(in_feats, n_hidden))
        for _ in range(1, n_layers - 1):
            self.layers.append(SageConv(n_hidden, n_hidden))
        self.layers.append(SageConv(n_hidden, n_classes))
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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--train-device', type=int, default=0,
                           help="")
    argparser.add_argument('--sample-device', type=int, default=1)
    argparser.add_argument('--dataset-path', type=str, default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    args = argparser.parse_args()

    