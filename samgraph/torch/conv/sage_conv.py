from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from samgraph.torch.ops import csrmm

class SageConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SageConv, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_self = nn.Linear(self.in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self.in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def forward(self, graph, feat):
        feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst[:graph.num_row,:]
        h_neigh = csrmm(graph.key, feat_src)

        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

        if self.activation is not None:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        
        return rst