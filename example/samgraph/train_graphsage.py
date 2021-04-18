import argparse
import time
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

import samgraph.torch as sam


def to_dgl_blocks(batch_key, num_layers):
    feat = sam.get_graph_feat(batch_key)
    label = sam.get_graph_label(batch_key)
    blocks = []
    for i in range(num_layers):
        row = sam.get_graph_row(batch_key, i)
        col = sam.get_graph_col(batch_key, i)
        blocks.append(dgl.create_block({('_U', '_V', '_U'): (row, col)}))

    return blocks, feat, label


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


def run(args):
    fanout_list = [int(fanout) for fanout in args.fan_out.split(',')]

    sam.init(args.dataset_path, args.sample_device, args.train_device,
             args.batch_size, fanout_list, args.num_epoch)

    th_train_device = th.device('cuda:%d' % args.train_device)

    in_feat = sam.dataset_num_feat_dim()
    num_class = sam.dataset_num_class()
    num_layer = len(fanout_list)

    model = SAGE(in_feat, args.num_hidden, num_class,
                 num_layer, F.relu, args.dropout)
    model = model.to(th_train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn.to(th_train_device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epoch = sam.num_epoch()
    num_step = sam.num_step_per_epoch()
    num_graph = len(fanout_list)

    # sam.start()

    model.train()
    for epoch in range(num_epoch):

        for step in range(num_step):
            t0 = time.time()
            sam.sample()
            graph_batch = sam.get_next_batch(epoch, step, num_graph)
            blocks, batch_input, batch_label = to_dgl_blocks(
                graph_batch.key, num_layer)
            t1 = time.time()

            t2 = time.time()
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Sample: {:.4f} secs | Train: {:.4f} secs | Time {:.4f} secs'.format(
                epoch, step, loss.item(), t1 - t0, t3 - t2, t3 - t0
            ))

            sam.profiler_report(epoch, step)

    sam.shutdown()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--train-device', type=int, default=0,
                           help="")
    argparser.add_argument('--sample-device', type=int, default=-1)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--num-epoch', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    args = argparser.parse_args()
    run(args)
