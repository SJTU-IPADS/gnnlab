import argparse
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import samgraph.torch as sam

class SAGE(nn.Module):
    def __init__(self,
                 in_feat,
                 n_hidden,
                 n_class,
                 n_layer,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layer
        self.n_hidden = n_hidden
        self.n_classes = n_class
        self.layers = nn.ModuleList()
        self.layers.append(sam.SageConv(in_feat, n_hidden))
        for _ in range(1, n_layer - 1):
            self.layers.append(sam.SageConv(n_hidden, n_hidden))
        self.layers.append(sam.SageConv(n_hidden, n_class))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph_batch, x):
        h = x
        for l, (layer, graph) in enumerate(zip(self.layers, graph_batch.graphs)):
            h = layer(graph, h)
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

    model = SAGE(in_feat, args.num_hidden, num_class, args.num_layer, F.relu, args.dropout)
    model = model.to(th_train_device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epoch = sam.num_epoch()
    num_step = sam.num_step_per_epoch()
    num_graph = len(fanout_list)

    sam.start()

    # for epoch in range(num_epoch):

    #     tic_step = time.time()
    #     for step in range(num_step):
    #         graph_batch = sam.get_next_batch(num_graph)
    #         batch_input = sam.get_graph_feat(graph_batch.key)
    #         batch_label = sam.get_graph_label(graph_batch.key)

    #         batch_pred = model(graph_batch, batch_input)
    #         loss = loss_fcn(batch_pred, batch_label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time {:.4f} secs'.format(
    #             epoch, step, loss.item(), time.time() - tic_step
    #         ))

    #         tic_step = time.time()
    
    sam.shutdown()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--train-device', type=int, default=0,
                           help="")
    argparser.add_argument('--sample-device', type=int, default=1)
    argparser.add_argument('--dataset-path', type=str, default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--num-epoch', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layer', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    args = argparser.parse_args()
    run(args)