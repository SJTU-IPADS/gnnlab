import argparse
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv

import samgraph.torch as sam


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
            GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h


def parse_args():
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--parse-args', action='store_true', default=False)
    argparser.add_argument('--type', type=str, default='gpu')
    argparser.add_argument('--cpu-hashtable-type', type=int,
                           default=sam.simple_hashtable())
    argparser.add_argument('--pipeline', action='store_true', default=False)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')

    argparser.add_argument('--num-epoch', type=int, default=20)
    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=[15, 10, 5])
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--report-per-count', type=int, default=1)

    run_config = vars(argparser.parse_args())
    if run_config['type'] == 'cpu':
        run_config['sampler_ctx'] = sam.cpu()
        run_config['trainer_ctx'] = sam.gpu(0)
    else:
        run_config['sampler_ctx'] = sam.gpu(1)
        run_config['trainer_ctx'] = sam.gpu(0)

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    return run_config


def get_run_config():
    args_run_config = parse_args()
    if args_run_config['parse_args']:
        return args_run_config

    run_config = {}
    run_config['type'] = 'gpu'
    run_config['cpu_hashtable_type'] = sam.simple_hashtable()
    # run_config['cpu_hashtable_type'] = sam.parallel_hashtable()
    run_config['pipeline'] = False
    # run_config['pipeline'] = True
    run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    if run_config['type'] == 'cpu':
        run_config['sampler_ctx'] = sam.cpu()
        run_config['trainer_ctx'] = sam.gpu(0)
    else:
        run_config['sampler_ctx'] = sam.gpu(1)
        run_config['trainer_ctx'] = sam.gpu(0)

    run_config['fanout'] = [15, 10, 5]
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_epoch'] = 20
    run_config['batch_size'] = 8192
    run_config['num_hidden'] = 256
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['report_per_count'] = 1

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.init()

    train_device = th.device('cuda:%d' % run_config['trainer_ctx'].device_id)
    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = GCN(in_feat, run_config['num_hidden'], num_class,
                num_layer, F.relu, run_config['dropout'])
    model = model.to(train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    if run_config['pipeline']:
        sam.start()

    model.train()

    for epoch in range(num_epoch):
        sample_times = []
        convert_times = []
        train_times = []
        total_times = []
        num_samples = []
        for step in range(num_step):
            t0 = time.time()
            if not run_config['pipeline']:
                sam.sample()
            batch_key = sam.get_next_batch(epoch, step)
            t1 = time.time()
            blocks, batch_input, batch_label = sam.get_dgl_blocks(
                batch_key, num_layer)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            sample_times.append(t1 - t0)
            convert_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)

            print('Epoch {:05d} | Step {:05d} | Samples {:.0f} | Time {:.4f} secs | Sample + Copy Time {:.4f} secs | Convert Time {:.4f} secs |  Train Time {:.4f} secs | Loss {:.4f} '.format(
                epoch, step, np.mean(num_samples[1:]), np.mean(total_times[1:]), np.mean(
                    sample_times[1:]), np.mean(convert_times[1:]), np.mean(train_times[1:]), loss
            ))
            if step % run_config['report_per_count'] == 0:
                sam.report(epoch, step)
    sam.shutdown()


if __name__ == '__main__':
    run()
