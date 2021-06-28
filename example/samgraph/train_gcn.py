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
            GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
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
    argparser.add_argument('--arch', type=str, default='arch0')
    argparser.add_argument('--pipeline', action='store_true', default=False)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')

    argparser.add_argument('--num-epoch', type=int, default=10)
    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=[5, 10, 15])
    argparser.add_argument('--batch-size', type=int, default=8000)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--report-per-count', type=int, default=1)
    argparser.add_argument('--cache-percentage', type=float, default=0)

    run_config = vars(argparser.parse_args())
    run_config['arch'] = sam.meepo_archs[run_config['arch']]
    run_config['arch_type'] = run_config['arch']['arch_type']
    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']
    run_config['sample_type'] = sam.kKHop0

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    return run_config


def get_run_config():
    args_run_config = parse_args()
    if args_run_config['parse_args']:
        return args_run_config

    run_config = {}
    run_config['arch'] = sam.meepo_archs['arch3']
    run_config['arch_type'] = run_config['arch']['arch_type']
    run_config['sample_type'] = sam.kKHop0
    run_config['pipeline'] = True
    # run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    run_config['dataset_path'] = '/graph-learning/samgraph/reddit'
    # run_config['dataset_path'] = '/graph-learning/samgraph/products'
    # run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']

    run_config['fanout'] = [5, 10, 15]
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_epoch'] = 10
    run_config['batch_size'] = 8000
    run_config['num_hidden'] = 256
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['report_per_count'] = 1
    run_config['cache_percentage'] = 0.2

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

    epoch_avg_sample_time = 0.0
    epoch_avg_copy_time = 0.0
    epoch_avg_train_time = 0.0
    epoch_avg_total_time = 0.0

    sample_times = []
    convert_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    for epoch in range(num_epoch):
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

            epoch_avg_sample_time += (t1 - t0)
            epoch_avg_train_time += ((t3 - t2) + (t2 - t1))
            epoch_avg_total_time += (t3 - t0)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)
            num_nodes.append(blocks[0].num_src_nodes())

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} |  Samples {:.0f} | Time {:.4f} secs | Sample + Copy Time {:.4f} secs | Convert Time {:.4f} secs |  Train Time {:.4f} secs | Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times), np.mean(
                    sample_times), np.mean(convert_times), np.mean(train_times), loss
            ))
            if step % run_config['report_per_count'] == 0:
                sam.report(epoch, step)

    epoch_avg_sample_time /= num_epoch
    epoch_avg_copy_time /= num_epoch
    epoch_avg_train_time /= num_epoch
    epoch_avg_total_time /= num_epoch

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        epoch_avg_total_time, epoch_avg_sample_time, epoch_avg_copy_time, epoch_avg_train_time))

    sam.shutdown()


if __name__ == '__main__':
    run()
