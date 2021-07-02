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
    argparser.add_argument('--cache-policy', type=int, default=1)
    argparser.add_argument('--cache-percentage', type=float, default=0)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--max-sampling-jobs', type=int, default=10)
    argparser.add_argument('--max-copying-jobs', type=int, default=10)

    argparser.add_argument('--num-epoch', type=int, default=10)
    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=[5, 10, 15])
    argparser.add_argument('--batch-size', type=int, default=8000)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

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
    run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # run_config['dataset_path'] = '/graph-learning/samgraph/reddit'
    # run_config['dataset_path'] = '/graph-learning/samgraph/products'
    # run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    run_config['cache_policy'] = sam.kCacheByHeuristic
    run_config['cache_percentage'] = 0.3

    run_config['max_sampling_jobs'] = 10
    # default max_copying_jobs should be 10, but when training on com-friendster,
    # we have to set this to 1 to prevent GPU out-of-memory
    run_config['max_copying_jobs'] = 1

    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']

    run_config['fanout'] = [5, 10, 15]
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    # we use the average result of 10 epochs, the first epoch is used to warm up the system
    run_config['num_epoch'] = 11
    run_config['batch_size'] = 8000
    run_config['num_hidden'] = 256
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5

    # arch1 doesn't support pipelining
    if run_config['arch_type'] == sam.kArch1:
        run_config['pipeline'] = False

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

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_train_times = []
    epoch_total_times = []

    sample_times = []
    copy_times = []
    convert_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    for epoch in range(num_epoch):
        for step in range(num_step):
            t0 = time.time()
            if not run_config['pipeline']:
                sam.sample_once()
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

            sample_time = sam.get_log_step_value(
                epoch, step, sam.kLogL1SampleTime)
            copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = (t3 - t2) + (t2 - t1)
            total_time = t3 - t0

            sam.log_step(epoch, step, sam.kLogL1TrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)

            sample_times.append(sample_time)
            copy_times.append(copy_time)
            convert_times.append(convert_time)
            train_times.append(train_time)
            total_times.append(total_time)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)
            num_nodes.append(blocks[0].num_src_nodes())

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(
                    sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), np.mean(convert_times[1:]), loss
            ))

            sam.report_step_average(epoch, step)

        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime))
        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_total_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]),  np.mean(epoch_train_times[1:])))

    sam.report_node_access()
    sam.shutdown()


if __name__ == '__main__':
    run()
