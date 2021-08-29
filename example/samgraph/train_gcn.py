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


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")
    argparser.add_argument(
        '--arch', type=str, default=default_run_config['arch'])
    argparser.add_argument('--sample_type', type=int,
                           default=default_run_config['sample_type'])
    argparser.add_argument('--pipeline', action='store_true',
                           default=default_run_config['pipeline'])

    argparser.add_argument('--dataset-path', type=str,
                           default=default_run_config['dataset_path'])
    argparser.add_argument('--cache-policy', type=int,
                           default=default_run_config['cache_policy'])
    argparser.add_argument('--cache-percentage', type=float,
                           default=default_run_config['cache_percentage'])
    argparser.add_argument('--max-sampling-jobs', type=int,
                           default=default_run_config['max_sampling_jobs'])
    argparser.add_argument('--max-copying-jobs', type=int,
                           default=default_run_config['max_copying_jobs'])

    argparser.add_argument('--num-epoch', type=int,
                           default=default_run_config['num_epoch'])
    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--batch-size', type=int,
                           default=default_run_config['batch_size'])
    argparser.add_argument('--num-hidden', type=int,
                           default=default_run_config['num_hidden'])
    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['arch'] = 'arch3'
    default_run_config['sample_type'] = sam.kKHop0
    default_run_config['pipeline'] = False  # default value must be false
    default_run_config['dataset_path'] = '/graph-learning/samgraph/reddit'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/products'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    default_run_config['cache_policy'] = sam.kCacheByHeuristic
    default_run_config['cache_percentage'] = 0.3

    default_run_config['max_sampling_jobs'] = 10
    # default max_copying_jobs should be 10, but when training on com-friendster,
    # we have to set this to 1 to prevent GPU out-of-memory
    default_run_config['max_copying_jobs'] = 1

    # default_run_config['fanout'] = [5, 10, 15]
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['batch_size'] = 8000
    default_run_config['num_hidden'] = 256
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5
    default_run_config['weight_decay'] = 0.0005

    run_config = parse_args(default_run_config)

    print('Evaluation time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(*run_config.items(), sep='\n')

    run_config['arch'] = sam.meepo_archs[run_config['arch']]
    run_config['arch_type'] = run_config['arch']['arch_type']
    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    # arch1 doesn't support pipelining
    if run_config['arch_type'] == sam.kArch1:
        run_config['pipeline'] = False

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.config_khop(run_config)
    sam.init()

    sample_device = th.device('cuda:%d' % run_config['sampler_ctx'].device_id)
    train_device  = th.device('cuda:%d' % run_config['trainer_ctx'].device_id)
    print("sampling using {}".format(th.cuda.get_device_name(sample_device)))
    print("training using {}".format(th.cuda.get_device_name(train_device)))
    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = GCN(in_feat, run_config['num_hidden'], num_class,
                num_layer, F.relu, run_config['dropout'])
    model = model.to(train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(
    ), lr=run_config['lr'], weight_decay=run_config['weight_decay'])

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    model.train()

    epoch_sample_times  = [0 for i in range(num_epoch)]
    epoch_copy_times    = [0 for i in range(num_epoch)]
    epoch_convert_times = [0 for i in range(num_epoch)]
    epoch_train_times   = [0 for i in range(num_epoch)]
    epoch_total_times   = [0 for i in range(num_epoch)]

    # sample_times  = [0 for i in range(num_epoch * num_step)]
    # copy_times    = [0 for i in range(num_epoch * num_step)]
    # convert_times = [0 for i in range(num_epoch * num_step)]
    # train_times   = [0 for i in range(num_epoch * num_step)]
    # total_times   = [0 for i in range(num_epoch * num_step)]
    # num_nodes     = [0 for i in range(num_epoch * num_step)]
    # num_samples   = [0 for i in range(num_epoch * num_step)]

    cur_step_key = 0
    for epoch in range(num_epoch):
        for step in range(num_step):
            t0 = time.time()
            sam.trace_step_begin_now (epoch * num_step + step, sam.kL0Event_Train_Step)
            if not run_config['pipeline']:
                sam.sample_once()
            elif epoch + step == 0:
                sam.start()
            batch_key = sam.get_next_batch(epoch, step)
            t1 = time.time()
            sam.trace_step_begin_now (batch_key, sam.kL1Event_Convert)
            blocks, batch_input, batch_label = sam.get_dgl_blocks(
                batch_key, num_layer)
            t2 = time.time()
            sam.trace_step_end_now (batch_key, sam.kL1Event_Convert)

            # Compute loss and prediction
            sam.trace_step_begin_now (batch_key, sam.kL1Event_Train)
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sam.trace_step_end_now   (batch_key, sam.kL1Event_Train)
            t3 = time.time()
            sam.trace_step_end_now (epoch * num_step + step, sam.kL0Event_Train_Step)

            # sample_time = sam.get_log_step_value(epoch, step, sam.kLogL1SampleTime)
            # copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t0

            # num_node = sam.get_log_step_value(epoch, step, sam.kLogL1NumNode)
            # num_sample = sam.get_log_step_value(epoch, step, sam.kLogL1NumSample)

            sam.log_step(epoch, step, sam.kLogL1TrainTime,   train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime,   train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime,   total_time)

            # sample_times  [cur_step_key] = sample_time
            # copy_times    [cur_step_key] = copy_time
            # convert_times [cur_step_key] = convert_time
            # train_times   [cur_step_key] = train_time
            # total_times   [cur_step_key] = total_time

            # num_nodes     [cur_step_key] = num_node
            # num_samples   [cur_step_key] = num_sample

            # print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
            #     epoch, step, num_node, num_sample, total_time, 
            #         sample_time, copy_time, train_time, convert_time, loss
            # ))

            # sam.report_step_average(epoch, step)
            sam.report_step(epoch, step)
            cur_step_key += 1

        epoch_sample_times  [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        epoch_copy_times    [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime)
        epoch_convert_times [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime)
        epoch_train_times   [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime)
        epoch_total_times   [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime)
        sam.forward_barrier()

    sam.report_step_average(num_epoch - 1, num_step - 1)
    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

    sam.report_node_access()
    sam.dump_trace()
    sam.shutdown()


if __name__ == '__main__':
    run()
