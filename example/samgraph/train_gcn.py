import argparse
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
from dgl.nn.pytorch import GraphConv

import samgraph.torch as sam
from common_config import *


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

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])
    argparser.add_argument('--unified-memory', action='store_true')
    argparser.add_argument('--unified-memory-percentage', type=float,
                            default=0)
    argparser.add_argument('--um-policy', type=str,
        choices=['default', 'degree', 'trainset', 'random', 'presample'],
        default='default')

    argparser.add_argument('--unified-memory-ctx', type=str, nargs='+', default=argparse.SUPPRESS)

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config())
    run_config['arch'] = 'arch3'
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [5, 10, 15]
    # run_config['fanout'] = [5, 10]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['weight_decay'] = 0.0005

    run_config.update(parse_args(run_config))

    # run_config["override_device"] = True
    # run_config["override_train_device"] = 'cuda:0'
    # run_config["override_sample_device"] = 'cuda:1'

    # run_config["dataset"] = 'reddit'
    # run_config["dataset"] = 'com-friendster'
    # run_config["dataset"] = 'ppi'
    # run_config["unified_memory"] = True

    process_common_config(run_config)
    assert(run_config['sample_type'] != 'random_walk')
    assert(run_config['unified_memory_percentage'] >= 0
           and run_config['unified_memory_percentage'] <= 1.0)

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.init()

    # sam.report_init()

    train_device = th.device(run_config['trainer_ctx'])

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

    print("before train")
    model.train()

    epoch_sample_times = [0 for i in range(num_epoch)]
    epoch_sample_coo_times = [0 for i in range(num_epoch)]
    epoch_sample_kernel_times = [0 for i in range(num_epoch)]
    epoch_copy_times = [0 for i in range(num_epoch)]
    epoch_convert_times = [0 for i in range(num_epoch)]
    epoch_train_times = [0 for i in range(num_epoch)]
    epoch_train_total_times_profiler = []
    epoch_total_times_profiler = [0 for i in range(num_epoch)]
    epoch_total_times_python = []
    epoch_cache_hit_rates = []
    epoch_miss_nbytes = []
    epoch_um_sample_nbytes = []
    epoch_um_sample_hit_nbytes = []
    epoch_um_hit_rate = []

    sample_times  = [0 for i in range(num_epoch * num_step)]
    # copy_times    = [0 for i in range(num_epoch * num_step)]
    core_sample_times = [0 for i in range(num_epoch * num_step)]
    fill_sample_input_times = [0 for i in range(num_epoch * num_step)]
    remap_times   = [0 for i in range(num_epoch * num_step)]
    # convert_times = [0 for i in range(num_epoch * num_step)]
    # train_times   = [0 for i in range(num_epoch * num_step)]
    # total_times   = [0 for i in range(num_epoch * num_step)]
    num_nodes     = [0 for i in range(num_epoch * num_step)]
    num_samples = [0 for i in range(num_epoch * num_step)]
    sample_coo_times = [0 for i in range(num_epoch * num_step)]
    sample_kernel_times = [0 for i in range(num_epoch * num_step)]
    sample_node_thpts = [0 for i in range(num_epoch * num_step)]
    sample_compact_edge_times = [0 for i in range(num_epoch * num_step)]

    cur_step_key = 0
    for epoch in range(num_epoch):
        tic = time.time()
        for step in range(num_step):
            t0 = time.time()
            sam.trace_step_begin_now(
                epoch * num_step + step, sam.kL0Event_Train_Step)
            if not run_config['pipeline']:
                sam.sample_once()
            elif epoch + step == 0:
                sam.start()
            batch_key = sam.get_next_batch()
            t1 = time.time()
            sam.trace_step_begin_now(batch_key, sam.kL1Event_Convert)
            blocks, batch_input, batch_label = sam.get_dgl_blocks(
                batch_key, num_layer)
            t2 = time.time()
            sam.trace_step_end_now(batch_key, sam.kL1Event_Convert)

            # Compute loss and prediction
            sam.trace_step_begin_now(batch_key, sam.kL1Event_Train)
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # wait for the train finish then we can free the data safely
            event_sync()

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()

            batch_input = None
            batch_label = None
            blocks = None

            sam.trace_step_end_now(batch_key, sam.kL1Event_Train)
            t3 = time.time()
            sam.trace_step_end_now(
                epoch * num_step + step, sam.kL0Event_Train_Step)

            sample_time = sam.get_log_step_value(epoch, step, sam.kLogL1SampleTime)
            core_sample_time = sam.get_log_step_value(epoch, step, sam.kLogL2CoreSampleTime)
            fill_sample_input_time = sam.get_log_step_value(epoch, step, sam.kLogL3RemapFillUniqueTime)
            remap_time = sam.get_log_step_value(epoch, step, sam.kLogL2IdRemapTime)
            # copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            sample_coo_time = sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleCooTime)
            sample_kernel_time = sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleKernelTime)
            sample_compact_edge_time = sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleCompactEdgesTime) + \
                sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleCountEdgeTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t0

            num_node = sam.get_log_step_value(epoch, step, sam.kLogL1NumNode)
            # num_sample = sam.get_log_step_value(epoch, step, sam.kLogL1NumSample)

            sam.log_step(epoch, step, sam.kLogL1TrainTime,   train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime,   train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime,   total_time)

            sample_times  [cur_step_key] = sample_time
            core_sample_times[cur_step_key] = core_sample_time
            fill_sample_input_times[cur_step_key] = fill_sample_input_time
            remap_times[cur_step_key] = remap_time
            # copy_times    [cur_step_key] = copy_time
            # convert_times [cur_step_key] = convert_time
            # train_times   [cur_step_key] = train_time
            # total_times   [cur_step_key] = total_time
            sample_coo_times[cur_step_key] = sample_coo_time
            sample_kernel_times[cur_step_key] = sample_kernel_time
            sample_compact_edge_times[cur_step_key] = sample_compact_edge_time

            # num_samples.append(num_sample)
            num_nodes     [cur_step_key] = num_node
            num_samples[cur_step_key] = num_sample

            sample_node_thpts[cur_step_key] = num_sample / sample_kernel_time
            # print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
            #     epoch, step, num_node, num_sample, total_time,
            #         sample_time, copy_time, train_time, convert_time, loss
            # ))

            # sam.report_step_average(epoch, step)
            # sam.report_step(epoch, step)
            cur_step_key += 1

        toc = time.time()
        epoch_um_sample_nbytes.append(sam.get_log_epoch_value(epoch, sam.kLogEpochUMSampleTotBytes))
        epoch_um_sample_hit_nbytes.append(sam.get_log_epoch_value(epoch, sam.kLogEpochUMSampleHitBytes))
        try:
            epoch_um_hit_rate.append(epoch_um_sample_hit_nbytes[-1] / epoch_um_sample_nbytes[-1])
        except ZeroDivisionError:
            epoch_um_hit_rate.append(np.NaN)
        feat_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochFeatureBytes)
        miss_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochMissBytes)
        epoch_miss_nbytes.append(miss_nbytes)
        epoch_cache_hit_rates.append(
            (feat_nbytes - miss_nbytes) / feat_nbytes)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        )
        epoch_total_times_python.append(toc - tic)
        epoch_sample_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochSampleTime)
        epoch_sample_coo_times[epoch] = sam.get_log_epoch_value(epoch, sam.kLogEpochSampleCooTime)
        epoch_sample_kernel_times[epoch] = sam.get_log_epoch_value(epoch, sam.kLogEpochSampleKernelTime)
        epoch_copy_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochCopyTime)
        epoch_convert_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochConvertTime)
        epoch_train_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochTrainTime)
        epoch_train_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        epoch_total_times_profiler[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochTotalTime)
        sam.forward_barrier()
        print('Epoch {:05d} | Time {:.4f}'.format(
            epoch, epoch_total_times_python[-1]))

    sam.report_step_average(num_epoch - 1, num_step - 1)
    print('[Avg] Epoch Time {:.4f} | Epoch Time(Profiler) {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times_python[1:]), np.mean(epoch_total_times_profiler[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

    test_result = []
    test_result.append(
        ('epoch_time:sample_time', np.mean(epoch_sample_times[1:])))
    test_result.append(('epoch_time:sample_coo_time', np.mean(epoch_sample_coo_times[1:])))
    test_result.append(('epoch_time:sample_kernel_time', np.mean(epoch_sample_kernel_times[1:])))
    test_result.append(('epoch_time:copy_time',
                    np.mean(epoch_copy_times[1:])))
    test_result.append(('convert_time', np.mean(epoch_convert_times[1:])))
    test_result.append(('train_time', np.mean(epoch_train_times[1:])))
    test_result.append(('epoch_time:train_total', np.mean(
            np.mean(epoch_train_times[1:]) + np.mean(epoch_convert_times[1:]))))
    test_result.append(
        ('cache_percentage', run_config['cache_percentage']))
    test_result.append(('cache_hit_rate', np.mean(
        epoch_cache_hit_rates[1:])))
    test_result.append(
        ('epoch_time:total', np.mean(epoch_total_times_python[1:])))
    test_result.append(('epoch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])))
    test_result.append(('batch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])/num_step))
    test_result.append(('batch_copy_time', np.mean(epoch_copy_times[1:])/num_step))
    test_result.append(('um_sample_hit_rate', np.mean(epoch_um_hit_rate[1:])))
    test_result.append(('num_nodes', np.mean(num_nodes[num_step:])))
    test_result.append(('num_samples', np.mean(num_samples[num_step:])))
    test_result.append(('sample_thpts(node/sec)', np.mean(sample_node_thpts[num_step:])))
    test_result.append(('sample_thpts(gb/sec)', np.mean(sample_node_thpts[num_step:]) * np.dtype(np.uint32).itemsize / 1024 / 1024 / 1024))
    test_result.append(('step_time:sample_time', np.mean(sample_times[num_step:])))
    test_result.append(('step_time:core_sample_time', np.mean(core_sample_times[num_step:])))
    test_result.append(('step_time:fill_sample_input_time', np.mean(fill_sample_input_times[num_step:])))
    test_result.append(('step_time:remap_time', np.mean(remap_times[num_step:])))
    test_result.append(('step_time:sample_coo_time', np.mean(sample_coo_times[num_step:])))
    test_result.append(('step_time:sample_kernel_time', np.mean(sample_kernel_times[num_step:])))
    test_result.append(('step_time:sample_compact_edge_time', np.mean(sample_compact_edge_times[num_step:])))
    for k, v in test_result:
        print('test_result:{:}={:.4f}'.format(k, v))

    sam.report_init()

    sam.report_node_access()
    sam.dump_trace()
    sam.shutdown()

    # print(sample_coo_times)
    # print(sample_kernel_times)

if __name__ == '__main__':
    run()
