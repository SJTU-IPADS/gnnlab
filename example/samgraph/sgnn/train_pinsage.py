import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import sys
import os
import datetime
import samgraph.torch as sam
from common_config import *

"""
  We have made the following modification(or say, simplification) on PinSAGE,
  because we only want to focus on the core algorithm of PinSAGE:
    1. we modify PinSAGE to make it be able to be trained on homogenous graph.
    2. we use cross-entropy loss instead of max-margin ranking loss describe in the paper.
"""


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, dropout, act=F.relu):
        super().__init__()

        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata['n'] = self.act(self.Q(self.dropout(h_src)))
            g.edata['w'] = weights.float()
            g.update_all(fn.u_mul_e('n', 'w', 'm'), fn.sum('m', 'n'))
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = g.dstdata['n']
            ws = g.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(
                z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class PinSAGE(nn.Module):
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

        self.layers.append(WeightedSAGEConv(
            in_feats, n_hidden, n_hidden, dropout, activation))
        for _ in range(1, n_layers - 1):
            self.layers.append(WeightedSAGEConv(
                n_hidden, n_hidden, n_hidden, dropout, activation))
        self.layers.append(WeightedSAGEConv(
            n_hidden, n_hidden, n_classes, dropout, activation))

    def forward(self, blocks, h):
        for layer, block in zip(self.layers, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("PinSAGE Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--random-walk-length', type=int,
                           default=default_run_config['random_walk_length'])
    argparser.add_argument('--random-walk-restart-prob',
                           type=float, default=default_run_config['random_walk_restart_prob'])
    argparser.add_argument('--num-random-walk', type=int,
                           default=default_run_config['num_random_walk'])
    argparser.add_argument('--num-neighbor', type=int,
                           default=default_run_config['num_neighbor'])
    argparser.add_argument('--num-layer', type=int,
                           default=default_run_config['num_layer'])

    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_mode=RunMode.SGNN))
    run_config['sample_type'] = 'random_walk'

    run_config['random_walk_length'] = 3
    run_config['random_walk_restart_prob'] = 0.5
    run_config['num_random_walk'] = 4
    run_config['num_neighbor'] = 5
    run_config['num_layer'] = 3

    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch6')
    assert(run_config['sample_type'] == 'random_walk')

    print_run_config(run_config)

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def run_init(run_config):
    sam.config(run_config)
    sam.data_init()

    if run_config['validate_configs']:
        sys.exit()


def sync():
    event = torch.cuda.Event(blocking=True)
    event.record()
    event.synchronize()


def run(worker_id, run_config):
    num_worker = run_config['num_worker']
    global_barrier = run_config['global_barrier']

    ctx = run_config['workers'][worker_id]
    device = torch.device(ctx)

    print('[Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

    sam.sample_init(worker_id, ctx)
    sam.train_init(worker_id, ctx)

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))

    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = PinSAGE(in_feat, run_config['num_hidden'], num_class,
                    num_layer, F.relu, run_config['dropout'])
    model = model.to(device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=run_config['lr'])

    num_epoch = sam.num_epoch()
    num_step = sam.num_local_step()

    model.train()

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_convert_times = []
    epoch_train_times = []
    epoch_total_times_python = []
    epoch_train_total_times_profiler = []
    epoch_cache_hit_rates = []

    copy_times = []
    convert_times = []
    train_times = []
    total_times = []

    # run start barrier
    global_barrier.wait()
    print('[Worker {:d}] run for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        tic = time.time()

        for step in range(worker_id, num_step * num_worker, num_worker):
            t0 = time.time()
            sam.sample_once()
            batch_key = sam.get_next_batch()
            if run_config['pipeline']:
                sync()
            t1 = time.time()
            blocks, batch_input, batch_label = sam.get_dgl_blocks_with_weights(
                batch_key, num_layer)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not run_config['pipeline']:
                sync()
                batch_input = None
                batch_label = None
                blocks = None

            if num_worker > 1:
                torch.distributed.barrier()

            t3 = time.time()

            copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t1

            sam.log_step(epoch, step, sam.kLogL1TrainTime, train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)

            copy_times.append(copy_time)
            convert_times.append(convert_time)
            train_times.append(train_time)
            total_times.append(total_time)

            sam.report_step(epoch, step)

        sync()

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_total_times_python.append(toc - tic)

        # epoch end barrier
        global_barrier.wait()

        feat_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochFeatureBytes)
        miss_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochMissBytes)
        epoch_cache_hit_rates.append(
            (feat_nbytes - miss_nbytes) / feat_nbytes)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        )
        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_convert_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_train_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        if worker_id == 0:
            print('Epoch {:05d} | Epoch Time {:.4f} | Sample {:.4f} | Copy {:.4f} | Total Train(Profiler) {:.4f}'.format(
                epoch, epoch_total_times_python[-1], epoch_sample_times[-1], epoch_copy_times[-1], epoch_train_total_times_profiler[-1]))

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    # run end barrier
    global_barrier.wait()
    run_end = time.time()

    print('[Train  Worker {:d}] Avg Epoch {:.4f} | Sample {:.4f} | Copy {:.4f} | Train Total (Profiler) {:.4f}'.format(
          worker_id, np.mean(epoch_total_times_python[1:]), np.mean(epoch_sample_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_total_times_profiler[1:])))

    global_barrier.wait()  # barrier for pretty print

    if worker_id == 0:
        test_result = []
        test_result.append(
            ('epoch_time:sample_time', np.mean(epoch_sample_times[1:])))
        test_result.append(('epoch_time:copy_time',
                           np.mean(epoch_copy_times[1:])))
        test_result.append(('convert_time', np.mean(epoch_convert_times[1:])))
        test_result.append(('train_time', np.mean(epoch_train_times[1:])))
        test_result.append(('epoch_time:train_total', np.mean(
            epoch_train_total_times_profiler[1:])))
        test_result.append(
            ('cache_percentage', run_config['cache_percentage']))
        test_result.append(('cache_hit_rate', np.mean(
            epoch_cache_hit_rates[1:])))
        test_result.append(
            ('epoch_time:total', np.mean(epoch_total_times_python[1:])))
        test_result.append(('run_time', run_end - run_start))
        for k, v in test_result:
            print('test_result:{:}={:.2f}'.format(k, v))

        # sam.dump_trace()

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()
    run_init(run_config)

    num_worker = run_config['num_worker']

    # global barrier is used to sync all the workers
    run_config['global_barrier'] = mp.Barrier(
        num_worker, timeout=get_default_timeout())

    if num_worker == 1:
        run(0, run_config)
    else:
        workers = []
        # sample processes
        for worker_id in range(num_worker):
            p = mp.Process(target=run, args=(worker_id, run_config))
            p.start()
            workers.append(p)

        ret = sam.wait_one_child()
        if ret != 0:
            for p in workers:
                p.kill()
        for p in workers:
            p.join()

        if ret != 0:
            sys.exit(1)
