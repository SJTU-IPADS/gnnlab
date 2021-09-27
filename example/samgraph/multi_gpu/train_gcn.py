import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import sys
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
        for _ in range(n_layers - 2):
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

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_multi_gpu=True))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [5, 10, 15]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['weight_decay'] = 0.0005

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch5')
    assert(run_config['sample_type'] != 'random_walk')

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    return run_config


def run_init(run_config):
    sam.config(run_config)
    sam.data_init()

    if run_config['validate_configs']:
        sys.exit()


def run_sample(worker_id, run_config):
    num_worker = run_config['num_sample_worker']
    global_barrier = run_config['global_barrier']

    ctx = run_config['sample_workers'][worker_id]

    print('[Sample Worker {:d}/{:d}] Started with PID {:d}'.format(
        worker_id, num_worker, os.getpid()))
    sam.sample_init(worker_id, ctx)
    sam.notify_sampler_ready(global_barrier)

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    if (worker_id == (num_worker - 1)):
        num_step = int(num_step - int(num_step /
                       num_worker * worker_id))
    else:
        num_step = int(num_step / num_worker)

    epoch_sample_total_times_python = []
    epoch_sample_total_times_profiler = []
    epoch_sample_times = []
    epoch_get_cache_miss_index_times = []
    epoch_buffer_graph_times = []

    print('[Sample Worker {:d}] run sample for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))

    # run start barrier
    global_barrier.wait()

    for epoch in range(num_epoch):
        if run_config['pipeline']:
            # epoch start barrier 1
            global_barrier.wait()

        tic = time.time()
        for step in range(num_step):
            sam.sample_once()
            sam.report_step(epoch, step)
        toc = time.time()

        epoch_sample_total_times_python.append(toc - tic)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime))
        epoch_get_cache_miss_index_times.append(
            sam.get_log_epoch_value(epoch, sam.KLogEpochSampleGetCacheMissIndexTime)
        )
        epoch_buffer_graph_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleSendTime)
        )
        epoch_sample_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTotalTime)
        )

        if not run_config['pipeline']:
            # epoch start barrier 2
            global_barrier.wait()

        # epoch end barrier
        global_barrier.wait()

    print('[Sample Worker {:d}] Avg Sample Time Per Epoch {:.4f} | Sample Time(Profiler) {:.4f}'.format(
        worker_id, np.mean(epoch_sample_total_times_python[1:]), np.mean(epoch_sample_times[1:])))
    
    if worker_id == 0:
        sam.report_step_average(epoch - 1, step - 1)

    # run end barrier
    global_barrier.wait()

    if worker_id == 0:
        test_result = {}

        test_result['sample_time'] = np.mean(epoch_sample_times[1:])
        test_result['cache_index_time'] = np.mean(epoch_get_cache_miss_index_times[1:])
        test_result['buffer_graph_time'] = np.mean(epoch_buffer_graph_times[1:])
        test_result['epoch_time:sampler_total'] = np.mean(epoch_sample_total_times_profiler[1:])
        for k, v in test_result.items():
            print('test_result:{:}={:.2f}'.format(k, v))

    sam.shutdown()


def run_train(worker_id, run_config):
    ctx = run_config['train_workers'][worker_id]
    num_worker = run_config['num_train_worker']
    global_barrier = run_config['global_barrier']

    train_device = torch.device(ctx)
    print('[Train  Worker {:d}/{:d}] Started with PID {:d}'.format(worker_id, num_worker, os.getpid()))

    # let the trainer initialization after sampler
    # sampler should presample before trainer initialization
    sam.wait_for_sampler_ready(global_barrier)
    sam.train_init(worker_id, ctx)

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=get_default_timeout())

    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = GCN(in_feat, run_config['num_hidden'], num_class,
                num_layer, F.relu, run_config['dropout'])
    model = model.to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[train_device], output_device=train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(
        model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    model.train()

    epoch_copy_times = []
    epoch_convert_times = []
    epoch_train_times = []
    epoch_total_times_python = []
    epoch_total_times_profiler = []
    epoch_cache_hit_rates = []

    copy_times = []
    convert_times = []
    train_times = []
    total_times = []

    align_up_step = int(
        int((num_step + num_worker - 1) / num_worker) * num_worker)
    
    # run start barrier
    global_barrier.wait()
    print('[Train  Worker {:d}] run train for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        tic = time.time()
        if run_config['pipeline']:
            need_steps = int(num_step / num_worker)
            if worker_id < num_step % num_worker:
                need_steps += 1
            sam.extract_start(need_steps)

        for step in range(worker_id, align_up_step, num_worker):
            if step < num_step:
                t0 = time.time()
                if (not run_config['pipeline']):
                    sam.sample_once()
                batch_key = sam.get_next_batch()
                t1 = time.time()
                blocks, batch_input, batch_label = sam.get_dgl_blocks(
                    batch_key, num_layer)
                t2 = time.time()
            else:
                t0 = t1 = t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + num_worker < num_step):
                batch_input = None
                batch_label = None

            t3 = time.time()

            copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t0

            sam.log_step(epoch, step, sam.kLogL1TrainTime, train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)

            feat_nbytes = sam.get_log_epoch_value(epoch, sam.kLogEpochFeatureBytes)
            miss_nbytes = sam.get_log_epoch_value(epoch, sam.kLogEpochMissBytes)
            epoch_cache_hit_rates.append((feat_nbytes - miss_nbytes) / feat_nbytes)

            copy_times.append(copy_time)
            convert_times.append(convert_time)
            train_times.append(train_time)
            total_times.append(total_time)

            # sam.report_step(epoch, step)

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()
        
        toc = time.time()
        
        epoch_total_times_python.append(toc - tic)
        
        # epoch end barrier
        global_barrier.wait()

        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_convert_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        if worker_id == 0:
            print('Epoch {:05d} | Total Trainer Time {:.4f} | Total Trainer Time(Profiler) {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
                epoch, epoch_total_times_python[-1], epoch_total_times_profiler[-1],  epoch_copy_times[-1], epoch_convert_times[-1], epoch_train_times[-1]))

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    if worker_id == 0:
        print('[Avg] Epoch Trainer Time {:.4f} | Epoch Trainer Time(Profiler) {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
            np.mean(epoch_total_times_python[1:]), np.mean(epoch_total_times_profiler[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

    # run end barrier
    global_barrier.wait()

    if worker_id == 0:
        test_result = {}
        test_result['copy_time'] = np.mean(epoch_copy_times[1:])
        test_result['convert_time'] = np.mean(epoch_convert_times[1:])
        test_result['train_time'] = np.mean(epoch_train_times[1:])
        test_result['epoch_time:trainer_total'] = np.mean(epoch_total_times_profiler[1:])
        test_result['cache_percentage'] = run_config['cache_percentage']
        test_result['cache_hit_rate'] = np.mean(epoch_cache_hit_rates[1:])
        for k, v in test_result.items():
            print('test_result:{:}={:.2f}'.format(k, v))

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()
    run_init(run_config)

    num_sample_worker = run_config['num_sample_worker']
    num_train_worker = run_config['num_train_worker']

    # global barrier is used to sync all the sample workers and train workers
    run_config['global_barrier'] = mp.Barrier(
        num_sample_worker + num_train_worker, timeout=get_default_timeout())

    workers = []
    # sample processes
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(worker_id, run_config))
        p.start()
        workers.append(p)

    # train processes
    for worker_id in range(num_train_worker):
        p = mp.Process(target=run_train, args=(worker_id, run_config))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
