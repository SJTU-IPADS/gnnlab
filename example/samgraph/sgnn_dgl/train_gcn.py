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
import datetime
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

    run_config.update(get_default_common_config(run_mode=RunMode.SGNN_DGL))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [5, 10, 15]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['weight_decay'] = 0.0005

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch7')
    assert(run_config['sample_type'] != 'random_walk')

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def run(worker_id, run_config):
    torch.set_num_threads(run_config['torch_thread_num'])
    num_worker = run_config['num_worker']
    global_barrier = run_config['global_barrier']
    ctx = run_config['workers'][worker_id]
    device = torch.device(ctx)

    run_config['worker_id'] = worker_id
    run_config['sampler_ctx'] = ctx
    run_config['trainer_ctx'] = ctx

    sam.config(run_config)
    sam.init()

    print('[Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))

    feat = sam.get_dataset_feat()
    label = sam.get_dataset_label()
    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = GCN(in_feat, run_config['num_hidden'], num_class,
                num_layer, F.relu, run_config['dropout'])
    model = model.to(device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])

    num_epoch = sam.num_epoch()
    num_step = sam.num_local_step()

    model.train()

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_convert_times = []
    epoch_train_times = []
    epoch_total_times = []

    # run start barrier
    global_barrier.wait()
    print('[Worker {:d}] run for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        tic = time.time()

        epoch_sample_time = 0
        epoch_copy_time = 0
        epoch_convert_time = 0
        epoch_train_time = 0
        epoch_total_time = 0

        for step in range(worker_id, num_step * num_worker, num_worker):
            t0 = time.time()
            sam.sample_once()
            batch_key = sam.get_next_batch()
            t1 = time.time()
            batch_input, batch_label = sam.load_subtensor(
                batch_key, feat, label, device)
            t2 = time.time()
            blocks, _, _ = sam.get_dgl_blocks(
                batch_key, num_layer, with_feat=False)
            t3 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # wait for the train finish then we can free the data safely
            train_end_event = torch.cuda.Event(blocking=True)
            train_end_event.record()
            train_end_event.synchronize()

            batch_input = None
            batch_label = None
            blocks = None

            if num_worker > 1:
                torch.distributed.barrier()

            t4 = time.time()

            epoch_sample_time += t1 - t0
            epoch_copy_time += t2 - t1
            epoch_convert_time += t3 - t2
            epoch_train_time += t4 - t3
            epoch_total_time += t4 - t0

            sam.report_step_average(epoch, step)

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_total_times.append(toc - tic)

        # epoch end barrier
        global_barrier.wait()

        epoch_sample_times.append(epoch_sample_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_convert_times.append(epoch_convert_time)
        epoch_train_times.append(epoch_train_time)
        if worker_id == 0:
            print('Epoch {:05d} | Epoch Time {:.4f} | Sample {:.4f} | Copy {:.4f} | Train {:.4f}({:.4f})'.format(
                epoch, epoch_total_time, epoch_sample_time, epoch_copy_time, epoch_convert_time + epoch_train_time, epoch_convert_time))

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    # run end barrier
    global_barrier.wait()
    run_end = time.time()

    print('[Train  Worker {:d}] Avg Epoch {:.4f} | Sample {:.4f} | Copy {:.4f} | Train Total (Profiler) {:.4f}'.format(
          worker_id, np.mean(epoch_total_times[1:]), np.mean(epoch_sample_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:]) + np.mean(epoch_convert_times[1:])))

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
            np.mean(epoch_train_times[1:]) + np.mean(epoch_convert_times[1:]))))
        test_result.append(
            ('epoch_time:total', np.mean(epoch_total_times[1:])))
        test_result.append(('run_time', run_end - run_start))
        for k, v in test_result:
            print('test_result:{:}={:.2f}'.format(k, v))

        # sam.dump_trace()

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()

    num_worker = run_config['num_worker']

    # global barrier is used to sync all the sample workers and train workers
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
