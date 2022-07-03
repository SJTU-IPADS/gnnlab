import argparse
from ssl import ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
import time
from typing import List
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
import re
from collections import defaultdict


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
    argparser.add_argument('--unified-memory', action='store_true', default=True)
    argparser.add_argument('--unified-memory-ctx', type=str, nargs='+', default=argparse.SUPPRESS)

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_mode=RunMode.FGNN))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [5, 10, 15]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['weight_decay'] = 0.0005

    run_config.update(parse_args(run_config))
    process_common_config(run_config)

    run_config['arch'] = 'arch9'
    run_config['_arch'] = 9
    # run_config['num_train_workers'] = 1
    # run_config['train_workers'] = ['cuda:0']
    # run_config['num_train_workers'] = 6
    assert(run_config['num_train_worker'] > 0)
    run_config['train_workers'] = [f'cuda:{i}' for i in range(run_config['num_train_worker'])]
    run_config['num_sample_worker'] = len(run_config['unified_memory_ctx'])
    run_config['sample_workers'] = run_config['unified_memory_ctx']
    
    # print(run_config)
    assert(run_config['arch'] == 'arch9')
    assert(run_config['sample_type'] != 'random_walk')
    assert(len(set(run_config['sample_workers']) | set(run_config['train_workers'])) == run_config['num_train_worker'] + run_config['num_sample_worker'])
    # assert(run_config['sample_workers'] == ['cuda:1', 'cuda:2'])

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    return run_config


def run_init(run_config):
    sam.config(run_config)
    sam.data_init()

    if run_config['validate_configs']:
        sys.exit()


def run_sample(run_config):
    ctxes:List[str] = run_config['unified_memory_ctx']
    num_worker = len(ctxes)

    global_barrier = run_config['global_barrier']

    print(f'[Sample Worker] Started with PID {os.getpid()}, ctx: {ctxes}', flush=True)
    # sam.sample_init(worker_id, ctx)
    sam.um_sample_init(num_worker)
    sam.notify_sampler_ready(global_barrier)

    num_epoch = sam.num_epoch()
    global_step = sam.steps_per_epoch()
    num_step = sam.num_local_step()

    epoch_sample_total_times_python = []
    epoch_pipeline_sample_total_times_python = []
    epoch_sample_total_times_profiler = []
    epoch_sample_times = []
    epoch_get_cache_miss_index_times = []
    epoch_enqueue_samples_times = []

    step_sampler = []
    step_sample_total_times = []
    step_sample_times = []
    step_coo_times = []
    step_core_sample_times = []
    step_remap_times = []
    step_shuffle_times = []
    step_num_nodes = []
    step_num_samples = []
    sampler_epoch_sample_total_times = defaultdict(list)
    sampler_epoch_sample_times = defaultdict(list)
    sampler_epoch_core_sample_times = defaultdict(list)
    sampler_epoch_coo_times = defaultdict(list)
    sampler_epoch_remap_times = defaultdict(list)
    sampler_epoch_shuffle_times = defaultdict(list)

    print('[Sample Worker (PID {}, ctx: {})] run sample for {} epochs with {} steps, global {} step'.format(
        os.getpid(), ctxes, num_epoch, num_step, global_step), flush=True)

    # run start barrier
    global_barrier.wait()

    for epoch in range(num_epoch):
        if run_config['pipeline']:
            # epoch start barrier 1
            global_barrier.wait()

        tic = time.time()
        for step in range(num_step):
            sam.sample_once()
            # sam.report_step(epoch, step)
        # print(f"epoch {epoch} done", flush=True)
        # time.sleep(10)
        toc0 = time.time()

        if not run_config['pipeline']:
            # epoch start barrier 2
            global_barrier.wait()

        # epoch end barrier
        global_barrier.wait()

        toc1 = time.time()

        epoch_sample_total_times_python.append(toc0 - tic)
        epoch_pipeline_sample_total_times_python.append(toc1 - tic)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime))
        epoch_get_cache_miss_index_times.append(
            sam.get_log_epoch_value(
                epoch, sam.KLogEpochSampleGetCacheMissIndexTime)
        )
        epoch_enqueue_samples_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleSendTime)
        )
        epoch_sample_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTotalTime)
        )

        sampler_cur_epoch_sample_total_timee = defaultdict(int)
        sampler_cur_epoch_sample_time = defaultdict(int)
        sampler_cur_epoch_core_sample_time = defaultdict(int)
        sampler_cur_epoch_coo_time = defaultdict(int)
        sampler_cur_epoch_remap_time = defaultdict(int)
        sampler_cur_epoch_shuffle_time = defaultdict(int)
        for step in range(global_step):
            step_sampler.append(int(sam.get_log_step_value(epoch, step, sam.kLogL1SamplerId)))
            step_sample_total_times.append(sam.get_log_step_value(epoch, step, sam.kLogL1SampleTotalTime))
            step_sample_times.append(sam.get_log_step_value(epoch, step, sam.kLogL1SampleTime))
            step_core_sample_times.append(sam.get_log_step_value(epoch, step, sam.kLogL2CoreSampleTime))
            step_coo_times.append(sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleCooTime))
            step_remap_times.append(sam.get_log_step_value(epoch, step, sam.kLogL2IdRemapTime))
            step_shuffle_times.append(sam.get_log_step_value(epoch, step, sam.kLogL2ShuffleTime))
            step_num_nodes.append(sam.get_log_step_value(epoch, step, sam.kLogL1NumNode))
            step_num_samples.append(sam.get_log_step_value(epoch, step, sam.kLogL1NumSample))
            
            cur_sampler = step_sampler[-1]
            sampler_cur_epoch_sample_total_timee[cur_sampler] += step_sample_total_times[-1]
            sampler_cur_epoch_sample_time[cur_sampler] += step_sample_times[-1]
            sampler_cur_epoch_core_sample_time[cur_sampler] += step_core_sample_times[-1]
            sampler_cur_epoch_coo_time[cur_sampler] += step_coo_times[-1]
            sampler_cur_epoch_remap_time[cur_sampler] += step_remap_times[-1]
            sampler_cur_epoch_shuffle_time[cur_sampler] += step_shuffle_times[-1]
        for k in sampler_cur_epoch_sample_time.keys():
            sampler_epoch_sample_total_times[k].append(sampler_cur_epoch_sample_total_timee[k])
            sampler_epoch_sample_times[k].append(sampler_cur_epoch_sample_time[k])
            sampler_epoch_core_sample_times[k].append(sampler_cur_epoch_core_sample_time[k])
            sampler_epoch_coo_times[k].append(sampler_cur_epoch_coo_time[k])
            sampler_epoch_remap_times[k].append(sampler_cur_epoch_remap_time[k])
            sampler_epoch_shuffle_times[k].append(sampler_cur_epoch_shuffle_time[k])
            

    sam.report_step_average(epoch - 1, step - 1)

    print('[Sampler Process {:d}] Avg Sampler Total Time(Profiler) {:.4f}'.format(
        os.getpid(), np.mean(epoch_sample_total_times_profiler[1:])))
    for k in sampler_epoch_sample_total_times.keys():
        print('[Sample Worker {}] Sampler Total Time(Profiler) {:.4f}'.format(
            k, np.mean(sampler_epoch_sample_total_times[k][1:])))

    # run end barrier
    global_barrier.wait()

    sam.report_init()

    test_result = []
    test_result.append(('sample_time', np.mean(epoch_sample_times[1:])))
    test_result.append(('get_cache_miss_index_time', np.mean(
        epoch_get_cache_miss_index_times[1:])))
    test_result.append(
        ('enqueue_samples_time', np.mean(epoch_enqueue_samples_times[1:])))
    test_result.append(('epoch_time:sample_total', np.mean(
        epoch_sample_total_times_python[1:])))
    test_result.append(('epoch_time:sample_total(profiler)', np.mean(epoch_sample_total_times_profiler[1:])))
    if run_config['pipeline']:
        test_result.append(
            ('pipeline_sample_epoch_time', np.mean(epoch_pipeline_sample_total_times_python[1:])))
    test_result.append(('init:presample', sam.get_log_init_value(sam.kLogInitL2Presample)))
    test_result.append(('init:load_dataset:mmap', sam.get_log_init_value(sam.kLogInitL3LoadDatasetMMap)))
    test_result.append(('init:load_dataset:copy:sampler', sam.get_log_init_value(sam.kLogInitL3LoadDatasetCopy)))
    test_result.append(('init:dist_queue:alloc+push',
        sam.get_log_init_value(sam.kLogInitL3DistQueueAlloc)+sam.get_log_init_value(sam.kLogInitL3DistQueuePush)))
    test_result.append(('init:dist_queue:pin:sampler', sam.get_log_init_value(sam.kLogInitL3DistQueuePin)))
    test_result.append(('init:internal:sampler', sam.get_log_init_value(sam.kLogInitL2InternalState)))
    test_result.append(('init:cache:sampler', sam.get_log_init_value(sam.kLogInitL2BuildCache)))

    for k in sampler_epoch_sample_times.keys():
        test_result.append((f'epoch_time:sampler({k}):sample_time', np.mean(sampler_epoch_sample_times[k][1:])))
        test_result.append((f'epoch_time:sampler({k}):core_sample_time', np.mean(sampler_epoch_core_sample_times[k][1:])))
        test_result.append((f'epoch_time:sampler({k}):coo_time', np.mean(sampler_epoch_coo_times[k][1:])))
        test_result.append((f'epoch_time:sampler({k}):remap:time', np.mean(sampler_epoch_remap_times[k][1:])))
        test_result.append((f'epoch_time:sampler({k}):shuffle_time', np.mean(sampler_epoch_shuffle_times[k][1:])))

    for ctx in ctxes:
        id = int(re.search(r'[0-9]+', ctx).group(0))
        my_sampler = np.array(step_sampler)[global_step:]
        my_sample_times = (np.array(step_sample_times)[global_step:])[my_sampler == id]
        my_core_sample_times = (np.array(step_core_sample_times)[global_step:])[my_sampler == id]
        my_coo_times = (np.array(step_coo_times)[global_step:])[my_sampler == id]
        my_remap_times = (np.array(step_remap_times)[global_step:])[my_sampler == id]
        my_shuffle_times = (np.array(step_shuffle_times)[global_step:])[my_sampler == id]
        my_num_nodes = (np.array(step_num_nodes)[global_step:])[my_sampler == id]
        my_num_samples = (np.array(step_num_samples)[global_step:])[my_sampler == id]
        test_result.append((f'sampler({ctx}):total_sample_task_cnt', len(my_sample_times)))
        test_result.append((f'sampler({ctx}):num_nodes', np.mean(my_num_nodes)))
        test_result.append((f'sampler({ctx}):num_samples', np.mean(my_num_samples)))
        test_result.append((f'step_time:sampler({ctx}):sample_time', np.mean(my_sample_times)))
        test_result.append((f'step_time:sampler({ctx}):sample_core_time', np.mean(my_core_sample_times)))
        test_result.append((f'step_time:sampler({ctx}):sample_coo_time', np.mean(my_coo_times)))
        test_result.append((f'step_time:sampler({ctx}):remap_time', np.mean(my_remap_times)))
        test_result.append((f'step_time:sampler({ctx}):shuffle_time', np.mean(my_shuffle_times)))

        # print(f'sampler({ctx}) sample_time {my_sample_times.tolist()}')
        # print(f'sampler({ctx}) sample_coo_time {my_coo_times.tolist()}')
        # print(f'sampler({ctx}) remap_time {my_remap_times.tolist()}')
        # print(f'sampler({ctx}) shuffle time {my_shuffle_times.tolist()}')

    for k, v in test_result:
        print('test_result:{:}={:.6f}'.format(k, v), flush=True)

    sys.stdout.flush()
    global_barrier.wait()  # barrier for pretty print
    # trainer print result

    sam.shutdown()


def run_train(worker_id, run_config):
    ctx = run_config['train_workers'][worker_id]
    num_worker = run_config['num_train_worker']
    global_barrier = run_config['global_barrier']

    train_device = torch.device(ctx)
    print('[Train  Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

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
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))

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
    epoch_train_total_times_profiler = []
    epoch_pipeline_train_total_times_python = []
    epoch_cache_hit_rates = []
    epoch_miss_nbytes = []
    epoch_feat_nbytes = []

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
    run_start = time.time()

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        tic = time.time()
        if run_config['pipeline'] or run_config['single_gpu']:
            need_steps = int(num_step / num_worker)
            if worker_id < num_step % num_worker:
                need_steps += 1
            sam.extract_start(need_steps)

        for step in range(worker_id, align_up_step, num_worker):
            
            if step < num_step:
                t0 = time.time()
                if (not run_config['pipeline']) and (not run_config['single_gpu']):
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

            # wait for the train finish then we can free the data safely
            event_sync()

            if (step + num_worker < num_step):
                batch_input = None
                batch_label = None
                blocks = None

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

            # sam.report_step_average(epoch, step)

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
        epoch_miss_nbytes.append(miss_nbytes)
        epoch_feat_nbytes.append(feat_nbytes)
        epoch_cache_hit_rates.append(
            (feat_nbytes - miss_nbytes) / feat_nbytes)
        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_convert_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_train_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        if worker_id == 0:
            print('Epoch {:05d} | Epoch Time {:.4f} | Total Train Time(Profiler) {:.4f} | Copy Time {:.4f}'.format(
                epoch, epoch_total_times_python[-1], epoch_train_total_times_profiler[-1], epoch_copy_times[-1]))

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    print('[Train  Worker {:d}] Avg Epoch Time {:.4f} | Train Total Time(Profiler) {:.4f} | Copy Time {:.4f}'.format(
          worker_id, np.mean(epoch_total_times_python[1:]), np.mean(epoch_train_total_times_profiler[1:]), np.mean(epoch_copy_times[1:])))

    # run end barrier
    sys.stdout.flush()
    global_barrier.wait()
    run_end = time.time()

    # sampler print init and result
    global_barrier.wait()  # barrier for pretty print
    sam.report_step_average(num_epoch - 1, num_step - 1)
    sam.report_init()

    if worker_id == 0:
        sam.report_step_average(epoch - 1, step - 1)
        sam.report_init()
        test_result = []
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
        test_result.append(('epoch_feat_nbytes', np.mean(epoch_feat_nbytes[1:])))
        test_result.append(('batch_feat_nbytes', np.mean(epoch_feat_nbytes[1:])/(align_up_step/num_worker)))
        test_result.append(('epoch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])))
        test_result.append(('batch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])/(align_up_step/num_worker)))
        test_result.append(('batch_copy_time', np.mean(epoch_copy_times[1:])/(align_up_step/num_worker)))
        test_result.append(('batch_train_time', np.mean(epoch_train_total_times_profiler[1:])/(align_up_step/num_worker)))
        if run_config['pipeline']:
            test_result.append(
                ('pipeline_train_epoch_time', np.mean(epoch_total_times_python[1:])))
        test_result.append(('run_time', run_end - run_start))
        test_result.append(('init:load_dataset:copy:trainer', sam.get_log_init_value(sam.kLogInitL3LoadDatasetCopy)))
        test_result.append(('init:dist_queue:pin:trainer', sam.get_log_init_value(sam.kLogInitL3DistQueuePin)))
        test_result.append(('init:internal:trainer', sam.get_log_init_value(sam.kLogInitL2InternalState)))
        test_result.append(('init:cache:trainer', sam.get_log_init_value(sam.kLogInitL2BuildCache)))
        for k, v in test_result:
            print('test_result:{:}={:.4f}'.format(k, v))

        # sam.dump_trace()

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()
    run_init(run_config)

    num_sample_worker = run_config['num_sample_worker']
    num_train_worker = run_config['num_train_worker']

    # global barrier is used to sync all the sample workers and train workers
    run_config['global_barrier'] = mp.Barrier(
        1 + num_train_worker, timeout=get_default_timeout())

    workers = []
    # sample processes
    samplers = mp.Process(target=run_sample, args=(run_config, ))
    samplers.start()
    workers.append(samplers)
    # for worker_id in range(num_sample_worker):
    #     p = mp.Process(target=run_sample, args=(worker_id, run_config))
    #     p.start()
    #     workers.append(p)

    # train processes
    for worker_id in range(num_train_worker):
        p = mp.Process(target=run_train, args=(worker_id, run_config))
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
