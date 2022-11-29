import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from dgl.nn.pytorch import SAGEConv
import dgl.multiprocessing as mp
import dgl
from torch.nn.parallel import DistributedDataParallel
import sys
import os
import datetime
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ["NCCL_DEBUG"] = "INFO"
'''
import samgraph.torch as sam
from common_config import *


class SAGE(nn.Module):
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
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        if use_amp:
            u, v = g.edges()
            u=u.long()
            v=v.long()
            return (h[u] * h[v]).sum(1)
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GraphSage Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--report-acc', type=int,
                           default=0)

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_mode=RunMode.FGNN))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [25, 10]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['unsupervised'] = True

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    # assert(run_config['unsupervised'] == True)
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_worker = run_config['num_sample_worker']
    global_barrier = run_config['global_barrier']

    ctx = run_config['sample_workers'][worker_id]

    print('[Sample Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

    sam.sample_init(worker_id, ctx)
    # sam.notify_sampler_ready(global_barrier)

    num_epoch = sam.num_epoch()
    num_step = sam.num_local_step()

    epoch_sample_total_times_python = []
    epoch_pipleine_sample_total_times_python = []
    epoch_sample_total_times_profiler = []
    epoch_sample_times = []
    epoch_get_cache_miss_index_times = []
    epoch_enqueue_samples_times = []

    print('[Sample Worker {:d}] run sample for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))

    # run start barrier
    global_barrier.wait()

    for epoch in range(num_epoch):
        if run_config['pipeline']:
            # epoch end barrier 1
            global_barrier.wait()

        tic = time.time()
        for step in range(num_step):
            sam.sample_once()
            # sam.report_step(epoch, step)
        toc0 = time.time()

        if not run_config['pipeline']:
            # epoch start barrier 2
            global_barrier.wait()

        # epoch end barrier
        global_barrier.wait()

        toc1 = time.time()

        epoch_sample_total_times_python.append(toc0 - tic)
        epoch_pipleine_sample_total_times_python.append(toc1 - tic)
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

    print('[Sample Worker {:d}] Avg Sample Total Time {:.4f} | Sampler Total Time(Profiler) {:.4f}'.format(
        worker_id, np.mean(epoch_sample_total_times_python[1:]), np.mean(epoch_sample_total_times_profiler[1:])))

    # run end barrier
    global_barrier.wait()

    if worker_id == 0:
        sam.report_step_average(num_epoch - 1, num_step - 1)
        sam.report_init()

    # print result
    if worker_id == 0:
        test_result = []
        test_result.append(('sample_time', np.mean(epoch_sample_times[1:])))
        test_result.append(('get_cache_miss_index_time', np.mean(
            epoch_get_cache_miss_index_times[1:])))
        test_result.append(
            ('enqueue_samples_time', np.mean(epoch_enqueue_samples_times[1:])))
        test_result.append(('epoch_time:sample_total', np.mean(
            epoch_sample_total_times_python[1:])))
        if run_config['pipeline']:
            test_result.append(
                ('pipeline_sample_epoch_time', np.mean(epoch_pipleine_sample_total_times_python[1:])))
        for k, v in test_result:
            print('test_result:{:}={:.4f}'.format(k, v))

    sys.stdout.flush()
    global_barrier.wait()  # barrier for pretty print
    # trainer print result

    sam.shutdown()


def run_train(worker_id, run_config):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # ctx= run_config['trainer_ctx'] # [worker_id]
    ctx = run_config['train_workers'][worker_id]
    num_worker = run_config['num_train_worker']
    global_barrier = run_config['global_barrier']

    if (run_config['report_acc'] != 0) and (worker_id == 0):
        import train_accuracy
        dgl_graph, valid_set, test_set, feat, label = \
            train_accuracy.load_accuracy_data(run_config['dataset'], run_config['root_path'])
        # use sample device to speedup the sampling
        # XXX: why can not work while graph is hold on this GPU ?
        acc_device = torch.device(run_config['sample_workers'][0])
        accuracy = train_accuracy.Accuracy(dgl_graph, valid_set, test_set, feat, label,
                            run_config['fanout'], run_config['batch_size'], acc_device)

    train_device = torch.device(ctx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    except:
        pass
    try:
        torch.set_float32_matmul_precision("medium")
    except:
        pass
    print('[Train  Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))

    # let the trainer initialization after sampler
    # sampler should presample before trainer initialization
    # sam.wait_for_sampler_ready(global_barrier)
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

    model = SAGE(in_feat, run_config['num_hidden'], num_class,
                 num_layer, F.relu, run_config['dropout'])
    model = model.to(train_device)
    predictor = DotPredictor().to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(model, device_ids=[train_device], output_device=train_device)
        if len(list(predictor.parameters())) > 0:
            predictor = DistributedDataParallel(predictor, device_ids=[train_device], output_device=train_device)

    # loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=run_config['lr'])

    scaler = GradScaler()

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    model.train()

    epoch_copy_times = []
    epoch_convert_times = []
    epoch_train_times = []
    epoch_total_times_python = []
    epoch_train_total_times_profiler = []
    epoch_cache_hit_rates = []

    # copy_times = []
    # convert_times = []
    # train_times = []
    # total_times = []
    total_steps = 0

    align_up_step = int(
        int((num_step + num_worker - 1) / num_worker) * num_worker)
    if num_step != align_up_step:
        print(num_step, align_up_step)
        assert(num_step == align_up_step)

    batch_keys = [0 for _ in range(align_up_step * num_epoch // num_worker)]
    train_local_time = [0 for _ in range(align_up_step * num_epoch // num_worker)]
    train_barrier_time = [0 for _ in range(align_up_step * num_epoch // num_worker)]

    # run start barrier
    global_barrier.wait()
    print('[Train  Worker {:d}] run train for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()
    run_acc_total = 0.0

    train_barrier = run_config['train_barrier']
    cur_num_batch = 0

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        epoch_acc_time = 0.0

        tic = time.time()
        if run_config['pipeline'] or run_config['single_gpu']:
            need_steps = int(num_step / num_worker)
            if worker_id < num_step % num_worker:
                need_steps += 1
            sam.extract_start(need_steps)

        for step in range(worker_id, align_up_step, num_worker):
            optimizer.zero_grad(set_to_none=True)
            if step < num_step:
                t0 = time.time()
                if (not run_config['pipeline']) and (not run_config['single_gpu']):
                    sam.sample_once()
                # sam.sample_once()
                batch_key = sam.get_next_batch()
                t1 = time.time()
                # blocks, batch_input, pair_graph, batch_label = sam.get_dgl_unsupervised_blocks(batch_key, num_layer)
                blocks, batch_input, batch_label = sam.get_dgl_blocks(batch_key, num_layer)
                pair_graph = sam.get_dgl_unsupervised_pair_graph(batch_key)
                t2 = time.time()

            # Compute loss and prediction
            if use_amp:
                with autocast(enabled=use_amp):
                    batch_pred = model(blocks, batch_input)
                    score = predictor(pair_graph, batch_pred)
                    loss = F.binary_cross_entropy_with_logits(score, batch_label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_pred = model(blocks, batch_input)
                score = predictor(pair_graph, batch_pred)
                loss = F.binary_cross_entropy_with_logits(score, batch_label)
                loss.backward()
                optimizer.step()

            # wait for the train finish then we can free the data safely
            event_sync()

            if step + num_worker < num_step:
                batch_input = None
                batch_label = None
                blocks = None

            t3 = time.time()

            copy_time = sam.get_log_step_value_by_key(batch_key, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t1

            sam.log_step_by_key(batch_key, sam.kLogL1TrainTime, train_time)
            sam.log_step_by_key(batch_key, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)
            batch_keys[cur_num_batch] = batch_key
            cur_num_batch+=1

            # copy_times.append(copy_time)
            # convert_times.append(convert_time)
            # train_times.append(train_time)
            # total_times.append(total_time)

            # sam.report_step(epoch, step)
            if (run_config['report_acc']) and \
                    (step % run_config['report_acc'] == 0) and (worker_id == 0):
                tt = time.time()
                acc = accuracy.valid_acc(model, train_device)
                acc_time = (time.time() - tt)
                epoch_acc_time += acc_time
                run_acc_total += acc_time
                print('Valid Acc: {:.2f}% | Acc Time: {:.4f} | Total Step: {:d} | Time Cost: {:.2f}'.format(
                    acc * 100.0, acc_time, total_steps, (time.time() - run_start - run_acc_total)))
            total_steps += run_config['num_train_worker']

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_total_times_python.append(toc - tic - epoch_acc_time)

        # epoch end barrier
        global_barrier.wait()

        feat_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochFeatureBytes)
        miss_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochMissBytes)
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
        if (run_config['report_acc'] != 0) and (worker_id == 0):
            tt = time.time()
            acc = accuracy.valid_acc(model, train_device)
            acc_time = (time.time() - tt)
            run_acc_total += acc_time
            print('Valid Acc: {:.2f}% | Acc Time: {:.4f} | Total Step: {:d} | Time Cost: {:.2f}'.format(
                acc * 100.0, acc_time, total_steps, (time.time() - run_start - run_acc_total)))
        if worker_id == 0:
            print('Epoch {:05d} | Epoch Time {:.4f} | Total Train Time(Profiler) {:.4f} | Copy Time {:.4f}'.format(
                epoch, epoch_total_times_python[-1], epoch_train_total_times_profiler[-1], epoch_copy_times[-1]))
        if worker_id == 0:
            sam.print_memory_usage()
        sys.stdout.flush()
        sys.stderr.flush()

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    if (run_config['report_acc'] != 0) and (worker_id == 0):
        tt = time.time()
        acc = accuracy.test_acc(model, train_device)
        acc_time = (time.time() - tt)
        run_acc_total += acc_time
        print('Test Acc: {:.2f}% | Acc Time: {:.4f} | Time Cost: {:.2f}'.format(acc * 100.0, acc_time, (time.time() - run_start - run_acc_total)))
    print('[Train  Worker {:d}] Avg Epoch Time {:.4f} | Train Total Time(Profiler) {:.4f} | Copy Time {:.4f}'.format(
          worker_id, np.mean(epoch_total_times_python[1:]), np.mean(epoch_train_total_times_profiler[1:]), np.mean(epoch_copy_times[1:])))

    # run end barrier
    sys.stdout.flush()
    global_barrier.wait()
    run_end = time.time()

    # sampler print init and result
    global_barrier.wait()  # barrier for pretty print
    if worker_id == 0:
        print(torch.cuda.memory_summary())
        sam.print_memory_usage()
        sam.report_step_average(num_epoch - 1, num_step - 1)
        sam.report_step_max(num_epoch - 1, num_step - 1)
        sam.report_step_min(num_epoch - 1, num_step - 1)
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
        test_result.append(('run_time', run_end - run_start))
        if run_config['pipeline']:
            test_result.append(
                ('pipeline_train_epoch_time', np.mean(epoch_total_times_python[1:])))
        for k, v in test_result:
            print('test_result:{:}={:.4f}'.format(k, v))

        sam.dump_trace()

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()
    run_init(run_config)

    use_amp = run_config['amp']

    num_sample_worker = run_config['num_sample_worker']
    num_train_worker = run_config['num_train_worker']

    # global barrier is used to sync all the sample workers and train workers
    run_config['global_barrier'] = mp.Barrier(
        num_sample_worker + num_train_worker, timeout=get_default_timeout())
    run_config['train_barrier'] = mp.Barrier(num_train_worker, timeout=get_default_timeout())

    workers = []
    # sample processes
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(
            worker_id, run_config))
        p.start()
        workers.append(p)

    # train processes
    for worker_id in range(num_train_worker):
        p = mp.Process(target=run_train, args=(
            worker_id, run_config))
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
