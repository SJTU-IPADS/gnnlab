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
import samgraph.torch as sam
from common_config import *

import collcache.torch as co

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

    run_config.update(get_default_common_config(run_mode=RunMode.SGNN))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [25, 10]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['unsupervised'] = True
    run_config['coll_ignore'] = True

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch6')
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


def run(worker_id, run_config):
    num_worker = run_config['num_worker']
    global_barrier = run_config['global_barrier']

    ctx = run_config['workers'][worker_id]
    device = torch.device(ctx)

    if (run_config['report_acc'] != 0) and (worker_id == 0):
        import train_accuracy
        dgl_graph, valid_set, test_set, feat, label = \
            train_accuracy.load_accuracy_data(run_config['dataset'], run_config['root_path'])
        # use sample device to speedup the sampling
        # XXX: why can not work while graph is hold on this GPU ?
        acc_device = torch.device(run_config['workers'][0])
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
    print('[Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))
    print(f"worker {worker_id} at process {os.getpid()}")
    with open(f"/tmp/infer_{worker_id}.pid", 'w') as f:
        print(f"{os.getpid()}", file=f, flush=True)
    time.sleep(5)
    sam.sample_init(worker_id, ctx)
    sam.train_init(worker_id, ctx)

    ds_emb = sam.get_dataset_feat()
    run_config['num_global_step_per_epoch'] = sam.num_local_step() * num_worker
    run_config['num_device'] = num_worker
    run_config['num_total_item'] = ds_emb.shape[0]

    co.config(run_config)
    co.coll_cache_record_init(worker_id)

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
    model = model.to(device)
    predictor = DotPredictor().to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device)
        if len(list(predictor.parameters())) > 0:
            predictor = DistributedDataParallel(predictor, device_ids=[train_device], output_device=train_device)


    # loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=run_config['lr'])

    scaler = GradScaler()

    num_epoch = sam.num_epoch()
    num_step = sam.num_local_step()

    model.train()

    epoch_sample_total_times = []
    epoch_sample_times = []
    epoch_get_cache_miss_index_times = []
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
    total_steps = 0

    # run start barrier
    global_barrier.wait()
    print('[Worker {:d}] run for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()
    run_acc_total = 0.0

    presc_start = time.time()
    print("presamping")
    for step in range(worker_id, num_step * num_worker, num_worker):
        sam.sample_once()
        batch_key = sam.get_next_batch()
        block_input_nodes = sam.get_graph_input_nodes(batch_key).to('cpu')
        event_sync()
        co.coll_torch_record(worker_id, block_input_nodes)
    sam.reset_progress()
    presc_stop = time.time()
    print(f"presamping takes {presc_stop - presc_start}")

    co.coll_torch_init_t(worker_id, worker_id, ds_emb, run_config["cache_percentage"])

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        epoch_acc_time = 0.0

        tic = time.time()

        for step in range(worker_id, num_step * num_worker, num_worker):
            optimizer.zero_grad(set_to_none=True)
            if True:
                t0 = time.time()
                if not run_config['pipeline']:
                    sam.sample_once()
                elif epoch + step == worker_id:
                    sam.extract_start(0)
                batch_key = sam.get_next_batch()
                # if epoch == 0:
                #     block_input_nodes = sam.get_graph_input_nodes(batch_key).to('cpu')
                #     event_sync()
                #     co.coll_torch_record(worker_id, block_input_nodes)
                # else:
                block_input_nodes = sam.get_graph_input_nodes(batch_key)
                batch_input = co.coll_torch_lookup_key_t_val_ret(worker_id, block_input_nodes)
                t1 = time.time()
                blocks, _, batch_label = sam.get_dgl_blocks(
                    batch_key, num_layer)
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
            if True:
                batch_input = None
                batch_label = None
                blocks = None

            t3 = time.time()

            copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t1

            sam.log_step_by_key(batch_key, sam.kLogL1TrainTime, train_time)
            sam.log_step_by_key(batch_key, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)

            copy_times.append(copy_time)
            convert_times.append(convert_time)
            train_times.append(train_time)
            total_times.append(total_time)

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
            total_steps += run_config['num_worker']

        event_sync()

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
        epoch_sample_total_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTotalTime)
        )
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
            print('Epoch {:05d} | Epoch Time {:.4f} | Sample {:.4f} | Copy {:.4f} | Total Train(Profiler) {:.4f}'.format(
                epoch, epoch_total_times_python[-1], epoch_sample_total_times[-1], epoch_copy_times[-1], epoch_train_total_times_profiler[-1]))

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    if (run_config['report_acc'] != 0) and (worker_id == 0):
        tt = time.time()
        acc = accuracy.test_acc(model, train_device)
        acc_time = (time.time() - tt)
        run_acc_total += acc_time
        print('Test Acc: {:.2f}% | Acc Time: {:.4f} | Time Cost: {:.2f}'.format(acc * 100.0, acc_time, (time.time() - run_start - run_acc_total)))
    print('[Train  Worker {:d}] Avg Epoch {:.4f} | Sample {:.4f} | Copy {:.4f} | Train Total (Profiler) {:.4f}'.format(
          worker_id, np.mean(epoch_total_times_python[1:]), np.mean(epoch_sample_total_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_total_times_profiler[1:])))
    # run end barrier
    global_barrier.wait()
    run_end = time.time()


    global_barrier.wait()  # barrier for pretty print

    if worker_id == 0:
        print(torch.cuda.memory_summary())
        sam.print_memory_usage()
        sam.report_step_average(num_epoch - 1, num_step * num_worker - 1)
        sam.report_init()
        co.report_step_average(0)
        test_result = []
        test_result.append(('sample_time', np.mean(epoch_sample_times[1:])))
        test_result.append(
            ('epoch_time:sample_total', np.mean(epoch_sample_total_times[1:])))
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
            print('test_result:{:}={:.4f}'.format(k, v))

        # sam.dump_trace()

    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()
    run_init(run_config)

    use_amp = run_config['amp']

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
