import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import sys
import samgraph.torch as sam
import datetime
from common_config import *

th=torch
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, order=1, act=None,
                 dropout=0, batch_norm=False, aggr="concat"):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        self.bias = nn.ParameterList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bias.append(nn.Parameter(th.zeros(out_dim)))

        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.batch_norm = batch_norm
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(order + 1):
                self.offset.append(nn.Parameter(th.zeros(out_dim)))
                self.scale.append(nn.Parameter(th.ones(out_dim)))

        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)

    def feat_trans(self, features, idx):  # linear transformation + activation + batch normalization
        h = self.lins[idx](features) + self.bias[idx]

        if self.act is not None:
            h = self.act(h)

        if self.batch_norm:
            mean = h.mean(dim=1).view(h.shape[0], 1)
            var = h.var(dim=1, unbiased=False).view(h.shape[0], 1) + 1e-9
            h = (h - mean) * self.scale[idx] * th.rsqrt(var) + self.offset[idx]

        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        D_norm = g.ndata['train_D_norm'] if 'train_D_norm' in g.ndata else g.ndata['full_D_norm']
        for _ in range(self.order):  # forward propagation
            g.ndata['h'] = h_hop[-1]
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * D_norm
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = th.cat(h_part, 1)
        else:
            raise NotImplementedError

        return h_out

#  in_feats,
#  n_hidden,
#  n_classes,
#  n_layers,
#  activation,
#  dropout

class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=3, arch="1-1-0",
                 activation=F.relu, dropout=0, batch_norm=False, aggr="concat"):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split('-')))
        assert(n_layers == len(orders))
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, order=orders[0], act=activation, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[0] + 1) * hid_dim

        for i in range(1, len(orders)-1):
            self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[i], act=activation, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
            pre_out = ((aggr == "concat") * orders[i] + 1) * hid_dim

        self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[-1], act=activation, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[-1] + 1) * hid_dim

        self.out_layer = GCNLayer(in_dim=pre_out, out_dim=out_dim, order=0, act=None, dropout=dropout, batch_norm=False, aggr=aggr)

    def forward(self, graph, features):
        h = features

        for layer in self.gcn:
            h = layer(graph, h)

        h = F.normalize(h, p=2, dim=1)
        h = self.out_layer(graph, h)

        return h




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
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--random-walk-length', type=int,
                           default=default_run_config['random_walk_length'])
    argparser.add_argument('--random-walk-restart-prob',
                           type=float, default=default_run_config['random_walk_restart_prob'])
    argparser.add_argument('--num-random-walk', type=int,
                           default=default_run_config['num_random_walk'])
    argparser.add_argument('--num-layer', type=int,
                           default=default_run_config['num_layer'], help="layer in samgraph. must be 1")
    argparser.add_argument('--num-model-layer', type=int,
                           default=default_run_config['num_model_layer'], help="model layer")

    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--no-ddp', action='store_false', dest='use_ddp',
                           default=default_run_config['use_ddp'])

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_mode=RunMode.FGNN))
    run_config['sample_type'] = 'saint'

    run_config['random_walk_length'] = 3
    run_config['random_walk_restart_prob'] = 0
    run_config['num_random_walk'] = 1
    run_config['num_layer'] = 1
    run_config['num_model_layer'] = 3

    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['use_ddp'] = True

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch5')
    assert (run_config['sample_type'] == 'saint'), "wrong sample_type: " + run_config['sample_type']

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

    print('[Sample Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))
    sam.sample_init(worker_id, ctx)
    sam.notify_sampler_ready(global_barrier)

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    if (worker_id == (num_worker - 1)):
        num_step = int(num_step - int(num_step /
                       num_worker) * worker_id)
    else:
        num_step = int(num_step / num_worker)

    epoch_sample_total_times_python = []
    epoch_pipeline_sample_total_times_python = []
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
            # epoch start barrier 1
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

    if worker_id == 0:
        sam.report_step_average(num_epoch - 1, num_step - 1)

    print('[Sample Worker {:d}] Avg Sample Total Time {:.4f} | Sampler Total Time(Profiler) {:.4f}'.format(
        worker_id, np.mean(epoch_sample_total_times_python[1:]), np.mean(epoch_sample_total_times_profiler[1:])))

    # run end barrier
    global_barrier.wait()

    if worker_id == 0:
        sam.report_init()

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
                ('pipeline_sample_epoch_time', np.mean(epoch_pipeline_sample_total_times_python[1:])))
        test_result.append(('init:presample', sam.get_log_init_value(sam.kLogInitL2Presample)))
        test_result.append(('init:load_dataset:mmap', sam.get_log_init_value(sam.kLogInitL3LoadDatasetMMap)))
        test_result.append(('init:load_dataset:copy:sampler', sam.get_log_init_value(sam.kLogInitL3LoadDatasetCopy)))
        test_result.append(('init:dist_queue:alloc+push',
          sam.get_log_init_value(sam.kLogInitL3DistQueueAlloc)+sam.get_log_init_value(sam.kLogInitL3DistQueuePush)))
        test_result.append(('init:dist_queue:pin:sampler', sam.get_log_init_value(sam.kLogInitL3DistQueuePin)))
        test_result.append(('init:internal:sampler', sam.get_log_init_value(sam.kLogInitL2InternalState)))
        test_result.append(('init:cache:sampler', sam.get_log_init_value(sam.kLogInitL2BuildCache)))
        for k, v in test_result:
            print('test_result:{:}={:.2f}'.format(k, v))

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
    num_model_layer = run_config['num_model_layer']

    # model = GCN(in_feat, run_config['num_hidden'], num_class, n_layers = num_layer, activation = F.relu, dropout = run_config['dropout'])
    # model = GCNNet(in_feat, run_config['num_hidden'], num_class, n_layers = num_model_layer, activation = F.relu, dropout = run_config['dropout'])
    model = SAGE(in_feat, run_config['num_hidden'], num_class, n_layers = num_model_layer, activation = F.relu, dropout = run_config['dropout'])
    model = model.to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[train_device], output_device=train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])

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

    # copy_times = []
    # convert_times = []
    # train_times = []
    # total_times = []

    align_up_step = int(int((num_step + num_worker - 1) / num_worker) * num_worker)

    batch_keys = [0 for _ in range(align_up_step * num_epoch // num_worker)]
    train_local_time = [0 for _ in range(align_up_step * num_epoch // num_worker)]
    train_barrier_time = [0 for _ in range(align_up_step * num_epoch // num_worker)]

    # run start barrier
    global_barrier.wait()
    print('[Train  Worker {:d}] run train for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()

    train_barrier = run_config['train_barrier']
    cur_num_batch = 0

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
                blocks, batch_input, batch_label = sam.get_dgl_subgraph(batch_key, num_layer)
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
            # t_train_f_b = time.time()
            # train_barrier.wait()

            if (step + num_worker < num_step):
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
            # train_local_time[cur_num_batch] = t_train_f_b - t2
            # train_barrier_time[cur_num_batch] = t3 - t_train_f_b
            cur_num_batch+=1

            # copy_times.append(copy_time)
            # convert_times.append(convert_time)
            # train_times.append(train_time)
            # total_times.append(total_time)

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
    global_barrier.wait()
    run_end = time.time()

    # sampler print init and result
    global_barrier.wait()  # barrier for pretty print
    # sam.report_step_average(num_epoch - 1, num_step - 1)
    # sam.report_init()

    def sync_print_trainer(buf, header=""):
        for next_one_to_print in range(num_worker):
            if next_one_to_print == worker_id:
                if worker_id == 0:
                    print(header)
                print(buf, flush=True)
            run_config['train_barrier'].wait()

    def get_val(key):
        return sam.get_log_step_value_by_key(key, sam.kLogL1FeatureBytes) - sam.get_log_step_value_by_key(key, sam.kLogL1RemoteBytes) - sam.get_log_step_value_by_key(key, sam.kLogL1MissBytes)
    sync_print_trainer(" ".join(["{:8.2f}".format(get_val(key) / 1024/1024) for key in batch_keys[-40:]]))

    def get_val(key):
        return sam.get_log_step_value_by_key(key, sam.kLogL1RemoteBytes)
    sync_print_trainer(" ".join(["{:8.2f}".format(get_val(key) / 1024/1024) for key in batch_keys[-40:]]))

    def get_val(key):
        return sam.get_log_step_value_by_key(key, sam.kLogL1MissBytes)
    sync_print_trainer(" ".join(["{:8.2f}".format(get_val(key) / 1024/1024) for key in batch_keys[-40:]]))

    class val_array:
        def __init__(self, initial_array):
            self.array = initial_array
        def prepend_avg(self):
            self.array = [np.average(self.array)] + self.array
            return self
        def to_time_str(self):
            return " ".join(["{:8.2f}".format(v * 1000 * 1000) for v in self.array])

    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL1TrainTime)               for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL1TrainTime")
    sync_print_trainer(val_array(train_local_time[-40:])   .prepend_avg().to_time_str(), header="train f+b")
    sync_print_trainer(val_array(train_barrier_time[-40:]) .prepend_avg().to_time_str(), header="train opt")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL1CopyTime)               for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL1CopyTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL1RecvTime)               for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL1RecvTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL2GraphCopyTime)          for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL2GraphCopyTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL2CacheCopyTime)          for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL2CacheCopyTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL3CacheGetIndexTime)      for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL3CacheGetIndexTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL3CacheCombineCacheTime)  for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL3CacheCombineCacheTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL3CacheCombineRemoteTime) for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL3CacheCombineRemoteTime")
    sync_print_trainer(val_array([sam.get_log_step_value_by_key(key, sam.kLogL3CacheCombineMissTime)   for key in batch_keys[-40:]]).prepend_avg().to_time_str(), header="kLogL3CacheCombineMissTime")

    t_b = time.time()
    for i in range(1000):
        train_barrier.wait()
    barrier_time = time.time() - t_b
    if worker_id == 0:
        print(f"1000 barrier: {barrier_time}s")

    t_b = time.time()
    for i in range(1000):
        event_sync()
    event_sync_time = time.time() - t_b
    if worker_id == 0:
        print(f"1000 event sync: {event_sync_time}s")

    if worker_id == 0:
        sam.report_step_average(num_epoch - 1, num_step - 1)
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
        num_sample_worker + num_train_worker, timeout=get_default_timeout())
    run_config['train_barrier'] = mp.Barrier(num_train_worker, timeout=get_default_timeout())

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

    ret = sam.wait_one_child()
    if ret != 0:
        for p in workers:
            p.kill()
    for p in workers:
        p.join()

    if ret != 0:
        sys.exit(1)
