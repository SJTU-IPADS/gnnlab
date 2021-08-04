import argparse
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

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
    default_run_config['arch'] = 'arch0'
    default_run_config['sample_type'] = sam.kKHop0
    default_run_config['pipeline'] = False  # default value must be false
    default_run_config['dataset_path'] = '/graph-learning/samgraph/reddit'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/products'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    default_run_config['cache_policy'] = sam.kCacheByHeuristic
    default_run_config['cache_percentage'] = 0.0

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
    default_run_config['sample_devices'] = [sam.cpu()]
    default_run_config['train_devices'] = [sam.gpu(0), sam.gpu(1)]

    run_config = parse_args(default_run_config)

    print('Evaluation time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(*run_config.items(), sep='\n')

    # TODO: change the arch configuration to support multi-GPU train
    '''
    run_config['arch'] = sam.meepo_archs[run_config['arch']]
    run_config['arch_type'] = run_config['arch']['arch_type']
    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']
    '''
    # FIXME: convert to sampler_ctx and trainer_ctx
    run_config['num_sample_worker'] = len(run_config['sample_devices'])
    run_config['num_train_worker'] = len(run_config['train_devices'])

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    # arch1 doesn't support pipelining
    if run_config['arch_type'] == sam.kArch1:
        run_config['pipeline'] = False

    return run_config

def run_init():
    sam.config(run_config)
    sam.config_khop(run_config)
    sam.init()

def run_sample(worker_id, run_config, epoch_barrier):
    queue = run_config['mpq']
    if run_config['pipeline']:
        sam.start()

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    for epoch in range(num_epoch):
        for step in range(num_step):
            if not run_config['pipeline']:
                sam.sample_once()
            batch_key = sam.get_next_batch(epoch, step)
            # TODO: add a API provide the blocks, input_nodes, output_nodes
            blocks, input_nodes, output_nodes = sam.get_dgl_blocks_multigpu(batch_key, num_layer)
            # TODO: send blocks and nodes
            queue.put((blocks, input_nodes, output_nodes))
        epoch_barrier.wait()

def run_train(worker_id, run_config, epoch_barrier):
    queue = run_config['mpq']
    dev_id = run_config['train_devices'][worker_id]
    num_worker = run_config['num_train_worker']


    train_device = th.device('cuda:%d' % dev_id)
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

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_convert_times = []
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
            # TODO: get blocks, input_nodes, output_nodes from queue
            # (blocks, input_nodes, output_nodes) = queue.get(True)
            # TODO: implement the extract step with input_nodes and output_nodes
            t1 = time.time()
            # TODO: implement the extract step with input_nodes and output_nodes
            batch_input, batch_label = sam.extract(input_nodes, output_nodes)
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
            train_time = t3 - t2
            total_time = t3 - t0

            sam.log_step(epoch, step, sam.kLogL1TrainTime, train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
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
        # sync with sample process each epoch
        epoch_barrier.wait()

        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime))
        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_convert_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_total_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

    sam.report_node_access()
    sam.shutdown()


if __name__ == '__main__':
    run_config = get_run_config()

    num_sample_worker = run_config['num_sample_worker']
    num_train_worker = run_config['num_train_worker']
    run_config["mpq"] = mp.Queue()
    epoch_barrier = mp.Barrier(num_sample_worker + num_train_worker)

    run_init()

    workers = []
    # sample processes
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(worker_id, run_config, epoch_barrier))
        p.start()
        workers.append(p)
    # train processes
    for worker_id in range(num_train_worker):
        p = mp.Process(target=run_train, args=(worker_id, run_config, epoch_barrier))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
