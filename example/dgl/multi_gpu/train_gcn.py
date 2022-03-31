"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
import fastgraph
import time
import numpy as np
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import math
import sys
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
        for _ in range(1, n_layers - 1):
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
    argparser.add_argument('--use-gpu-sampling', action='store_true',
                           default=default_run_config['use_gpu_sampling'])
    argparser.add_argument('--no-use-gpu-sampling',
                           dest='use_gpu_sampling', action='store_false')
    argparser.add_argument('--devices', nargs='+',
                           type=int, default=default_run_config['devices'])
    argparser.add_argument('--dataset', type=str,
                           default=default_run_config['dataset'])
    argparser.add_argument('--root-path', type=str,
                           default=default_run_config['root_path'])
    argparser.add_argument('--pipelining', action='store_true',
                           default=default_run_config['pipelining'])
    argparser.add_argument(
        '--no-pipelining', dest='pipelining', action='store_false',)
    argparser.add_argument('--num-sampling-worker', type=int,
                           default=default_run_config['num_sampling_worker'])

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--num-epoch', type=int,
                           default=default_run_config['num_epoch'])
    argparser.add_argument('--num-hidden', type=int,
                           default=default_run_config['num_hidden'])
    argparser.add_argument('--batch-size', type=int,
                           default=default_run_config['batch_size'])
    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])

    argparser.add_argument('--validate-configs',
                           action='store_true', default=False)

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['use_gpu_sampling'] = False
    default_run_config['devices'] = [0, 1]
    default_run_config['dataset'] = 'reddit'
    # default_run_config['dataset'] = 'products'
    # default_run_config['dataset'] = 'papers100M'
    # default_run_config['dataset'] = 'com-friendster'
    default_run_config['root_path'] = '/graph-learning/samgraph/'
    default_run_config['pipelining'] = False
    default_run_config['num_sampling_worker'] = 0
    # default_run_config['num_sampling_worker'] = 16

    # DGL fanouts from front to back are from leaf to root
    default_run_config['fanout'] = [5, 10, 15]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.01
    default_run_config['dropout'] = 0.5
    default_run_config['weight_decay'] = 0.0005

    run_config = parse_args(default_run_config)

    assert(len(run_config['devices']) > 0)
    assert(run_config['num_sampling_worker'] >= 0)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_worker'] = len(run_config['devices'])
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_sampling_worker'] = run_config['num_sampling_worker'] // run_config['num_worker']

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'])
    num_train_set = dataset.train_set.shape[0]

    # [prefetch_factor]: number of samples loaded in advance by each worker.
    # 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)
    # DGL uses a custom dataset, it makes PyTorch thinks a batch is a sample.
    if run_config['pipelining'] == False and run_config['num_sampling_worker'] > 0:
        # make it sequential. sample all the batch before training.
        # assumed that drop last = False
        num_samples_per_epoch = math.ceil(
            num_train_set / run_config['num_worker'])
        num_batch_per_epoch = math.ceil(
            num_samples_per_epoch / run_config['batch_size'])
        run_config['num_prefetch_batch'] = num_batch_per_epoch
        run_config['prefetch_factor'] = math.ceil(
            num_batch_per_epoch / run_config['num_sampling_worker'])
    else:
        # default prefetch factor is 2
        run_config['prefetch_factor'] = 2

    if run_config['use_gpu_sampling']:
        run_config['sample_devices'] = run_config['devices']
        run_config['train_devices'] = run_config['devices']
        #  GPU sampling requires sample_device to be 0
        run_config['num_sampling_worker'] = 0
        # default prefetch factor is 2
        run_config['prefetch_factor'] = 2
    else:
        run_config['sample_devices'] = ['cpu' for _ in run_config['devices']]
        run_config['train_devices'] = run_config['devices']

    run_config['num_thread'] = torch.get_num_threads(
    ) // run_config['num_worker']

    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        print('config:{:}={:}'.format(k, v))

    run_config['dataset'] = dataset
    run_config['g'] = dataset.to_dgl_graph()

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def load_subtensor(feat, label, input_nodes, output_nodes, train_device):
    # feat/label is on CPU while input_nodes/output_nodes is on GPU or CPU
    input_nodes = input_nodes.to(feat.device)
    output_nodes = output_nodes.to(label.device)

    batch_inputs = torch.index_select(
        feat, 0, input_nodes.long()).to(train_device)
    batch_labels = torch.index_select(
        label, 0, output_nodes.long()).to(train_device)

    return batch_inputs, batch_labels


def get_data_iterator(run_config, dataloader):
    if run_config['use_gpu_sampling']:
        return iter(dataloader)
    else:
        # iterator = iter(dataloader)
        # if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
        # q = iterator.iter_._worker_result_queue
        # while q.qsize() < run_config['num_prefetch_batch']:
        #     # timeout is 1 second
        #     with q._notempty:
        #         q._notempty.wait(timeout=1.0)
        # return iterator
        if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
            return [data for data in iter(dataloader)]
        else:
            return iter(dataloader)


def run(worker_id, run_config):
    torch.set_num_threads(run_config['num_thread'])
    sample_device = torch.device(run_config['sample_devices'][worker_id])
    train_device = torch.device(run_config['train_devices'][worker_id])
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))

    dataset = run_config['dataset']
    g = run_config['g'].to(sample_device)
    feat = dataset.feat
    label = dataset.label

    train_nids = dataset.train_set.to(sample_device)
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    sampler = dgl.dataloading.MultiLayerNeighborSampler(run_config['fanout'])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nids,
        sampler,
        use_ddp=num_worker > 1,
        batch_size=run_config['batch_size'],
        shuffle=True,
        drop_last=False,
        prefetch_factor=run_config['prefetch_factor'],
        num_workers=run_config['num_sampling_worker'])

    model = GCN(in_feats, run_config['num_hidden'],
                n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[train_device], output_device=train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(
        model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
    num_epoch = run_config['num_epoch']

    model.train()

    epoch_sample_times = []
    epoch_graph_copy_times = []
    epoch_copy_times = []
    epoch_train_times = []
    epoch_total_times = []
    epoch_num_nodes = []
    epoch_num_samples = []

    sample_times = []
    graph_copy_times = []
    copy_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    for epoch in range(num_epoch):
        epoch_sample_time = 0.0
        epoch_graph_copy_time = 0.0
        epoch_copy_time = 0.0
        epoch_train_time = 0.0
        epoch_num_node = 0
        epoch_num_sample = 0

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        # before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        # https://pytorch.org/docs/stable/data.html
        if (num_worker > 1):
            dataloader.set_epoch(epoch)

        tic = time.time()
        t0 = time.time()
        for step, (input_nodes, output_nodes, blocks) in enumerate(get_data_iterator(run_config, dataloader)):
            if not run_config['pipelining']:
                event_sync()
            t1 = time.time()
            # graph are copied to GPU here
            blocks = [block.int().to(train_device) for block in blocks]
            t2 = time.time()
            batch_inputs, batch_labels = load_subtensor(
                feat, label, input_nodes, output_nodes, train_device)
            if not run_config['pipelining']:
                event_sync()
            t3 = time.time()
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not run_config['pipelining']:
                event_sync()

            num_samples.append(sum([block.num_edges() for block in blocks]))
            num_nodes.append(blocks[0].num_src_nodes())

            batch_inputs = None
            batch_labels = None
            blocks = None

            t4 = time.time()

            sample_times.append(t1 - t0)
            graph_copy_times.append(t2 - t1)
            copy_times.append(t3 - t1)
            train_times.append(t4 - t3)
            total_times.append(t4 - t0)

            epoch_sample_time += sample_times[-1]
            epoch_graph_copy_time += graph_copy_times[-1]
            epoch_copy_time += copy_times[-1]
            epoch_train_time += train_times[-1]
            epoch_num_node += num_nodes[-1]
            epoch_num_sample += num_samples[-1]

            if worker_id == 0:
                print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Graph copy {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                    epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times), np.mean(sample_times), np.mean(graph_copy_times), np.mean(copy_times), np.mean(train_times), loss))
            t0 = time.time()

        event_sync()

        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_sample_times.append(epoch_sample_time)
        epoch_graph_copy_times.append(epoch_graph_copy_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(toc - tic)
        epoch_num_samples.append(epoch_num_sample)
        epoch_num_nodes.append(epoch_num_node)

    if num_worker > 1:
        torch.distributed.barrier()

    print('[Worker {:d}({:s})] Avg Epoch Time {:.4f} | Avg Nodes {:.0f} | Avg Samples {:.0f} | Sample Time {:.4f} | Graph copy {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        worker_id, torch.cuda.get_device_name(train_device), np.mean(epoch_total_times[1:]), np.mean(epoch_num_nodes), np.mean(epoch_num_samples), np.mean(epoch_sample_times[1:]), np.mean(epoch_graph_copy_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:])))

    if num_worker > 1:
        torch.distributed.barrier()

    if worker_id == 0:
        test_result = {}
        test_result['epoch_time'] = np.mean(epoch_total_times[1:])
        test_result['sample_time'] = np.mean(epoch_sample_times[1:])
        test_result['copy_time'] = np.mean(epoch_copy_times[1:])
        test_result['train_time'] = np.mean(epoch_train_times[1:])
        for k, v in test_result.items():
            print('test_result:{:}={:.2f}'.format(k, v))


if __name__ == '__main__':
    run_config = get_run_config()
    num_worker = run_config['num_worker']

    if num_worker == 1:
        run(0, run_config)
    else:
        workers = []
        for worker_id in range(num_worker):
            p = mp.Process(target=run, args=(worker_id, run_config))
            p.start()
            workers.append(p)
        
        wait_and_join(workers)

