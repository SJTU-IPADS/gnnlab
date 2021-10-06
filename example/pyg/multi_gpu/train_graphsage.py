import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
import sys
import math
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.nn import SAGEConv
from torch_geometric.data import NeighborSampler


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation, dropout):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(1, num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--devices', nargs='+',
                           type=int, default=default_run_config['devices'])
    argparser.add_argument('--dataset', type=str,
                           default=default_run_config['dataset'])
    argparser.add_argument('--root-path', type=str,
                           default='/graph-learning/samgraph/')
    argparser.add_argument('--pipelining', action='store_true',
                           default=default_run_config['pipelining'])
    argparser.add_argument(
        '--no-pipelining', dest='pipelining', action='store_false')
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

    argparser.add_argument('--validate-configs',
                           action='store_true', default=False)

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['devices'] = [0, 1]
    default_run_config['dataset'] = 'reddit'
    # default_run_config['dataset'] = 'products'
    # default_run_config['dataset'] = 'papers100M'
    # default_run_config['dataset'] = 'com-friendster'
    default_run_config['root_path'] = '/graph-learning/samgraph/'
    default_run_config['pipelining'] = False
    default_run_config['num_sampling_worker'] = 16

    # In PyG, the order from root to leaf is from front to end
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5

    run_config = parse_args(default_run_config)

    assert(len(run_config['devices']) > 0)
    assert(run_config['num_sampling_worker'] >= 0)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_worker'] = len(run_config['devices'])
    run_config['num_sampling_worker'] = run_config['num_sampling_worker'] // run_config['num_worker']

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'], force_load64=True)
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

    run_config['num_thread'] = torch.get_num_threads(
    ) // run_config['num_worker']

    run_config['seed'] = 0

    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        print('config:{:}={:}'.format(k, v))

    run_config['dataset'] = dataset
    run_config['g'] = dataset.to_pyg_graph()

    if run_config['validate_configs']:
        sys.exit()

    return run_config

def get_data_iterator(run_config, dataloader):
    if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
        return [data for data in iter(dataloader)]
    else:
        return iter(dataloader)

def sync_device():
    train_end_event = torch.cuda.Event(blocking=True)
    train_end_event.record()
    train_end_event.synchronize()


def run(worker_id, run_config):
    torch.set_num_threads(run_config['num_thread'])
    dev_id = run_config['devices'][worker_id]
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=num_worker,
                                             rank=worker_id)

    dataset = run_config['dataset']
    g = run_config['g']

    feat = dataset.feat
    label = dataset.label
    train_nids = dataset.train_set
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    if num_worker > 1:
        dataloader_sampler = DistributedSampler(train_nids, num_replicas=num_worker,
                                                rank=worker_id, shuffle=True, drop_last=False, seed=run_config['seed'])
        dataloader = NeighborSampler(g,
                                     sizes=run_config['fanout'],
                                     batch_size=run_config['batch_size'],
                                     node_idx=train_nids,
                                     return_e_id=False,
                                     num_workers=run_config['num_sampling_worker'],
                                     sampler=dataloader_sampler,
                                     prefetch_factor=run_config['prefetch_factor']
                                     )
    else:
        dataloader = NeighborSampler(g,
                                     sizes=run_config['fanout'],
                                     batch_size=run_config['batch_size'],
                                     node_idx=train_nids,
                                     shuffle=True,
                                     return_e_id=False,
                                     drop_last=False,
                                     num_workers=run_config['num_sampling_worker'],
                                     prefetch_factor=run_config['prefetch_factor']
                                     )

    model = SAGE(in_feats, run_config['num_hidden'], n_classes,
                 run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(dev_id)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])
    num_epoch = run_config['num_epoch']

    model.train()

    epoch_sample_times = []
    epoch_copy_times = []
    epoch_train_times = []
    epoch_total_times = []

    sample_times = []
    copy_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []

    for epoch in range(num_epoch):
        epoch_sample_time = 0.0
        epoch_copy_time = 0.0
        epoch_train_time = 0.0

        tic = time.time()

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        # before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        # https://pytorch.org/docs/stable/data.html
        if num_worker > 1:
            dataloader_sampler.set_epoch(epoch)

        t0 = time.time()
        for step, (batch_size, n_id, adjs) in enumerate(get_data_iterator(run_config, dataloader)):
            t1 = time.time()
            adjs = [adj.to(dev_id) for adj in adjs]
            batch_inputs = feat[n_id].to(dev_id)
            batch_labels = label[n_id[:batch_size]].to(dev_id)
            if not run_config['pipelining']:
                sync_device()
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(batch_inputs, adjs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not run_config['pipelining']:
                sync_device()

            num_samples.append(sum([adj.adj_t.nnz() for adj in adjs]))
            num_nodes.append(batch_inputs.shape[0])
            batch_inputs = None
            batch_labels = None
            adjs = None

            if num_worker > 1:
                torch.distributed.barrier()

            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            epoch_sample_time += (t1 - t0)
            epoch_copy_time += (t2 - t1)
            epoch_train_time += (t3 - t2)

            if worker_id == 0:
                print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                    epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()

        sync_device()

        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_sample_times.append(epoch_sample_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(toc - tic)

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
        mp.spawn(run, args=(run_config,), nprocs=num_worker, join=True)
