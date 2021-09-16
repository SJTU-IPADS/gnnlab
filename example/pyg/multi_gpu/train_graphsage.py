import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import fastgraph
import time
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

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['devices'] = [0, 1]
    default_run_config['dataset'] = 'reddit'
    # default_run_config['dataset'] = 'products'
    # default_run_config['dataset'] = 'papers100M'
    # default_run_config['dataset'] = 'com-friendster'
    default_run_config['root_path'] = '/graph-learning/samgraph/'
    default_run_config['num_sampling_worker'] = 16

    # In PyG, the order from root to leaf is from front to end
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5

    run_config = parse_args(default_run_config)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_worker'] = len(run_config['devices'])
    run_config['num_sampling_worker'] = run_config['num_sampling_worker'] // run_config['num_worker']

    print('Evaluation time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(*run_config.items(), sep='\n')

    return run_config


def run(worker_id, run_config):
    dev_id = run_config['devices'][worker_id]
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=num_worker,
                                             rank=worker_id)
    torch.cuda.set_device(dev_id)

    dataset = run_config['dataset']
    g = run_config['g']

    feat = dataset.feat
    label = dataset.label
    train_nids = dataset.train_set
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    if num_worker > 1:
        dataloader_sampler = DistributedSampler(train_nids, num_replicas=num_worker,
                                                rank=worker_id, shuffle=True, seed=run_config['seed'])
        dataloader = NeighborSampler(g,
                                     sizes=run_config['fanout'],
                                     batch_size=run_config['batch_size'],
                                     node_idx=train_nids,
                                     return_e_id=False,
                                     num_workers=run_config['num_sampling_worker'],
                                     sampler=dataloader_sampler,
                                     #  prefetch_factor=run_config['num_epoch']
                                     )
    else:
        dataloader = NeighborSampler(g,
                                     sizes=run_config['fanout'],
                                     batch_size=run_config['batch_size'],
                                     node_idx=train_nids,
                                     shuffle=True,
                                     return_e_id=False,
                                     num_workers=run_config['num_sampling_worker'],
                                     #  prefetch_factor=run_config['num_epoch']
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
        epoch_total_time = 0.0

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        # before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        # https://pytorch.org/docs/stable/data.html
        if num_worker > 1:
            dataloader_sampler.set_epoch(epoch)

        t0 = time.time()
        for step, (batch_size, n_id, adjs) in enumerate(dataloader):
            t1 = time.time()
            adjs = [adj.to(dev_id) for adj in adjs]
            batch_inputs = feat[n_id].to(dev_id)
            batch_labels = label[n_id[:batch_size]].to(dev_id)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(batch_inputs, adjs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t3 = time.time()

            sample_times.append(t1 - t0)
            copy_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            epoch_sample_time += (t1 - t0)
            epoch_copy_time += (t2 - t1)
            epoch_train_time += (t3 - t2)
            epoch_total_time += (t3 - t0)

            num_sample = 0
            for adj in adjs:
                num_sample += adj.adj_t.nnz()
            num_samples.append(num_sample)
            num_nodes.append(batch_inputs.shape[0])

            if worker_id == 0:
                print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                    epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), loss))
            t0 = time.time()

        if num_worker > 1:
            torch.distributed.barrier()

        epoch_sample_times.append(epoch_sample_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(epoch_total_time)

    if num_worker > 1:
        torch.distributed.barrier()

    if worker_id == 0:
        print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
            np.mean(epoch_total_times[1:]), np.mean(epoch_sample_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:])))


if __name__ == '__main__':
    run_config = get_run_config()

    dataset = fastgraph.dataset(
        run_config['dataset'], run_config['root_path'], force_load64=True)
    g = dataset.to_pyg_graph()
    run_config['dataset'] = dataset
    run_config['g'] = g
    run_config['seed'] = 12345

    num_worker = run_config['num_worker']

    if num_worker == 1:
        run(0, run_config)
    else:
        mp.spawn(run, args=(run_config,), nprocs=num_worker, join=True)
