import argparse
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.function as fn

import samgraph.torch as sam
from common_config import *

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

        # D_norm = g.ndata['train_D_norm'] if 'train_D_norm' in g.ndata else g.ndata['full_D_norm']

        degs = graph.in_degrees().to(features)
        # h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        for _ in range(self.order):  # forward propagation
            g.ndata['h'] = h_hop[-1]
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h / (degs.unsqueeze(-1) + 1)
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

    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])

    argparser.add_argument('--unified-memory', action='store_true')
    argparser.add_argument('--unified-memory-percentage', type=float,
                            default=0)
    argparser.add_argument('--um-policy', type=str,
        choices=['default', 'degree', 'trainset', 'random', 'presample'],
        default='default')

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config())
    run_config['arch'] = 'arch3'
    run_config['sample_type'] = 'saint'

    run_config['random_walk_length'] = 3
    run_config['random_walk_restart_prob'] = 0
    run_config['num_random_walk'] = 1
    run_config['num_layer'] = 1
    run_config['num_model_layer'] = 3

    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5

    run_config.update(parse_args(run_config))

    # run_config["override_device"] = True
    # run_config["override_train_device"] = 'cuda:0'
    # run_config["override_sample_device"] = 'cuda:1'

    # run_config["dataset"] = 'reddit'
    # run_config["dataset"] = 'com-friendster'
    # run_config["dataset"] = 'ppi'
    # run_config["unified_memory"] = True

    process_common_config(run_config)
    assert(run_config['sample_type'] == 'saint')
    assert(run_config['unified_memory_percentage'] >= 0
           and run_config['unified_memory_percentage'] <= 1.0)

    print_run_config(run_config)

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.init()

    # sam.report_init()

    train_device = th.device(run_config['trainer_ctx'])

    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']
    num_model_layer = run_config['num_model_layer']

    # model = GCN(in_feat, run_config['num_hidden'], num_class, n_layers = num_layer, activation = F.relu, dropout = run_config['dropout'])
    # model = GCNNet(in_feat, run_config['num_hidden'], num_class, n_layers = num_model_layer, activation = F.relu, dropout = run_config['dropout'])
    model = SAGE(in_feat, run_config['num_hidden'], num_class, n_layers = num_model_layer, activation = F.relu, dropout = run_config['dropout'])
    model = model.to(train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    print("before train")
    model.train()

    epoch_sample_times = [0 for i in range(num_epoch)]
    epoch_copy_times = [0 for i in range(num_epoch)]
    epoch_convert_times = [0 for i in range(num_epoch)]
    epoch_train_times = [0 for i in range(num_epoch)]
    epoch_train_total_times_profiler = []
    epoch_total_times_profiler = [0 for i in range(num_epoch)]
    epoch_total_times_python = []
    epoch_cache_hit_rates = []
    epoch_miss_nbytes = []

    sample_times  = [0 for i in range(num_epoch * num_step)]
    # copy_times    = [0 for i in range(num_epoch * num_step)]
    # convert_times = [0 for i in range(num_epoch * num_step)]
    # train_times   = [0 for i in range(num_epoch * num_step)]
    # total_times   = [0 for i in range(num_epoch * num_step)]
    # num_nodes     = [0 for i in range(num_epoch * num_step)]
    num_samples = [0 for i in range(num_epoch * num_step)]
    sample_kernel_times = [0 for i in range(num_epoch * num_step)]

    cur_step_key = 0
    for epoch in range(num_epoch):
        tic = time.time()
        for step in range(num_step):
            t0 = time.time()
            sam.trace_step_begin_now(
                epoch * num_step + step, sam.kL0Event_Train_Step)
            if not run_config['pipeline']:
                sam.sample_once()
            elif epoch + step == 0:
                sam.start()
            batch_key = sam.get_next_batch()
            t1 = time.time()
            sam.trace_step_begin_now(batch_key, sam.kL1Event_Convert)
            subgraph, batch_input, batch_label = sam.get_dgl_subgraph(batch_key, num_layer)
            t2 = time.time()
            sam.trace_step_end_now(batch_key, sam.kL1Event_Convert)

            # print("before train", ",".join([str(block.formats()) for block in subgraph]))

            # Compute loss and prediction
            sam.trace_step_begin_now(batch_key, sam.kL1Event_Train)
            batch_pred = model(subgraph, batch_input)
            # print("in     train", ",".join([str(block.formats()) for block in subgraph]))
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # wait for the train finish then we can free the data safely
            event_sync()

            # print("after  train", ",".join([str(block.formats()) for block in subgraph]))

            num_sample = subgraph.num_edges()

            batch_input = None
            batch_label = None
            subgraph = None

            sam.trace_step_end_now(batch_key, sam.kL1Event_Train)
            t3 = time.time()
            sam.trace_step_end_now(
                epoch * num_step + step, sam.kL0Event_Train_Step)

            sample_time = sam.get_log_step_value(epoch, step, sam.kLogL1SampleTime)
            # copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            sample_kernel_time = sam.get_log_step_value(epoch, step, sam.kLogL3KHopSampleCooTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t0

            # num_node = sam.get_log_step_value(epoch, step, sam.kLogL1NumNode)
            # num_sample = sam.get_log_step_value(epoch, step, sam.kLogL1NumSample)

            sam.log_step(epoch, step, sam.kLogL1TrainTime,   train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime,   train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime,   total_time)

            sample_times  [cur_step_key] = sample_time
            # copy_times    [cur_step_key] = copy_time
            # convert_times [cur_step_key] = convert_time
            # train_times   [cur_step_key] = train_time
            # total_times   [cur_step_key] = total_time
            sample_kernel_times[cur_step_key] = sample_kernel_time

            # num_samples.append(num_sample)
            # num_nodes     [cur_step_key] = num_node
            num_samples[cur_step_key] = num_sample

            # print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
            #     epoch, step, num_node, num_sample, total_time,
            #         sample_time, copy_time, train_time, convert_time, loss
            # ))

            # sam.report_step_average(epoch, step)
            # sam.report_step(epoch, step)
            cur_step_key += 1

        toc = time.time()
        feat_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochFeatureBytes)
        miss_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochMissBytes)
        epoch_miss_nbytes.append(miss_nbytes)
        epoch_cache_hit_rates.append(
            (feat_nbytes - miss_nbytes) / feat_nbytes)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        )
        epoch_total_times_python.append(toc - tic)
        epoch_sample_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochSampleTime)
        epoch_copy_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochCopyTime)
        epoch_convert_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochConvertTime)
        epoch_train_times[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochTrainTime)
        epoch_train_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        epoch_total_times_profiler[epoch] = sam.get_log_epoch_value(
            epoch, sam.kLogEpochTotalTime)
        sam.forward_barrier()
        print('Epoch {:05d} | Time {:.4f}'.format(
            epoch, epoch_total_times_python[-1]))

    sam.report_step_average(num_epoch - 1, num_step - 1)
    print('[Avg] Epoch Time {:.4f} | Epoch Time(Profiler) {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times_python[1:]), np.mean(epoch_total_times_profiler[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

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
        ('cache_percentage', run_config['cache_percentage']))
    test_result.append(('cache_hit_rate', np.mean(
        epoch_cache_hit_rates[1:])))
    test_result.append(
        ('epoch_time:total', np.mean(epoch_total_times_python[1:])))
    test_result.append(('step_sample_time', np.mean(sample_times)))
    test_result.append(('sample_kernel_time', np.mean(sample_kernel_times)))
    test_result.append(('epoch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])))
    test_result.append(('batch_miss_nbytes', np.mean(epoch_miss_nbytes[1:])/num_step))
    test_result.append(('batch_copy_time', np.mean(epoch_copy_times[1:])/num_step))
    for k, v in test_result:
        print('test_result:{:}={:.4f}'.format(k, v))

    sam.report_init()

    sam.report_node_access()
    sam.dump_trace()
    sam.shutdown()


if __name__ == '__main__':
    run()
