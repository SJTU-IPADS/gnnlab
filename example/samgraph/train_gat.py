import torch as th
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GATConv
import time
import torch.nn.functional as F
import numpy as np
import samgraph.torch as sam


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, blocks, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](blocks[l], h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        return logits


def get_run_config():
    run_config = {}

    run_config['type'] = 'gpu'
    run_config['cpu_hashtable_type'] = sam.simple_hashtable()
    # run_config['cpu_hashtable_type'] = sam.parallel_hashtable()
    run_config['pipeline'] = False
    # run_config['pipeline'] = True
    run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    if run_config['type'] == 'cpu':
        run_config['sampler_ctx'] = sam.cpu()
        run_config['trainer_ctx'] = sam.gpu(0)
    else:
        run_config['sampler_ctx'] = sam.gpu(1)
        run_config['trainer_ctx'] = sam.gpu(0)

    run_config['fanout'] = [10, 5]
    run_config['num_fanout'] = run_config['num_layers'] = len(
        run_config['fanout'])
    run_config['num_epoch'] = 20
    run_config['num_heads'] = 8
    run_config['num_out_heads'] = 1
    run_config['num_hidden'] = 32
    run_config['residual'] = False
    run_config['in_drop'] = 0.6
    run_config['attn-drop'] = 0.6
    run_config['lr'] = 0.005
    run_config['weight_decay'] = 5e-4
    run_config['negative_slope'] = 0.2
    run_config['batch_size'] = 8192
    run_config['report_per_count'] = 1

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.init()

    train_device = th.device('cuda:%d' % run_config['trainer_ctx'].device_id)
    in_feats = sam.feat_dim()
    n_classes = sam.num_class()
    num_layers = run_config['num_layers']

    heads = ([run_config['num_heads']] * run_config['num_layers']) + \
        [run_config['num_out_heads']]
    model = GAT(run_config['num_layers'], in_feats, run_config['num_hidden'], n_classes, heads, F.elu,
                run_config['in_drop'], run_config['attn-drop'], run_config['negative_slope'], run_config['residual'])
    model = model.to(train_device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(
    ), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
    num_epoch = run_config['num_epoch']

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    if run_config['pipeline']:
        sam.start()

    model.train()

    for epoch in range(num_epoch):
        sample_times = []
        convert_times = []
        train_times = []
        total_times = []
        num_samples = []

        for step in range(num_step):
            t0 = time.time()
            if not run_config['pipeline']:
                sam.sample()
            batch_key = sam.get_next_batch(epoch, step)
            t1 = time.time()
            blocks, batch_input, batch_label = sam.get_dgl_blocks(
                batch_key, num_layers)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            sample_times.append(t1 - t0)
            convert_times.append(t2 - t1)
            train_times.append(t3 - t2)
            total_times.append(t3 - t0)

            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)

            print('Epoch {:05d} | Step {:05d} | Samples {:.0f} | Time {:.4f} secs | Sample + Copy Time {:.4f} secs | Convert Time {:.4f} secs |  Train Time {:.4f} secs | Loss {:.4f} '.format(
                epoch, step, np.mean(num_samples[1:]), np.mean(total_times[1:]), np.mean(
                    sample_times[1:]), np.mean(convert_times[1:]), np.mean(train_times[1:]), loss
            ))
            if step % run_config['report_per_count'] == 0:
                sam.report(epoch, step)

    sam.shutdown()


if __name__ == '__main__':
    run()
