import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ["NCCL_DEBUG"] = "INFO"
'''
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
    argparser.add_argument('--sample-type', type=int,
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
    argparser.add_argument('--num-train-worker', type=int,
                           default=default_run_config['num_train_worker'])
    argparser.add_argument('--num-sample-worker', type=int,
                           default=default_run_config['num_sample_worker'])
    argparser.add_argument('--log-file', type=str,
                           default=default_run_config['log_file'])

    return vars(argparser.parse_args())


def get_run_config():
    default_run_config = {}
    default_run_config['arch'] = 'arch5'
    default_run_config['sample_type'] = sam.kKHop2
    default_run_config['pipeline'] = False  # default value must be false
    default_run_config['dataset_path'] = '/graph-learning/samgraph/reddit'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/products'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/papers100M'
    # default_run_config['dataset_path'] = '/graph-learning/samgraph/com-friendster'

    default_run_config['cache_policy'] = sam.kCacheByHeuristic
    # default_run_config['cache_policy'] = sam.kCacheByPreSample
    default_run_config['cache_percentage'] = 0.0

    default_run_config['max_sampling_jobs'] = 10
    # default max_copying_jobs should be 10, but when training on com-friendster,
    # we have to set this to 1 to prevent GPU out-of-memory
    default_run_config['max_copying_jobs'] = 2

    # default_run_config['fanout'] = [5, 10, 15]
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['batch_size'] = 8000
    default_run_config['num_hidden'] = 256
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5
    default_run_config['weight_decay'] = 0.0005

    default_run_config['num_train_worker'] = 1
    default_run_config['num_sample_worker'] = 1
    default_run_config['log_file'] = '/tmp/samgraph_multi.log'

    run_config = parse_args(default_run_config)

    print('Evaluation time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(*run_config.items(), sep='\n')

    run_config['arch'] = sam.meepo_archs[run_config['arch']]
    run_config['arch_type'] = run_config['arch']['arch_type']
    # sampler_ctx and trainer_ctx is not useful
    run_config['sampler_ctx'] = run_config['arch']['sampler_ctx']
    run_config['trainer_ctx'] = run_config['arch']['trainer_ctx']

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    # arch1 doesn't support pipelining
    if run_config['arch_type'] == sam.kArch1:
        run_config['pipeline'] = False
    # get omp threads
    run_config['omp_num_threads'] = os.getenv('SAMGRAPH_OMP_NUM_THREADS')

    return run_config

def run_init(run_config):
    sam.config(run_config)
    sam.config_khop(run_config)
    sam.data_init()

def run_sample(worker_id, run_config, epoch_barrier):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_train_worker = run_config['num_train_worker']
    num_sample_worker = run_config['num_sample_worker']
    ctx = sam.gpu(num_train_worker + worker_id)
    print('pid of run_sample is: ', os.getpid())
    sam.sample_init(ctx.device_type, ctx.device_id, worker_id, num_sample_worker, num_train_worker)
    epoch_barrier.wait()

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()
    if (worker_id == (num_sample_worker - 1)):
        num_step = int(num_step - int(num_step / num_sample_worker * worker_id))
    else:
        num_step = int(num_step / num_sample_worker)
    # align the train_workers
    # num_step = int(int(num_step / num_train_worker) * num_train_worker)

    sample_epoch_t_l = []
    epoch_sample_times = []

    print(f"sample num_epoch: {num_epoch}, num_step: {num_step}")
    for epoch in range(num_epoch):
        if (run_config['pipeline']):
            epoch_barrier.wait()
        sample_epoch_t = time.time()
        for step in range(num_step):
            # print(f'sample epoch {epoch}, step {step}')
            sam.sample_once()
            # sam.report_step(epoch, step)
        sample_epoch_t_l.append(time.time() - sample_epoch_t)
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime))
        if (not run_config['pipeline']):
            epoch_barrier.wait()
        epoch_barrier.wait()
    print("average sample epoch time: {:.4f}".format(np.mean(sample_epoch_t_l[1:])))
    print("average sample epoch time by profiler: {:.4f}".format(np.mean(epoch_sample_times[1:])))
    # print result
    with open(run_config['log_file'], 'a+', encoding='utf8') as log_file:
        print('test_result:sample_time={:.2f}'.format(np.mean(sample_epoch_t_l[1:])), file=log_file)
    sam.shutdown()

def run_train(worker_id, run_config, epoch_barrier):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # ctx= run_config['trainer_ctx'] # [worker_id]
    ctx= sam.gpu(worker_id)
    num_worker = run_config['num_train_worker']
    train_device = torch.device('cuda:%d' % ctx.device_id)
    print(f"train_device: {train_device}, num_worker: {num_worker}, worker_id: {worker_id}, ", 'pid: ', os.getpid())
    torch.cuda.set_device(train_device)

    # let the trainer initialization after sampler
    #   sampler should presample before trainer initialization
    epoch_barrier.wait()
    sam.train_init(ctx.device_type, ctx.device_id)

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id)

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
    epoch_t_l = []

    align_up_step = int(int((num_step + num_worker - 1) / num_worker) * num_worker)

    print(f"train num_epoch: {num_epoch}, num_step: {num_step}")
    for epoch in range(num_epoch):
        epoch_barrier.wait()
        epoch_t = time.time()
        if (run_config['pipeline']):
            need_steps = int(num_step / num_worker)
            if (worker_id < num_step % num_worker):
                need_steps += 1
            sam.extract_start(need_steps)
        for step in range(worker_id, align_up_step, num_worker):
            if (step < num_step):
                t0 = time.time()
                if (not run_config['pipeline']):
                    sam.sample_once()
                # sam.sample_once()
                batch_key = sam.get_next_batch()
                t1 = time.time()
                blocks, batch_input, batch_label = sam.get_dgl_blocks(batch_key, num_layer)
                t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()

            if (step + num_worker < num_step):
                batch_input = None
                batch_label = None

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

            '''
            num_sample = 0
            for block in blocks:
                num_sample += block.num_edges()
            num_samples.append(num_sample)
            num_nodes.append(blocks[0].num_src_nodes())

            print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
                epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times[1:]), np.mean(
                    sample_times[1:]), np.mean(copy_times[1:]), np.mean(train_times[1:]), np.mean(convert_times[1:]), loss
            ))

            sam.report_step(epoch, step)
            '''

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()
        epoch_t_l.append(time.time() - epoch_t)
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
        print('Epoch {:05d} | Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f} | PID {:04d}'.format(
            epoch, epoch_total_times[-1],  epoch_sample_times[-1],  epoch_copy_times[-1], epoch_convert_times[-1], epoch_train_times[-1], os.getpid()))
        print('Epoch {:05d} | Time {:.4f}'.format(epoch, epoch_t_l[-1]))
        sam.report_epoch_average(epoch)

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))
    print('Avg Epoch Time {:.4f}'.format(np.mean(epoch_t_l[1:])))

    test_result = {}
    test_result['epoch_time'] = np.mean(epoch_t_l[1:])
    test_result['copy_time'] = np.mean(epoch_copy_times[1:])
    test_result['convert_time'] = np.mean(epoch_convert_times[1:])
    test_result['train_time'] = np.mean(epoch_train_times[1:])
    # print the results to log file
    with open(run_config['log_file'], 'a+', encoding='utf8') as log_file:
        for k,v in test_result.items():
            print('test_result:{:}={:.2f}'.format(k, v), file=log_file)

    sam.report_node_access()
    sam.shutdown()


if __name__ == '__main__':
    # clear the shared memory files of samgraph
    os.system('rm /dev/shm/shared_meta_data*')
    run_config = get_run_config()

    # print the configuration to log file
    with open(run_config['log_file'], 'a+', encoding='utf8') as log_file:
        print('config:eval_tsp="{:}"'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), file=log_file)
        for k,v in run_config.items():
            print('config:{:}={:}'.format(k, v), file=log_file)

    num_sample_worker = run_config['num_sample_worker']
    num_train_worker = run_config['num_train_worker']
    epoch_barrier = mp.Barrier(num_sample_worker + num_train_worker)

    run_init(run_config)

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
