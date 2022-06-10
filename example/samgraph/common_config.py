import samgraph.common as sam
import time
import os
from enum import Enum

__all__ = ['event_sync', 'get_default_timeout', 'get_dataset_list', 'get_default_common_config', 'add_common_arguments', 'process_common_config', 'print_run_config', 'RunMode']

class RunMode(Enum):
    NORMAL = 0  # arch0, arch1, arch2, arch3, arch4, for applications in example/samgraph
    FGNN = 1  # arch5, for applications in example/samgraph/multi_gpu and example/samgraph/balance_switcher
    SGNN = 2  # arch6, for applications in example/samgraph/sgnn
    SGNN_DGL = 3  # arch7, for applications in example/samgraph/sgnn_dgl

def _empty_func():
    pass
def import_torch_once():
    global torch
    import torch
    global import_torch_func_ptr
    import_torch_func_ptr = _empty_func
import_torch_func_ptr = import_torch_once

def event_sync():
    import_torch_func_ptr()
    event = torch.cuda.Event(blocking=True)
    event.record()
    event.synchronize()


def get_default_timeout():
    # in seconds
    return 300.0


def get_dataset_list():
    return ['papers100M', 'com-friendster',
            'reddit', 'products', 'twitter', 'uk-2006-05', 'papers100M_empty',
            'papers100M_300', 'papers100M_600',
            'papers100M-undir',
            'mag240m-homo',
            'ppi']


def get_default_common_config(run_mode: RunMode = RunMode.NORMAL, **kwargs):
    default_common_config = {}

    default_common_config['_run_mode'] = run_mode

    if run_mode == RunMode.FGNN:
        default_common_config['arch'] = 'arch5'
        default_common_config['num_train_worker'] = 1
        default_common_config['num_sample_worker'] = 1
    elif run_mode == RunMode.SGNN:
        default_common_config['arch'] = 'arch6'
        default_common_config['num_worker'] = 1
    elif run_mode == RunMode.SGNN_DGL:
        default_common_config['arch'] = 'arch7'
        default_common_config['num_worker'] = 1
    else:
        default_common_config['arch'] = 'arch3'

    default_common_config['sample_type'] = 'khop0'
    default_common_config['root_path'] = '/graph-learning/samgraph/'
    default_common_config['dataset'] = 'reddit'
    # default_common_config['dataset'] = 'products'
    # default_common_config['dataset'] = 'papers100M'
    # default_common_config['dataset'] = 'com-friendster'
    default_common_config['pipeline'] = False

    # default_common_config['cache_policy'] = 'heuristic'
    default_common_config['cache_policy'] = 'pre_sample'
    default_common_config['cache_percentage'] = 0.0

    default_common_config['num_epoch'] = 10
    default_common_config['batch_size'] = 8000
    default_common_config['num_hidden'] = 256

    default_common_config['max_sampling_jobs'] = 10
    default_common_config['max_copying_jobs'] = 1

    default_common_config['barriered_epoch'] = 0
    default_common_config['presample_epoch'] = 1
    # 40 is faster than 80 in aliyun machine
    default_common_config['omp_thread_num'] = 40

    default_common_config.update(kwargs)

    return default_common_config


def add_common_arguments(argparser, run_config):
    run_mode = run_config['_run_mode']

    if run_mode == RunMode.FGNN or run_config['arch'] == 'arch5':
        run_config['arch'] = 'arch5'
        run_config['_run_mode'] = RunMode.FGNN
        argparser.add_argument('--num-train-worker', type=int,
                               default=run_config['num_train_worker'])
        argparser.add_argument('--num-sample-worker', type=int,
                               default=run_config['num_sample_worker'])
        argparser.add_argument('--single-gpu', action='store_true', default=False)
    elif run_mode == RunMode.SGNN or run_config['arch'] == 'arch6':
        run_config['arch'] = 'arch6'
        run_config['_run_mode'] = RunMode.SGNN
        argparser.add_argument('--num-worker', type=int,
                               default=run_config['num_worker'])
    elif run_mode == RunMode.SGNN_DGL or run_config['arch'] == 'arch7':
        run_config['arch'] = 'arch7'
        run_config['_run_mode'] = RunMode.SGNN_DGL
        argparser.add_argument('--num-worker', type=int,
                               default=run_config['num_worker'])
    else:
        argparser.add_argument('--arch', type=str, choices=sam.builtin_archs.keys(),
                               default=run_config['arch'])
        run_config['_run_multi_gpu'] = 'false'
        argparser.add_argument('--override-device',
                               action='store_true', default=False)
        argparser.add_argument('--override-train-device',
                               type=str, default='cuda:0')
        argparser.add_argument('--override-sample-device',
                               type=str, default='cuda:1')

    argparser.add_argument('--sample-type', type=str, choices=sam.sample_types.keys(),
                           default=run_config['sample_type'])

    argparser.add_argument('--pipeline', action='store_true',
                            default=run_config['pipeline'])
    argparser.add_argument('--no-pipeline', action='store_false', dest='pipeline',
                            default=run_config['pipeline'])

    argparser.add_argument('--root-path', type=str,
                           default=run_config['root_path'])
    argparser.add_argument('--dataset', type=str, choices=get_dataset_list(),
                           default=run_config['dataset'])

    argparser.add_argument('--cache-policy', type=str, choices=sam.cache_policies.keys(),
                           default=run_config['cache_policy'])
    argparser.add_argument('--cache-percentage', type=float,
                           default=run_config['cache_percentage'])
    argparser.add_argument('--max-sampling-jobs', type=int,
                           default=run_config['max_sampling_jobs'])
    argparser.add_argument('--max-copying-jobs', type=int,
                           default=run_config['max_copying_jobs'])

    argparser.add_argument('--num-epoch', type=int,
                           default=run_config['num_epoch'])
    argparser.add_argument('--batch-size', type=int,
                           default=run_config['batch_size'])
    argparser.add_argument('--num-hidden', type=int,
                           default=run_config['num_hidden'])

    argparser.add_argument('--barriered-epoch', type=int,
                           default=run_config['barriered_epoch'])
    argparser.add_argument('--presample-epoch', type=int,
                           default=run_config['presample_epoch'])
    argparser.add_argument('--omp-thread-num', type=int,
                           default=run_config['omp_thread_num'])

    argparser.add_argument('--validate-configs',
                           action='store_true', default=False)

    argparser.add_argument('-ll', '--log-level', choices=[
                           'info', 'debug', 'warn', 'error'], type=str, dest='_log_level', default='error')
    argparser.add_argument('-pl', '--profile-level', choices=[
                           '0', '1', '2', '3'], type=str, dest='_profile_level', default='0')
    argparser.add_argument('--empty-feat', type=str,
                           dest='_empty_feat', default='0')


def process_common_config(run_config):
    run_config['dataset_path'] = run_config['root_path'] + \
        run_config['dataset']

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1

    run_mode = run_config['_run_mode']
    arch = run_config['arch']
    run_config['_arch'] = sam.builtin_archs[arch]['arch']
    if run_mode == RunMode.NORMAL:
        run_config['sampler_ctx'] = sam.builtin_archs[arch]['sampler_ctx']
        run_config['trainer_ctx'] = sam.builtin_archs[arch]['trainer_ctx']

        if run_config['override_device']:
            run_config['sampler_ctx'] = run_config['override_sample_device']
            run_config['trainer_ctx'] = run_config['override_train_device']
    elif run_mode == RunMode.FGNN:
        import_torch_func_ptr()
        run_config['omp_thread_num'] //= run_config['num_train_worker']
        run_config['torch_thread_num'] = torch.get_num_threads() // run_config['num_train_worker']
        assert(
            'num_sample_worker' in run_config and run_config['num_sample_worker'] > 0)
        assert(
            'num_train_worker' in run_config and run_config['num_train_worker'] > 0)
        # trainer gpu id should start from 0 under heterogeneous environment or NCCL will throw an error
        run_config['train_workers'] = [
            sam.gpu(i) for i in range(run_config['num_train_worker'])]
        run_config['sample_workers'] = [sam.gpu(
            run_config['num_train_worker'] + i) for i in range(run_config['num_sample_worker'])]
        if (run_config['single_gpu'] == True):
            run_config['num_sample_worker'] = 1
            run_config['num_train_worker'] = 1
            run_config['train_workers'] = [sam.gpu(0)]
            run_config['sample_workers'] = [sam.gpu(0)]
            run_config['pipeline'] = False
    elif run_mode == RunMode.SGNN:
        run_config['omp_thread_num'] //= run_config['num_worker']
        run_config['workers'] = [sam.gpu(i)
                                 for i in range(run_config['num_worker'])]
    elif run_mode == RunMode.SGNN_DGL:
        import_torch_func_ptr()
        run_config['omp_thread_num'] //= run_config['num_worker']
        run_config['torch_thread_num'] = torch.get_num_threads() // run_config['num_worker']
        run_config['workers'] = [sam.gpu(i)
                                 for i in range(run_config['num_worker'])]
    else:
        assert(False)

    run_config['_sample_type'] = sam.sample_types[run_config['sample_type']]
    run_config['_cache_policy'] = sam.cache_policies[run_config['cache_policy']]
    assert(run_config['cache_percentage'] >=
           0 and run_config['cache_percentage'] <= 1.0)

    assert(run_config['max_sampling_jobs'] > 0)
    assert(run_config['max_copying_jobs'] > 0)

    # arch1 doesn't support pipelining
    if run_config['arch'] == 'arch1':
        run_config['pipeline'] = False

    os.environ['SAMGRAPH_LOG_LEVEL'] = run_config['_log_level']
    os.environ["SAMGRAPH_PROFILE_LEVEL"] = run_config['_profile_level']
    os.environ['SAMGRAPH_EMPTY_FEAT'] = run_config['_empty_feat']


def print_run_config(run_config):
    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        if not k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

    for k, v in run_config.items():
        if k.startswith('_'):
            print('config:{:}={:}'.format(k, v))
