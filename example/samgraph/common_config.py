import samgraph.torch as sam
import time


def get_dataset_list():
    return ['papers100M', 'com-friendster',
            'reddit', 'products', 'twitter', 'uk-2006-05']


def get_default_common_config(run_multi_gpu=False, **kwargs):
    default_common_config = {}

    default_common_config['_run_multi_gpu'] = run_multi_gpu

    if run_multi_gpu:
        default_common_config['_run_multi_gpu'] = True
        default_common_config['arch'] = 'arch5'
        default_common_config['num_train_worker'] = 1
        default_common_config['num_sample_worker'] = 1
    else:
        default_common_config['arch'] = 'arch3'

    default_common_config['sample_type'] = 'khop0'
    default_common_config['root_path'] = '/graph-learning/samgraph/'
    default_common_config['dataset'] = 'reddit'
    # default_common_config['dataset'] = 'products'
    # default_common_config['dataset'] = 'papers100M'
    # default_common_config['dataset'] = 'com-friendster'
    default_common_config['pipeline'] = False

    default_common_config['cache_policy'] = 'heuristic'
    # default_common_config['cache_policy'] = 'pre_sample'
    default_common_config['cache_percentage'] = 0.0

    default_common_config['num_epoch'] = 3
    default_common_config['batch_size'] = 8000
    default_common_config['num_hidden'] = 256

    default_common_config['max_sampling_jobs'] = 10
    default_common_config['max_copying_jobs'] = 1

    default_common_config['barriered_epoch'] = 0
    default_common_config['presample_epoch'] = 0
    default_common_config['omp_thread_num'] = 40

    default_common_config.update(kwargs)

    return default_common_config


def add_common_arguments(argparser, run_config):
    run_multi_gpu = run_config['_run_multi_gpu']

    if run_multi_gpu or run_config['arch'] == 'arch5':
        run_config['arch'] = 'arch5'
        run_config['_run_multi_gpu'] = 'true'
        assert(run_config['arch'] == 'arch5')
        argparser.add_argument('--num-train-worker', type=int,
                               default=run_config['num_train_worker'])
        argparser.add_argument('--num-sample-worker', type=int,
                               default=run_config['num_sample_worker'])
    else:
        argparser.add_argument('--arch', type=str, choices=sam.builtin_archs.keys(),
                               default=run_config['arch'])
        run_config['_run_multi_gpu'] = 'false'

    argparser.add_argument('--sample-type', type=str, choices=sam.sample_types.keys(),
                           default=run_config['sample_type'])

    argparser.add_argument('--pipeline', action='store_true',
                           default=run_config['pipeline'])
    argparser.add_argument('--no-pipeline', action='store_false', dest='pipelining',
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


def process_common_config(run_config):
    run_config['dataset_path'] = run_config['root_path'] + \
        run_config['dataset']

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1

    arch = run_config['arch']
    run_config['_arch'] = sam.builtin_archs[arch]['arch']
    if run_config['arch'] != 'arch5':
        run_config['sampler_ctx'] = sam.builtin_archs[arch]['sampler_ctx']
        run_config['trainer_ctx'] = sam.builtin_archs[arch]['trainer_ctx']
    else:
        assert(
            'num_sample_worker' in run_config and run_config['num_sample_worker'] > 0)
        assert(
            'num_train_worker' in run_config and run_config['num_train_worker'] > 0)
        run_config['sample_workers'] = [
            sam.gpu(i) for i in range(run_config['num_sample_worker'])]
        run_config['train_workers'] = [sam.gpu(
            run_config['num_sample_worker'] + i) for i in range(run_config['num_train_worker'])]

    run_config['_sample_type'] = sam.sample_types[run_config['sample_type']]
    run_config['_cache_policy'] = sam.cache_policies[run_config['cache_policy']]
    assert(run_config['cache_percentage'] >=
           0 and run_config['cache_percentage'] <= 100)

    assert(run_config['max_sampling_jobs'] > 0)
    assert(run_config['max_copying_jobs'] > 0)

    # arch1 doesn't support pipelining
    if run_config['arch'] == 'arch1':
        run_config['pipeline'] = False


def print_run_config(run_config):
    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        if not k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

    for k, v in run_config.items():
        if k.startswith('_'):
            print('config:{:}={:}'.format(k, v))
