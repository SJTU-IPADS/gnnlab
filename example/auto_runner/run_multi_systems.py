from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
dgl_app_dir = os.path.join(here, '../dgl')
samgraph_app_dir = os.path.join(here, '../samgraph')


"""
    if log_dir is not None, it will only parse logs
"""


def sampling_performance_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    ConfigList(
        test_group_name='DGL motivation test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'num_epoch',
        [3]
    ).override(
        'num_sampling_worker',
        [16]
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling', 'no_use_gpu_sampling']
    ).override(
        'BOOL_pipelining',
        ['no_pipelining']
    ).override(
        'devices',
        ['0'],
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=dgl_app_dir,
        logdir=log_dir,
        mock=mock
    )

    toc = time.time()

    print('motivation test uses {:.4f} secs'.format(toc - tic))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    args = argparser.parse_args()

    sampling_performance_test(args.log_folder)
