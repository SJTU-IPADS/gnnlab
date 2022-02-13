from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../dgl/multi_gpu')

"""
    if log_dir is not None, it will only parse logs
"""


def motivation_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=2,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_col_definition(
        col_id=3,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 3],
        BOOL_use_gpu_sampling='use_gpu_sampling'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        BOOL_use_gpu_sampling='no_use_gpu_sampling'
    ).create()

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
        [10]
    ).override(
        'num_sampling_worker',
        [24] # Because it may race with extracting.
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
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()

    print('motivation test uses {:.4f} secs'.format(toc - tic))


def breakdown_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=3
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=10,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=11,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='DGL breakdown test'
    ).select(
        'app',
        [App.gcn, App.graphsage, App.pinsage]
    ).override(
        'num_epoch',
        [10]
    ).combo(
        'app',
        [App.pinsage],
        'num_sampling_worker',
        [40]
        # ).combo(
        #     'app',
        #     [App.gcn, App.graphsage],
        #     'num_sampling_worker',
        #     [8]
    ).combo(
        'app',
        [App.gcn],
        'fanout',
        ['5 10 15']
    ).combo(
        'app',
        [App.graphsage],
        'fanout',
        ['25 10']
        # ).combo(
        #     'app',
        #     [App.pinsage],
        #     'num_epoch',
        #     [1]
    ).combo(
        'app',
        [App.gcn, App.graphsage],
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
    ).override(
        'BOOL_pipelining',
        ['no_pipelining']
    ).override(
        'devices',
        ['0'],
    ).combo(
        'dataset',
        [Dataset.uk_2006_05],
        'BOOL_validate_configs',
        ['validate_configs']
    ).combo(
        'app',
        [App.pinsage],
        'BOOL_validate_configs',
        ['validate_configs']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()

    print('breakdown test uses {:.4f} secs'.format(toc - tic))


def pinsage_breakdown_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=4,
        num_col=3
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='DGL PinSAGE breakdown test'
    ).select(
        'app',
        [App.pinsage]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
    ).override(
        'BOOL_pipelining',
        ['no_pipelining']
    ).override(
        'devices',
        ['0']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()

    print('pinsage breakdown test uses {:.4f} secs'.format(toc - tic))

def scalability_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=8,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_col_definition(
        col_id=3,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        devices='0',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        devices='0 1',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        devices='0 1 2',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        devices='0 1 2 3',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        devices='0 1 2 3 4',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        devices='0 1 2 3 4 5',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        devices='0 1 2 3 4 5 6',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        devices='0 1 2 3 4 5 6 7',
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=0,
        col_range=[3, 3],
        devices='0',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=1,
        col_range=[3, 3],
        devices='0 1',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=2,
        col_range=[3, 3],
        devices='0 1 2',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=3,
        col_range=[3, 3],
        devices='0 1 2 3',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=4,
        col_range=[3, 3],
        devices='0 1 2 3 4',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=5,
        col_range=[3, 3],
        devices='0 1 2 3 4 5',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=6,
        col_range=[3, 3],
        devices='0 1 2 3 4 5 6',
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=7,
        col_range=[3, 3],
        devices='0 1 2 3 4 5 6 7',
        BOOL_pipelining='pipelining'
    ).create()

    ConfigList(
        test_group_name='DGL scalability test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'fanout',
        ['5 10 15']
    ).override(
        'num_epoch',
        [10]
    ).override(
        'devices',
        ['0', '0 1', '0 1 2', '0 1 2 3', '0 1 2 3 4',
            '0 1 2 3 4 5', '0 1 2 3 4 5 6', '0 1 2 3 4 5 6 7'],
    ).override(
        'BOOL_pipelining',
        ['pipelining', 'no_pipelining']
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir,
        left_wrap='',
        right_wrap='',
        sep='\t'
    )

    toc = time.time()
    print('scalability test uses {:.4f} secs'.format(toc - tic))


def scalability_twitter_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=8,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        devices='0',
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        devices='0 1',
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        devices='0 1 2',
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        devices='0 1 2 3',
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        devices='0 1 2 3 4',
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        devices='0 1 2 3 4 5',
    ).update_row_definition(
        row_id=6,
        col_range=[0, 0],
        devices='0 1 2 3 4 5 6',
    ).update_row_definition(
        row_id=7,
        col_range=[0, 0],
        devices='0 1 2 3 4 5 6 7',
    ).create()

    ConfigList(
        test_group_name='DGL Twitter scalability test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.twitter]
    ).override(
        'fanout',
        ['5 10 15']
    ).override(
        'num_epoch',
        [10]
    ).override(
        'devices',
        ['0', '0 1', '0 1 2', '0 1 2 3', '0 1 2 3 4',
            '0 1 2 3 4 5', '0 1 2 3 4 5 6', '0 1 2 3 4 5 6 7'],
    ).override(
        'BOOL_pipelining',
        ['pipelining']
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir,
        left_wrap='',
        right_wrap='',
        sep='\t'
    )

    toc = time.time()
    print('scalability twitter test uses {:.4f} secs'.format(toc - tic))


def overall_perf_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=6,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).create()

    ConfigList(
        test_group_name='DGL overall performance test'
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).select(
        'dataset',
        [Dataset.products, Dataset.papers100M, Dataset.twitter]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7'],
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).override(
        'BOOL_pipelining',
        ['pipelining']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()
    print('DGL overall performance test uses {:.4f} secs'.format(toc - tic))


def pinsage_overall_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=4,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='DGL PinSAGE overall performance test'
    ).select(
        'app',
        [App.pinsage]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7'],
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).override(
        'BOOL_pipelining',
        ['pipelining']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()
    print('DGL Pinsage overall performance test uses {:.4f} secs'.format(toc - tic))


def cpu_overall_perf_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=6,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).create()

    ConfigList(
        test_group_name='DGL CPU overall performance test'
    ).override(
        'num_epoch',
        [10]
    ).combo(
        'app',
        [App.pinsage],
        'num_epoch',
        [1]
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7'],
    ).override(
        'num_sampling_worker',
        [24]
    ).override(
        'BOOL_pipelining',
        ['pipelining']
    ).combo(
        'app',
        [App.gcn, App.graphsage],
        'BOOL_use_gpu_sampling',
        ['no_use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()
    print('DGL CPU overall performance test uses {:.4f} secs'.format(toc - tic))


def motivation2_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=1,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_col_definition(
        col_id=3,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 3],
        app=App.gcn
    ).create()

    ConfigList(
        test_group_name='DGL Motivation 2 test'
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
        'devices',
        ['0 1 2 3 4 5 6 7'],
    ).override(
        'num_sampling_worker',
        [40]
    ).override(
        'BOOL_pipelining',
        ['no_pipelining']
    ).override(
        'BOOL_use_gpu_sampling',
        ['no_use_gpu_sampling']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()
    print('DGL motivation 2 test uses {:.4f} secs'.format(toc - tic))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    argparser.add_argument('-m', '--mock', action='store_true', default=False)
    args = argparser.parse_args()

    # motivation_test(args.log_folder)
    # breakdown_test(args.log_folder)
    # scalability_test(args.log_folder, args.mock)
    # scalability_pipeline_test(args.log_folder, args.mock)
    # overall_perf_test(args.log_folder)
    # cpu_overall_perf_test(args.log_folder)
    # motivation2_test(args.log_folder, args.mock)
    # scalability_twitter_test(args.log_folder, args.mock)
    pinsage_breakdown_test(args.log_folder, args.mock)
    # pinsage_overall_test(args.log_folder, args.mock)
