from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../pyg/multi_gpu')

"""
    if log_dir is not None, it will only parse logs
"""


def overall_perf_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_pyg_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

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
        app=App.gcn,
        dataset=Dataset.products,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.twitter,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.papers100M,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.products,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.twitter,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.papers100M,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=0,
        col_range=[3, 3],
        app=App.gcn,
        dataset=Dataset.products,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=1,
        col_range=[3, 3],
        app=App.gcn,
        dataset=Dataset.twitter,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=2,
        col_range=[3, 3],
        app=App.gcn,
        dataset=Dataset.papers100M,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=3,
        col_range=[3, 3],
        app=App.gcn,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=4,
        col_range=[3, 3],
        app=App.graphsage,
        dataset=Dataset.products,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=5,
        col_range=[3, 3],
        app=App.graphsage,
        dataset=Dataset.twitter,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=6,
        col_range=[3, 3],
        app=App.graphsage,
        dataset=Dataset.papers100M,
        BOOL_pipelining='pipelining'
    ).update_row_definition(
        row_id=7,
        col_range=[3, 3],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='pipelining'
    ).create()

    ConfigList(
        test_group_name='PyG overall performance test'
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'BOOL_pipelining',
        ['pipelining', 'no_pipelining']
    ).override(
        'num_sampling_worker',
        [0],
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7']
    # ).override(
    #         'BOOL_validate_configs',
    #         ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()
    print(
        'PyG overall performance test uses {:.4f} secs'.format(toc - tic))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("SGNN DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    argparser.add_argument('-m', '--mock', action='store_true', default=False)
    args = argparser.parse_args()

    overall_perf_test(args.log_folder, args.mock)
