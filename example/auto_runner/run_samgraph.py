from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../samgraph/multi_gpu')

"""
    if log_dir is not None, it will only parse logs
"""


def breakdown_test(log_folder=None):
    tic = time.time()

    if log_folder:
        mock = True
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        mock = False
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=5
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='convert_time'
    ).update_col_definition(
        col_id=3,
        definition='train_time'
    ).update_col_definition(
        col_id=4,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 4],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 4],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 4],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=3,
        col_range=[0, 4],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 4],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 4],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=6,
        col_range=[0, 4],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=7,
        col_range=[0, 4],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 4],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 4],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=10,
        col_range=[0, 4],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=11,
        col_range=[0, 4],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='Samgraph breakdown test'
    ).select(
        'app',
        [App.gcn, App.graphsage, App.pinsage]
    ).combo(
        'app',
        [App.gcn, App.graphsage],
        'sample_type',
        ['khop2']
    ).combo(
        'app',
        [App.pinsage],
        'sample_type',
        ['random_walk']
    ).override(
        'num_epoch',
        [3]
    ).override(
        'cache_percentage',
        [0]
    ).override(
        'omp-thread-num',
        [40]
    ).combo(
        'app',
        [App.gcn],
        'fanout',
        ['5 10 15']
    ).combo(
        'app',
        [App.gcn],
        'fanout',
        ['5 10 15']
    ).override(
        'BOOL_pipeline',
        ['no_pipeline']
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


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    args = argparser.parse_args()

    breakdown_test(args.log_folder)
