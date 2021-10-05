from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../samgraph/sgnn')

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
            here, f'run-logs/logs_sgnn_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=4,
        num_col=6
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:sample_time'
    ).update_col_definition(
        col_id=1,
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=2,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=3,
        definition='epoch_time:total'
    ).update_col_definition(
        col_id=4,
        definition='cache_percentage'
    ).update_col_definition(
        col_id=5,
        definition='cache_hit_rate'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 5],
        arch='arch0',
        cache_percentage=0
    ).update_row_definition(
        row_id=1,
        col_range=[0, 5],
        arch='arch0',
        cache_percentage=0.18
    ).update_row_definition(
        row_id=2,
        col_range=[0, 5],
        arch='arch2',
        cache_percentage=0
    ).update_row_definition(
        row_id=3,
        col_range=[0, 5],
        arch='arch2',
        cache_percentage=0.04
    ).create()

    ConfigList(
        test_group_name='SGNN motivation test'
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
        'fanout',
        ['5 10 15']
    ).override(
        'BOOL_pipeline',
        ['no_pipeline']
    ).override(
        'arch',
        ['arch0', 'arch2']
    ).override(
        'cache_policy',
        ['degree']
    ).combo(
        'arch',
        'arch2',
        'cache_percentage',
        [0.04, 0]
    ).combo(
        'arch',
        'arch0',
        'cache_percentage',
        [0.18, 0]
    # ).override(
    #         'BOOL_validate_configs',
    #         ['validate_configs']
    ).run(
        appdir=os.path.join(here, '../samgraph'),
        logdir=log_dir,
        mock=mock
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )

    toc = time.time()

    print('motivation test uses {:.4f} secs'.format(toc - tic))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("SGNN DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    args = argparser.parse_args()

    motivation_test(args.log_folder)
