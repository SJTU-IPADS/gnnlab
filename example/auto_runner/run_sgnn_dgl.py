from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../samgraph/sgnn_dgl')

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
            here, f'run-logs/logs_sgnn_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=3,
        num_col=3
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:sample_time'
    ).update_col_definition(
        col_id=1,
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=2,
        definition='epoch_time:train_total'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        dataset=Dataset.twitter
    ).create()

    ConfigList(
        test_group_name='SGNN DGL breakdown test'
    ).select(
        'app',
        [App.pinsage]
    ).select(
        'dataset',
        [Dataset.products, Dataset.papers100M, Dataset.twitter]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'BOOL_pipeline',
        ['no_pipeline']
    ).override(
        'num_worker',
        [1],
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
    argparser = argparse.ArgumentParser("SGNN DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    args = argparser.parse_args()

    breakdown_test(args.log_folder)
