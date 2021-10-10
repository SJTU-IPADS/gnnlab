from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../samgraph/sgnn')

"""
    if log_dir is not None, it will only parse logs
"""


def motivation_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
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
        cache_percentage=0.21
    ).update_row_definition(
        row_id=2,
        col_range=[0, 5],
        arch='arch2',
        cache_percentage=0
    ).update_row_definition(
        row_id=3,
        col_range=[0, 5],
        arch='arch2',
        cache_percentage=0.07
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
        [0.07, 0]
    ).combo(
        'arch',
        'arch0',
        'cache_percentage',
        [0.21, 0]
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


def overall_pipeline_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_sgnn_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=2
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:total'
    ).update_col_definition(
        col_id=1,
        definition='cache_percentage'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='SGNN Overall Pipeline test'
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
        [10]
    ).override(
        'omp-thread-num',
        [40]
    ).override(
        'cache-policy',
        ['degree']
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
    ).override(
        'BOOL_pipeline',
        ['pipeline']
    ).override(
        'num_worker',
        [8]
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.04} # 0.05 is OK, but 0.06 fails
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.01}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.11}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.15}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.00}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.06} # 0.07 is still OK, 0.08 fails
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.04}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.0}
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

    print('overall pipeline test uses {:.4f} secs'.format(toc - tic))

def overall_no_pipeline_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_sgnn_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=2
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:total'
    ).update_col_definition(
        col_id=1,
        definition='cache_percentage'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

    ConfigList(
        test_group_name='SGNN Overall Non-Pipeline test'
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
        [10]
    ).override(
        'omp-thread-num',
        [40]
    ).override(
        'cache-policy',
        ['degree']
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
    ).override(
        'BOOL_pipeline',
        ['no_pipeline']
    ).override(
        'num_worker',
        [8]
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.04}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.01}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.11}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.15}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.00}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.06}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.04}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.0}
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

    print('SGNN Overall Non-pipeline test uses {:.4f} secs'.format(toc - tic))

def gcn_scalability_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_sgnn_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=8,
        num_col=4
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
    ).update_row_definition(
        row_id=0,
        col_range=[0, 3],
        num_worker=1,
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        num_worker=2,
    ).update_row_definition(
        row_id=2,
        col_range=[0, 3],
        num_worker=3,
    ).update_row_definition(
        row_id=3,
        col_range=[0, 3],
        num_worker=4,
    ).update_row_definition(
        row_id=4,
        col_range=[0, 3],
        num_worker=5,
    ).update_row_definition(
        row_id=5,
        col_range=[0, 3],
        num_worker=6,
    ).update_row_definition(
        row_id=6,
        col_range=[0, 3],
        num_worker=7,
    ).update_row_definition(
        row_id=7,
        col_range=[0, 3],
        num_worker=8,

    ).create()

    ConfigList(
        test_group_name='SGNN GCN scalability test'
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
        'cache_policy',
        ['degree']
    ).override(
        'cache_percentage',
        [0.03]
    ).override(
        'num_worker',
        [1, 2, 3, 4, 5, 6, 7, 8],
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
        logdir=log_dir,
        left_wrap='',
        right_wrap='',
        sep='\t'
    )

    toc = time.time()
    print('SGNN GCN Scalability test uses {:.4f} secs'.format(toc - tic))


def pinsage_scalability_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_sgnn_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=8,
        num_col=4
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
    ).update_row_definition(
        row_id=0,
        col_range=[0, 3],
        num_worker=1,
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        num_worker=2,
    ).update_row_definition(
        row_id=2,
        col_range=[0, 3],
        num_worker=3,
    ).update_row_definition(
        row_id=3,
        col_range=[0, 3],
        num_worker=4,
    ).update_row_definition(
        row_id=4,
        col_range=[0, 3],
        num_worker=5,
    ).update_row_definition(
        row_id=5,
        col_range=[0, 3],
        num_worker=6,
    ).update_row_definition(
        row_id=6,
        col_range=[0, 3],
        num_worker=7,
    ).update_row_definition(
        row_id=7,
        col_range=[0, 3],
        num_worker=8,

    ).create()

    ConfigList(
        test_group_name='SGNN PinSAGE scalability test'
    ).select(
        'app',
        [App.pinsage]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'num_epoch',
        [10]
    ).override(
        'cache_policy',
        ['degree']
    ).override(
        'cache_percentage',
        [0.06]
    ).override(
        'num_worker',
        [1, 2, 3, 4, 5, 6, 7, 8],
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
        logdir=log_dir,
        left_wrap='',
        right_wrap='',
        sep='\t'
    )

    toc = time.time()
    print('SGNN PinSAGE Scalability test uses {:.4f} secs'.format(toc - tic))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("SGNN DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    argparser.add_argument('-m', '--mock', action='store_true', default=False)
    args = argparser.parse_args()

    # motivation_test(args.log_folder, args.mock)
    # overall_pipeline_test(args.log_folder, args.mock)
    # overall_no_pipeline_test(args.log_folder, args.mock)
    gcn_scalability_test(args.log_folder, args.mock)
    # pinsage_scalability_test(args.log_folder, args.mock)
