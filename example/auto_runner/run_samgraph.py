from common import *
import datetime
import argparse
import time

here = os.path.abspath(os.path.dirname(__file__))
app_dir = os.path.join(here, '../samgraph/multi_gpu')

"""
    if log_dir is not None, it will only parse logs
"""


def breakdown_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=10
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:sample_total'
    ).update_col_definition(
        col_id=1,
        definition='sample_time'
    ).update_col_definition(
        col_id=2,
        definition='get_cache_miss_index_time'
    ).update_col_definition(
        col_id=3,
        definition='enqueue_samples_time'
    ).update_col_definition(
        col_id=4,
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=5,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=6,
        definition='train_time'
    ).update_col_definition(
        col_id=7,
        definition='convert_time'
    ).update_col_definition(
        col_id=8,
        definition='cache_percentage'
    ).update_col_definition(
        col_id=9,
        definition='cache_hit_rate'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 9],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 9],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 9],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 9],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 9],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 9],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 9],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 9],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 9],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 9],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 9],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 9],
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
        [10]
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
        [App.graphsage],
        'fanout',
        ['25 10']
    ).override(
        'BOOL_pipeline',
        ['no_pipeline']
    ).multi_combo(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.products]},
        'cache_percentage',
        ['1.0']
    ).multi_combo(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.papers100M]},
        'cache_percentage',
        ['0.21']
    ).multi_combo(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.twitter]},
        'cache_percentage',
        ['0.25']
    ).multi_combo(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.uk_2006_05]},
        'cache_percentage',
        ['0.14']
    ).multi_combo(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.products]},
        'cache_percentage',
        ['1.0']
    ).multi_combo(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.papers100M]},
        'cache_percentage',
        ['0.25']
    ).multi_combo(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.twitter]},
        'cache_percentage',
        ['0.32']
    ).multi_combo(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.uk_2006_05]},
        'cache_percentage',
        ['0.18']
    ).multi_combo(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.products]},
        'cache_percentage',
        ['1.0']
    ).multi_combo(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.papers100M]},
        'cache_percentage',
        ['0.22']
    ).multi_combo(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.twitter]},
        'cache_percentage',
        ['0.26']
    ).multi_combo(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.uk_2006_05]},
        'cache_percentage',
        ['0.13']
        # ).override(
        #     'BOOL_validate_configs',
        #     ['validate_configs']
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=mock
    ).parse_logs_no_output(
        logtable=log_table
    )

    with open(os.path.join(log_dir, 'test_result.txt'), 'w', encoding='utf8') as f:
        for i in range(log_table.num_row):
            f.write(
                '& {{{:s} = {:s} + {:s} + {:s}}} & {{{:s}}}~~({{{:s}{:.0f}\%}},{{{:s}{:.0f}\%}}) & {{{:s} = {:s} + {:s}}} \\\\ % {:s}\n'.format(
                    log_table.data[i][0],
                    log_table.data[i][1],
                    log_table.data[i][2],
                    log_table.data[i][3],
                    log_table.data[i][4],
                    '' if float(log_table.data[i][8]) == 1.0 else '~~',
                    float(log_table.data[i][8]) * 100,
                    '' if float(log_table.data[i][9]) == 1.0 else '~~',
                    float(log_table.data[i][9]) * 100,
                    log_table.data[i][5],
                    log_table.data[i][6],
                    log_table.data[i][7],
                    os.sep.join(
                        os.path.normpath(log_table.row_log_reference[i][0]).split(os.sep)[-2:])
                ))

    toc = time.time()

    print('breakdown test uses {:.4f} secs'.format(toc - tic))


def overall_test(log_folder=None, mock=False):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
        num_col=2
    ).update_col_definition(
        col_id=0,
        definition='pipeline_train_epoch_time'
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
        dataset=Dataset.twitter,
        num_sample_worker=2
    ).update_row_definition(
        row_id=6,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.papers100M,
        num_sample_worker=2
    ).update_row_definition(
        row_id=7,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05,
        num_sample_worker=1
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
        test_group_name='Samgraph overall test'
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
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0, 'num_sample_worker': 3,  'num_train_worker': 5}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.20, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.18, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.11, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0, 'num_sample_worker': 4,  'num_train_worker': 4}
    ).multi_combo_multi_override_list(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.papers100M]},
        [
            {'cache_percentage': 0.24, 'num_sample_worker': 2,  'num_train_worker': 6},
            # {'cache_percentage': 0.24, 'num_sample_worker': 3,  'num_train_worker': 5}
        ]
    ).multi_combo_multi_override_list(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.twitter]},
        [
            {'cache_percentage': 0.31, 'num_sample_worker': 2,  'num_train_worker': 6},
            # {'cache_percentage': 0.31, 'num_sample_worker': 3,  'num_train_worker': 5}
        ]
    ).multi_combo_multi_override_list(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.uk_2006_05]},
        [
            {'cache_percentage': 0.16, 'num_sample_worker': 1,  'num_train_worker': 7},
            # {'cache_percentage': 0.16, 'num_sample_worker': 2,  'num_train_worker': 6},
        ]
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0, 'num_sample_worker': 1,  'num_train_worker': 7}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.21, 'num_sample_worker': 1,  'num_train_worker': 7}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.23, 'num_sample_worker': 1,  'num_train_worker': 7}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.09, 'num_sample_worker': 1,  'num_train_worker': 7}
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

    print('overall test uses {:.4f} secs'.format(toc - tic))


def gcn_scalability_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=18,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:sample_total'
    ).update_col_definition(
        col_id=1,
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=2,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=3,
        definition='pipeline_train_epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=6,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=7,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=8,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=9,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=10,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=11,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=12,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=6,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=13,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=14,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=15,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=16,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=17,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=0,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=1,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=2,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=3,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=4,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=5,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=6,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=6,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=7,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=7,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=8,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=9,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=10,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=11,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=12,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=6,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=13,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=14,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=15,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=16,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=17,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).create()

    ConfigList(
        test_group_name='Samgraph GCN scalability test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'sample_type',
        ['khop2']
    ).override(
        'num_epoch',
        [10]
    ).override(
        'omp-thread-num',
        [40]
    ).combo(
        'app',
        [App.gcn],
        'fanout',
        ['5 10 15']
    ).multi_combo_multi_override_list(
        'and',
        {'app' : [App.gcn]},
        [
            {'num_sample_worker': 1, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 7, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},

            {'num_sample_worker': 1, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 6, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 7, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 6, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
        ]
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

    print('Samgraph GCN scalability test uses {:.4f} secs'.format(toc - tic))

def gcn_twitter_scalability_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=18,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='pipeline_train_epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=1
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=2
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=3
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=4
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=5
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=6
    ).update_row_definition(
        row_id=6,
        col_range=[0, 0],
        num_sample_worker=1,
        num_train_worker=7
    ).update_row_definition(
        row_id=7,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=1
    ).update_row_definition(
        row_id=8,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=2
    ).update_row_definition(
        row_id=9,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=3
    ).update_row_definition(
        row_id=10,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=4
    ).update_row_definition(
        row_id=11,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=5
    ).update_row_definition(
        row_id=12,
        col_range=[0, 0],
        num_sample_worker=2,
        num_train_worker=6
    ).update_row_definition(
        row_id=13,
        col_range=[0, 0],
        num_sample_worker=3,
        num_train_worker=1
    ).update_row_definition(
        row_id=14,
        col_range=[0, 0],
        num_sample_worker=3,
        num_train_worker=2
    ).update_row_definition(
        row_id=15,
        col_range=[0, 0],
        num_sample_worker=3,
        num_train_worker=3
    ).update_row_definition(
        row_id=16,
        col_range=[0, 0],
        num_sample_worker=3,
        num_train_worker=4
    ).update_row_definition(
        row_id=17,
        col_range=[0, 0],
        num_sample_worker=3,
        num_train_worker=5
    ).create()

    ConfigList(
        test_group_name='Samgraph GCN Twitter scalability test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.twitter]
    ).override(
        'sample_type',
        ['khop2']
    ).override(
        'num_epoch',
        [10]
    ).override(
        'omp-thread-num',
        [40]
    ).combo(
        'app',
        [App.gcn],
        'fanout',
        ['5 10 15']
    ).multi_combo_multi_override_list(
        'and',
        {'app' : [App.gcn]},
        [
            {'num_sample_worker': 1, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 7, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.18},
            {'num_sample_worker': 2, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 2, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.18},
            {'num_sample_worker': 3, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
        ]
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

    print('Samgraph GCN Twitter scalability test uses {:.4f} secs'.format(toc - tic))

def pinsage_scalability_test(log_folder, mock):
    tic = time.time()

    if log_folder:
        log_dir = os.path.join(os.path.join(here, f'run-logs/{log_folder}'))
    else:
        log_dir = os.path.join(
            here, f'run-logs/logs_samgraph_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=18,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:sample_total'
    ).update_col_definition(
        col_id=1,
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=2,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=3,
        definition='pipeline_train_epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=6,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        num_sample_worker=1,
        num_train_worker=7,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=8,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=9,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=10,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=11,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=12,
        col_range=[0, 2],
        num_sample_worker=2,
        num_train_worker=6,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=13,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=1,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=14,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=2,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=15,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=3,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=16,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=4,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=17,
        col_range=[0, 2],
        num_sample_worker=3,
        num_train_worker=5,
        BOOL_pipeline='no_pipeline'
    ).update_row_definition(
        row_id=0,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=1,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=2,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=3,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=4,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=5,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=6,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=6,
        col_range=[3, 3],
        num_sample_worker=1,
        num_train_worker=7,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=7,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=8,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=9,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=10,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=11,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=12,
        col_range=[3, 3],
        num_sample_worker=2,
        num_train_worker=6,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=13,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=1,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=14,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=2,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=15,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=3,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=16,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=4,
        BOOL_pipeline='pipeline'
    ).update_row_definition(
        row_id=17,
        col_range=[3, 3],
        num_sample_worker=3,
        num_train_worker=5,
        BOOL_pipeline='pipeline'
    ).create()

    ConfigList(
        test_group_name='Samgraph PinSAGE scalability test'
    ).select(
        'app',
        [App.pinsage]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'sample_type',
        ['random_walk']
    ).override(
        'num_epoch',
        [10]
    ).override(
        'omp-thread-num',
        [40]
    ).multi_combo_multi_override_list(
        'and',
        {'app' : [App.pinsage]},
        [
            {'num_sample_worker': 1, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 7, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 6, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 1, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 2, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 3, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 4, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 5, 'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},

            {'num_sample_worker': 1, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 6, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 7, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 6, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 1, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 2, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 3, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 4, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 5, 'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.21},
        ]
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

    print('Samgraph PinSAGE scalability test uses {:.4f} secs'.format(toc - tic))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("DGL runner")
    argparser.add_argument('-l', '--log-folder', default=None)
    argparser.add_argument('-m', '--mock', action='store_true', default=False)
    args = argparser.parse_args()

    breakdown_test(args.log_folder, args.mock)
    # overall_test(args.log_folder, args.mock)
    # gcn_scalability_test(args.log_folder, args.mock)
    # gcn_twitter_scalability_test(args.log_folder, args.mock)
    # pinsage_scalability_test(args.log_folder, args.mock)
