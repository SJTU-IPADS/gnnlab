import argparse
import datetime
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
from runner_helper2 import *
from logtable_def import *

MOCK = False
RERUN_TESTS = False
NUM_EPOCH = 3
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

HERE = os.path.abspath(os.path.dirname(__file__))
DGL_APP_DIR = os.path.join(HERE, '../../example/dgl/multi_gpu')
FGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/multi_gpu')
SGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/sgnn')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def DGL_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl')
def FGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_fgnn')
def SGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_sgnn')


GNUPLOT_FILE = os.path.join(HERE, 'scale-gcn.plt')
def OUT_DATA_FILE(): return os.path.join(OUTPUT_DIR, 'fig14a.res')
def OUT_DATA_FILE_FULL(): return os.path.join(OUTPUT_DIR, 'fig14a-full.res')
def OUT_FIGURE_FILE(): return os.path.join(OUTPUT_DIR, 'fig14a.eps')


def global_config(args):
    global NUM_EPOCH, MOCK
    MOCK = args.mock
    NUM_EPOCH = args.num_epoch
    RERUN_TESTS = args.rerun_tests

    if RERUN_TESTS:
        out_dir = find_recent_outdir(HERE, 'output_')
        if out_dir:
            global OUTPUT_DIR, OUTPUT_DIR_SHORT
            OUTPUT_DIR = os.path.join(HERE, out_dir)
            OUTPUT_DIR_SHORT = out_dir


def dgl_scalability_test():
    logtable = get_dgl_logtable()

    configs = ConfigList(
        'DGL  GCN Scalability Test'
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
        [NUM_EPOCH]
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
        appdir=DGL_APP_DIR,
        logdir=DGL_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=DGL_LOG_DIR()
    )

    return configs, logtable


def fgnn_scalability_test():
    logtable = get_fgnn_logtable()

    configs = ConfigList(
        'FGNN GCN Scalability Test'
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
        [NUM_EPOCH]
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
        {'app': [App.gcn]},
        [
            {'num_sample_worker': 1, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 6,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 7,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 6,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.20},
        ]
    ).run(
        appdir=FGNN_APP_DIR,
        logdir=FGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=FGNN_LOG_DIR()
    )

    return configs, logtable



def sgnn_scalability_test():
    logtable = get_sgnn_logtable()

    configs = ConfigList(
        'SGNN GCN scalability test'
    ).select(
        'app',
        [App.gcn]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
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
        appdir=SGNN_APP_DIR,
        logdir=SGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=SGNN_LOG_DIR()
    )

    return configs, logtable


def run_fig14a_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')
    table_format = '{:}\t{:}\t{:}\t{:}\t{:}\n'
    table_format_full = '{:}\t{:}\t{:}\t{:}\t{:}\t# {:}\n'
    with open(OUT_DATA_FILE(), 'w') as f1, open(OUT_DATA_FILE_FULL(), 'w') as f2:
        f1.write(table_format.format('"GPUs"',
                                     '"DGL"', '"1S"', '"2S"', '"3S"'))
        f2.write(table_format_full.format('"GPUs"',
                '"DGL"', '"1S"', '"2S"', '"3S"', '""'))

        print(f'Running tests for fig 14a({OUTPUT_DIR_SHORT})...')
        _, dgl_logtable = dgl_scalability_test()
        _, sgnn_logtable = sgnn_scalability_test()
        _, fgnn_logtable = fgnn_scalability_test()

        print('Parsing logs...')
        gpus = [1, 2, 3, 4, 5, 6, 7, 8]
        dgl_data = [data[0] for data in dgl_logtable.data]
        sgnn_data = [data[0] for data in sgnn_logtable.data]
        fgnn_1s_data = ['-'] + [fgnn_logtable.data[i][0] for i in range(0, 7)]
        fgnn_2s_data = ['-', '-'] + [fgnn_logtable.data[i][0]
                                     for i in range(7, 13)]
        fgnn_3s_data = ['-', '-', '-'] + [fgnn_logtable.data[i][0]
                                          for i in range(13, 18)]

        data_refs = [[] for _ in range(8)]
        for i in range(8):
            data_refs[i] += list(dgl_logtable.data_refs[i])
        for i in range(8):
            data_refs[i] += list(sgnn_logtable.data_refs[i])
        for i in range(0, 7):
            data_refs[i + 1] += list(fgnn_logtable.data_refs[i])
        for i in range(7, 13):
            data_refs[i - 7 + 2] += list(fgnn_logtable.data_refs[i])
        for i in range(13, 18):
            data_refs[i - 13 + 3] += list(fgnn_logtable.data_refs[i])

        for i in range(8):
            f1.write(table_format.format(str(gpus[i]), str(dgl_data[i]), str(sgnn_data[i]), str(
                fgnn_1s_data[i]), str(fgnn_2s_data[i]), str(fgnn_3s_data[i])))
            f2.write(table_format.format(str(gpus[i]), str(dgl_data[i]), str(sgnn_data[i]), str(
                fgnn_1s_data[i]), str(fgnn_2s_data[i]), str(fgnn_3s_data[i]), ' '.join(data_refs[i])))

    print('Ploting...')
    os.system(
        f'gnuplot -e "outfile=\'{OUT_FIGURE_FILE()}\';resfile=\'{OUT_DATA_FILE()}\'" {GNUPLOT_FILE}')

    print('Done')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Fig 14a Tests Runner")
    argparser.add_argument('--num-epoch', type=int, default=NUM_EPOCH,
                           help='Number of epochs to run per test case')
    argparser.add_argument('--mock', action='store_true', default=MOCK,
                           help='Show the run command for each test case but not actually run it')
    argparser.add_argument(
        '--rerun-tests', action='store_true', default=RERUN_TESTS, help='Rerun the most recently tests')
    args = argparser.parse_args()

    global_config(args)
    run_fig14a_tests()
