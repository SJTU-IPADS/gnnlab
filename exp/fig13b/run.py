import argparse
import datetime
import os

from common import *
from common.runner2 import *
from logtable_def import *

MOCK = False
RERUN_TESTS = False
NUM_EPOCH = 3
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

HERE = os.path.abspath(os.path.dirname(__file__))
DGL_APP_DIR = os.path.join(HERE, '../../example/samgraph/sgnn_dgl')
FGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/multi_gpu')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def DGL_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl')
def FGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_fgnn')


GNUPLOT_FILE = os.path.join(HERE, 'scale-pinsage.plt')
def OUT_DATA_FILE(): return os.path.join(OUTPUT_DIR, 'fig13b.res')
def OUT_DATA_FILE_FULL(): return os.path.join(OUTPUT_DIR, 'fig13b-full.res')
def OUT_FIGURE_FILE(): return os.path.join(OUTPUT_DIR, 'fig13b.eps')


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
        'DGL  PinSAGE Scalability Test'
    ).select(
        'app',
        [App.pinsage]
    ).select(
        'dataset',
        [Dataset.papers100M]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
    ).override(
        'num_worker',
        [1, 2, 3, 4, 5, 6, 7, 8],
    ).override(
        'BOOL_pipeline',
        ['pipeline']
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
        'FGNN PinSAGE Scalability Test'
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
        [NUM_EPOCH]
    ).override(
        'omp-thread-num',
        [40]
    ).multi_combo_multi_override_list(
        'and',
        {'app': [App.pinsage]},
        [
            {'num_sample_worker': 1, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 6,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 1, 'num_train_worker': 7,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 2, 'num_train_worker': 6,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 1,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 2,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 3,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 4,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21},
            {'num_sample_worker': 3, 'num_train_worker': 5,
                'BOOL_pipeline': 'pipeline', 'cache_percentage': 0.21}
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


def run_fig13b_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')
    table_format = '{:}\t{:}\t{:}\t{:}\t{:}\n'
    table_format_full = '{:}\t{:}\t{:}\t{:}\t{:}\t# {:}\n'
    with open(OUT_DATA_FILE(), 'w') as f1, open(OUT_DATA_FILE_FULL(), 'w') as f2:
        f1.write(table_format.format('"GPUs"',
                                     '"DGL"', '"1S"', '"2S"', '"3S"'))
        f2.write(table_format_full.format('"GPUs"',
                                          '"DGL"', '"1S"', '"2S"', '"3S"', '""'))

        print(f'Running tests for fig 13b({OUTPUT_DIR_SHORT})...')
        _, dgl_logtable = dgl_scalability_test()
        _, fgnn_logtable = fgnn_scalability_test()

        print('Parsing logs...')
        gpus = [1, 2, 3, 4, 5, 6, 7, 8]
        dgl_data = [data[0] for data in dgl_logtable.data]
        fgnn_1s_data = ['-'] + [fgnn_logtable.data[i][0] for i in range(0, 7)]
        fgnn_2s_data = ['-', '-'] + [fgnn_logtable.data[i][0]
                                     for i in range(7, 13)]
        fgnn_3s_data = ['-', '-', '-'] + [fgnn_logtable.data[i][0]
                                          for i in range(13, 18)]

        data_refs = [[] for _ in range(8)]
        for i in range(8):
            data_refs[i] += list(dgl_logtable.data_refs[i])
        for i in range(0, 7):
            data_refs[i + 1] += list(fgnn_logtable.data_refs[i])
        for i in range(7, 13):
            data_refs[i - 7 + 2] += list(fgnn_logtable.data_refs[i])
        for i in range(13, 18):
            data_refs[i - 13 + 3] += list(fgnn_logtable.data_refs[i])

        for i in range(8):
            f1.write(table_format.format(str(gpus[i]), str(dgl_data[i]), str(
                fgnn_1s_data[i]), str(fgnn_2s_data[i]), str(fgnn_3s_data[i])))
            f2.write(table_format_full.format(str(gpus[i]), str(dgl_data[i]), str(
                fgnn_1s_data[i]), str(fgnn_2s_data[i]), str(fgnn_3s_data[i]), ' '.join(data_refs[i])))

    print('Ploting...')
    os.system(
        f'gnuplot -e "outfile=\'{OUT_FIGURE_FILE()}\';resfile=\'{OUT_DATA_FILE()}\'" {GNUPLOT_FILE}')

    print('Done')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Fig 13b Tests Runner")
    argparser.add_argument('--num-epoch', type=int, default=NUM_EPOCH)
    argparser.add_argument('--mock', action='store_true', default=MOCK)
    argparser.add_argument(
        '--rerun-tests', action='store_true', default=RERUN_TESTS)
    args = argparser.parse_args()

    global_config(args)
    run_fig13b_tests()
