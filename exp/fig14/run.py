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
FGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/multi_gpu')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def FGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_fgnn')


GNUPLOT_FILE = os.path.join(HERE, 'scale-break.plt')
def OUT_DATA_FILE(): return os.path.join(OUTPUT_DIR, 'fig14.res')
def OUT_FIGURE_FILE(): return os.path.join(OUTPUT_DIR, 'fig14.eps')


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


def fgnn_scalability_breakdown_test():
    logtable = get_fgnn_logtable()

    configs = ConfigList(
        'FGNN GCN Scalability Breakdown Test'
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

            {'num_sample_worker': 1, 'num_train_worker': 1,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 2,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 3,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 4,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 5,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 6,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 1, 'num_train_worker': 7,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 1,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 2,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 3,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 4,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 5,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 2, 'num_train_worker': 6,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 1,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
            {'num_sample_worker': 3, 'num_train_worker': 2,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 3,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 4,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.19},
            {'num_sample_worker': 3, 'num_train_worker': 5,
                'BOOL_pipeline': 'no_pipeline', 'cache_percentage': 0.20},
        ]
    ).run(
        appdir=FGNN_APP_DIR,
        logdir=FGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=FGNN_LOG_DIR(),
    )

    return configs, logtable


def run_fig14_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')
    table_format = '{:}\t{:}\t{:}\t{:}\t{:}\t# {:}\n'
    with open(OUT_DATA_FILE(), 'w') as f:
        print(f'Running tests for fig 13a({OUTPUT_DIR_SHORT})...')
        _, fgnn_logtable = fgnn_scalability_breakdown_test()

        print('Parsing logs...')
        x_axis0 = ['"1S 1T"', '"1S 2T"', '"1S 3T"',
                   '"1S 4T"', '"1S 5T"', '"1S 6T"', '"1S 7T"']
        x_axis1 = ['"2S 1T"', '"2S 2T"', '"2S 3T"',
                   '"2S 4T"', '"2S 5T"', '"2S 6T"']
        x_axis2 = ['"3S 1T"', '"3S 2T"', '"3S 3T"', '"3S 4T"', '"3S 5T"']

        fgnn_data = fgnn_logtable.data
        data_refs = fgnn_logtable.data_refs

        f.write('" " ? ? ? ?\n')
        for i in range(0, 7):
            f.write(table_format.format(x_axis0[i], str(fgnn_data[i][0]), str(
                fgnn_data[i][1]), str(fgnn_data[i][2]), str(fgnn_data[i][3]), ' '.join(data_refs[i])))

        f.write('" " ? ? ? ?\n')
        for i in range(7, 13):
            f.write(table_format.format(x_axis1[i - 7], str(fgnn_data[i][0]), str(
                fgnn_data[i][1]), str(fgnn_data[i][2]), str(fgnn_data[i][3]), ' '.join(data_refs[i])))

        f.write('" " ? ? ? ?\n')
        for i in range(13, 18):
            f.write(table_format.format(x_axis2[i - 13], str(fgnn_data[i][0]), str(
                fgnn_data[i][1]), str(fgnn_data[i][2]), str(fgnn_data[i][3]), ' '.join(data_refs[i])))

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
    run_fig14_tests()
