from email import message
from logtable_def import get_dgl_logtable, get_sgnn_logtable
from runner_helper2 import *
import argparse
import datetime
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))

MOCK = False
RERUN_TESTS = False
NUM_EPOCH = 3
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

HERE = os.path.abspath(os.path.dirname(__file__))

DGL_APP_DIR = os.path.join(HERE, '../../example/dgl/multi_gpu')
SGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def DGL_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl')
def SGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_sgnn')
def OUT_FILE(): return os.path.join(OUTPUT_DIR, 'table1.dat')


def global_config(args):
    global NUM_EPOCH, MOCK, RERUN_TESTS
    MOCK = args.mock
    NUM_EPOCH = args.num_epoch
    RERUN_TESTS = args.rerun_tests

    if RERUN_TESTS:
        out_dir = find_recent_outdir(HERE, 'output_')
        if out_dir:
            global OUTPUT_DIR, OUTPUT_DIR_SHORT
            OUTPUT_DIR = os.path.join(HERE, out_dir)
            OUTPUT_DIR_SHORT = out_dir


def dgl_motivation_test():
    logtable = get_dgl_logtable()

    configs = ConfigList(
        'DGL  Motivation Test',
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
        'num_sampling_worker',
        [24]  # Because it may race with extracting.
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
        appdir=DGL_APP_DIR,
        logdir=DGL_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=DGL_LOG_DIR()
    )

    return configs, logtable


def sgnn_motivation_test():
    logtable = get_sgnn_logtable()

    configs = ConfigList(
        'SGNN Motivation Test'
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
        'fanout',
        ['5 10 15']
    ).override(
        'BOOL_pipeline',
        ['no_pipeline'],
    ).override(
        'omp_thread_num',
        [40]
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
        [0.20, 0]  # 0.21
        # ).override(
        #         'BOOL_validate_configs',
        #         ['validate_configs']
    ).run(
        appdir=SGNN_APP_DIR,
        logdir=SGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=SGNN_LOG_DIR()
    )

    return configs, logtable


def run_table1_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')
    table_format = '{:<23s} {:>8s} {:>8s} {:>6s} {:>6s}    # {:s}\n'
    with open(OUT_FILE(), 'w') as f:
        f.write(table_format.format('GNN Systems',
                'Sample', 'Extract', 'Train', 'Total', ''))

        print(
            f'Running tests for table 1({OUTPUT_DIR_SHORT})...')
        _, dgl_logtable = dgl_motivation_test()
        _, sgnn_logtable = sgnn_motivation_test()

        print('Parsing logs...')
        for system, data, ref in zip(['DGL', ' w/ GPU-base Sampling'], dgl_logtable.data, dgl_logtable.data_refs):
            f.write(table_format.format(system, str(data[0]), str(
                data[1]), str(data[2]), str(data[3]), ' '.join(ref)))

        for system, data, ref in zip(['SGNN', ' w/ GPU-base Caching', ' w/ GPU-base Sampling', ' w/ Both'], sgnn_logtable.data, sgnn_logtable.data_refs):
            f.write(table_format.format(system, str(data[0]), str(
                data[1]), str(data[2]), str(data[3]), ' '.join(ref)))

        print('Done')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Table 1 Runner")
    argparser.add_argument('--num-epoch', type=int, default=NUM_EPOCH,
                           help='Number of epochs to run per test case')
    argparser.add_argument('--mock', action='store_true', default=MOCK,
                           help='Show the run command for each test case but not actually run it')
    argparser.add_argument(
        '--rerun-tests', action='store_true', default=RERUN_TESTS, help='Rerun the most recently tests')
    args = argparser.parse_args()

    global_config(args)
    run_table1_tests()
