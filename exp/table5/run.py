from __future__ import division
import argparse
import datetime
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
from runner_helper2 import *
from logtable_def import *

MOCK = False
RERUN_TESTS = False
NUM_EPOCH = 1
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

HERE = os.path.abspath(os.path.dirname(__file__))

DGL_APP_DIR = os.path.join(HERE, '../../example/dgl/multi_gpu')
# Simulate PinSAGE(GPU version) on DGL
DGL_PINSAGE_APP_DIR = os.path.join(HERE, '../../example/samgraph/sgnn_dgl')
PYG_APP_DIR = os.path.join(HERE, '../../example/pyg/multi_gpu')
FGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/multi_gpu')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def DGL_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl')
def DGL_PINSAGE_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl_pinsage')
def PYG_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_pyg')
def FGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_fgnn')


def OUT_FILE(): return os.path.join(OUTPUT_DIR, 'table5.dat')
def OUT_FILE_FULL(): return os.path.join(OUTPUT_DIR, 'table5-full.dat')


CONFIG_NAME_FORMAT = '{:28s}'


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


def dgl_breakdown_test():
    logtable = get_dgl_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('DGL Breakdown Test')
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
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
    ).combo(
        'app',
        [App.gcn, App.graphsage],
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
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


def dgl_pinsage_breakdown_test():
    logtable = get_dgl_pinsage_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('DGL PinSAGE Breakdown test')
    ).select(
        'app',
        [App.pinsage]
    ).select(
        'dataset',
        [Dataset.products, Dataset.papers100M, Dataset.twitter, Dataset.uk_2006_05]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
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
        appdir=DGL_PINSAGE_APP_DIR,
        logdir=DGL_PINSAGE_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=DGL_PINSAGE_LOG_DIR()
    )

    return configs, logtable


def pyg_breakdown_test():
    logtable = get_pyg_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('PyG Breakdown Test')
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
    ).override(
        'BOOL_pipelining',
        ['no_pipelining']
    ).override(
        'num_sampling_worker',
        [40],
    ).override(
        'devices',
        ['0']
        # ).override(
        #         'BOOL_validate_configs',
        #         ['validate_configs']
    ).run(
        appdir=PYG_APP_DIR,
        logdir=PYG_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=PYG_LOG_DIR()
    )

    return configs, logtable


def fgnn_breakdown_test():
    logtable = get_fgnn_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('FGNN Breakdown Test')
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
        [NUM_EPOCH]
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
        appdir=FGNN_APP_DIR,
        logdir=FGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=FGNN_LOG_DIR()
    )

    return configs, logtable


def run_table5_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')

    dotline = '-' * 134

    table_header_format = ' {:^3s} | {:^2s} | {:^6s}   {:^8s}   {:^6s} | {:^6s}   {:^8s}   {:^6s} | {:^29s}   {:^23s}   {:^6s}\n'
    table_content_format = ' {:^3s} | {:^2s} | {:^6s}   {:^8s}   {:^6s} | {:^6s}   {:^8s}   {:^6s} | {:^5s} = {:^5s} + {:^5s} + {:^5s}   {:^8s} ({:4s}%, {:4s}%)   {:^6s}\n'
    table_content_format_full = ' {:^3s} | {:^2s} | {:^6s}   {:^8s}   {:^6s} | {:^6s}   {:^8s}   {:^6s} | {:^5s} = {:^5s} + {:^5s} + {:^5s}   {:^8s} ({:4s}%, {:4s}%)   {:^6s} # {:s}\n'
    with open(OUT_FILE(), 'w') as f1, open(OUT_FILE_FULL(), 'w') as f2:
        f1.write('{:s}\n {:^8s} | {:^26s} | {:^26s} | {:^61s} \n{:s}\n'.format(
            dotline, '', 'DGL', 'PyG', 'FGNN', dotline))
        f1.write(table_header_format.format('GNN',
                                            'DS', 'Sample', 'Extract', 'Train', 'Sample', 'Extract', 'Train', 'Sample = S + M + C', 'Extract (Ratio, Hit%)', 'Train'))
        f1.write(dotline + '\n')


        f2.write('{:s}\n {:^8s} | {:^26s} | {:^26s} | {:^61s} \n{:s}\n'.format(
            dotline, '', 'DGL', 'PyG', 'FGNN', dotline))
        f2.write(table_header_format.format('GNN',
                                            'DS', 'Sample', 'Extract', 'Train', 'Sample', 'Extract', 'Train', 'Sample = S + M + C', 'Extract (Ratio, Hit%)', 'Train'))
        f2.write(dotline + '\n')
        print(
            f'Running tests for table 5({OUTPUT_DIR_SHORT})...')
        _, dgl_logtable = dgl_breakdown_test()
        _, dgl_pinsage_logtable = dgl_pinsage_breakdown_test()
        _, pyg_logtable = pyg_breakdown_test()
        _, fgnn_logtable = fgnn_breakdown_test()

        print('Parsing logs...')
        num_models = 3
        num_datasets = 4
        for m_idx, model in enumerate(['GCN', 'GSG', 'PSG']):
            for d_idx, dataset in enumerate(['PR', 'TW', 'PA', 'UK']):
                idx = m_idx * num_datasets + d_idx

                dgl_data = dgl_logtable.data + dgl_pinsage_logtable.data
                pyg_data = pyg_logtable.data if model != 'PSG' else [
                    ['X', 'X', 'X']] * (num_models * num_datasets)
                fgnn_data = fgnn_logtable.data

                data_refs = [
                    ' '.join(dgl_logtable.data_refs[m_idx * num_datasets + d_idx] if model !=
                             'PSG' else dgl_pinsage_logtable.data_refs[d_idx]),
                    '' if model == 'PSG' else ' '.join(
                        pyg_logtable.data_refs[m_idx * num_datasets + d_idx]),
                    ' '.join(
                        fgnn_logtable.data_refs[m_idx * num_datasets + d_idx])
                ]

                f1.write(table_content_format.format(
                    model,
                    dataset,
                    dgl_data[idx][0], dgl_data[idx][1], dgl_data[idx][2],
                    pyg_data[idx][0], pyg_data[idx][1], pyg_data[idx][2],
                    fgnn_data[idx][0], fgnn_data[idx][1], fgnn_data[idx][2],
                    fgnn_data[idx][3], fgnn_data[idx][4], fgnn_data[idx][5],
                    fgnn_data[idx][6], fgnn_data[idx][7],
                ))

                f2.write(table_content_format_full.format(
                    model,
                    dataset,
                    dgl_data[idx][0], dgl_data[idx][1], dgl_data[idx][2],
                    pyg_data[idx][0], pyg_data[idx][1], pyg_data[idx][2],
                    fgnn_data[idx][0], fgnn_data[idx][1], fgnn_data[idx][2],
                    fgnn_data[idx][3], fgnn_data[idx][4], fgnn_data[idx][5],
                    fgnn_data[idx][6], fgnn_data[idx][7],
                    ' '.join(data_refs)
                ))
            f1.write(dotline + '\n')
            f2.write(dotline + '\n')

        print('Done')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Table 5 Tests Runner")
    argparser.add_argument('--num-epoch', type=int, default=NUM_EPOCH)
    argparser.add_argument('--mock', action='store_true', default=MOCK)
    argparser.add_argument(
        '--rerun-tests', action='store_true', default=RERUN_TESTS)
    args = argparser.parse_args()

    global_config(args)
    run_table5_tests()
