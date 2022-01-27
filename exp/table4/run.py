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

DGL_APP_DIR = os.path.join(HERE, '../../example/dgl/multi_gpu')
# Simulate PinSAGE(GPU version) on DGL
DGL_PINSAGE_APP_DIR = os.path.join(HERE, '../../example/samgraph/sgnn_dgl')
PYG_APP_DIR = os.path.join(HERE, '../../example/pyg/multi_gpu')
SGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/sgnn')
FGNN_APP_DIR = os.path.join(HERE, '../../example/samgraph/multi_gpu')

OUTPUT_DIR = os.path.join(HERE, f'output_{TIMESTAMP}')
OUTPUT_DIR_SHORT = f'output_{TIMESTAMP}'
def DGL_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl')
def DGL_PINSAGE_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_dgl_pinsage')
def PYG_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_pyg')
def SGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_sgnn')
def FGNN_LOG_DIR(): return os.path.join(OUTPUT_DIR, 'logs_fgnn')


def OUT_FILE(): return os.path.join(OUTPUT_DIR, 'table4.dat')
def OUT_FILE_FULL(): return os.path.join(OUTPUT_DIR, 'table4-full.dat')

CONFIG_NAME_FORMAT = '{:25s}'


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


def dgl_overall_test():
    logtable = get_dgl_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('DGL Overall Test')
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).select(
        'dataset',
        [Dataset.products, Dataset.papers100M, Dataset.twitter, Dataset.uk_2006_05]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7'],
    ).override(
        'BOOL_use_gpu_sampling',
        ['use_gpu_sampling']
    ).override(
        'BOOL_pipelining',
        ['pipelining']
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


def dgl_pinsage_overall_test():
    logtable = get_dgl_pinsage_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('DGL PinSAGE Overall Test')
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
        ['pipeline']
    ).override(
        'num_worker',
        [2],
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


def pyg_overall_test():
    logtable = get_pyg_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('PyG Overall Test')
    ).select(
        'app',
        [App.gcn, App.graphsage]
    ).override(
        'num_epoch',
        [NUM_EPOCH]
    ).override(
        'BOOL_pipelining',
        ['pipelining']
    ).override(
        'num_sampling_worker',
        [40],
    ).override(
        'devices',
        ['0 1 2 3 4 5 6 7']
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


def sgnn_overall_test():
    logtable = get_sgnn_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('SGNN Overall Test')
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
        {'cache_percentage': 0.04}  # 0.05 is OK, but 0.06 fails
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
        {'cache_percentage': 0.06}  # 0.07 is still OK, 0.08 fails
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
        appdir=SGNN_APP_DIR,
        logdir=SGNN_LOG_DIR(),
        mock=MOCK
    ).parse_logs(
        logtable=logtable,
        logdir=SGNN_LOG_DIR()
    )

    return configs, logtable


def fgnn_overall_test():
    logtable = get_fgnn_logtable()

    configs = ConfigList(
        CONFIG_NAME_FORMAT.format('FGNN Overall Test')
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
        {'cache_percentage': 0.22, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.gcn], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.11, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.products]},
        {'cache_percentage': 1.0, 'num_sample_worker': 4,  'num_train_worker': 4}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.papers100M]},
        {'cache_percentage': 0.24, 'num_sample_worker': 2,  'num_train_worker': 6}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.twitter]},
        {'cache_percentage': 0.31, 'num_sample_worker': 2,  'num_train_worker': 6},
    ).multi_combo_multi_override(
        'and',
        {'app': [App.graphsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.16, 'num_sample_worker': 1,  'num_train_worker': 7},
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
        {'cache_percentage': 0.25, 'num_sample_worker': 1,  'num_train_worker': 7}
    ).multi_combo_multi_override(
        'and',
        {'app': [App.pinsage], 'dataset': [Dataset.uk_2006_05]},
        {'cache_percentage': 0.09, 'num_sample_worker': 1,  'num_train_worker': 7}
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


def run_table4_tests():
    os.system(f'mkdir -p {OUTPUT_DIR}')

    table_format = '{:<16s} {:<8s} {:>6s} {:>6s} {:>6s} {:>10s}\n'
    table_format_full = '{:<16s} {:<8s} {:>6s} {:>6s} {:>6s} {:>10s}    # {:s}\n'
    with open(OUT_FILE(), 'w') as f1, open(OUT_FILE_FULL(), 'w') as f2:
        f1.write(table_format.format('GNN Models',
                'Dataset', 'DGL', 'PyG', 'SGNN', 'FGNN'))
        f2.write(table_format_full.format('GNN Models',
                'Dataset', 'DGL', 'PyG', 'SGNN', 'FGNN', ' '))

        print(
            f'Running tests for table 4({OUTPUT_DIR_SHORT})...')
        _, dgl_logtable = dgl_overall_test()
        _, dgl_pinsage_logtable = dgl_pinsage_overall_test()
        _, pyg_logtable = pyg_overall_test()
        _, sgnn_logtable = sgnn_overall_test()
        _, fgnn_logtable = fgnn_overall_test()


        print('Parsing logs...')
        num_datasets = 4
        for m_idx, model in enumerate(['GCN', 'GraphSAGE', 'PinSAGE']):
            for d_idx, dataset in enumerate(['PR', 'TW', 'PA', 'UK']):
                data_refs = [
                    ' '.join(dgl_logtable.data_refs[m_idx * num_datasets + d_idx] if model !=
                             'PinSAGE' else dgl_pinsage_logtable.data_refs[d_idx]),
                    '' if model == 'PinSAGE' else ' '.join(
                        pyg_logtable.data_refs[m_idx * num_datasets + d_idx]),
                    ' '.join(
                        sgnn_logtable.data_refs[m_idx * num_datasets + d_idx]),
                    ' '.join(
                        fgnn_logtable.data_refs[m_idx * num_datasets + d_idx])
                ]

                f1.write(table_format.format(
                    model,
                    dataset,
                    str(dgl_logtable.data[m_idx * num_datasets + d_idx][0]) if model != 'PinSAGE' else str(
                        dgl_pinsage_logtable.data[d_idx][0]),
                    'X' if model == 'PinSAGE' else str(
                        pyg_logtable.data[m_idx * num_datasets + d_idx][0]),
                    str(sgnn_logtable.data[m_idx * num_datasets + d_idx][0]),
                    '{:s}({:s}S)'.format(str(fgnn_logtable.data[m_idx * num_datasets + d_idx][0]),
                                         fgnn_logtable.data_configs[m_idx * num_datasets + d_idx][0]['num_sample_worker'])
                ))

                f2.write(table_format_full.format(
                    model,
                    dataset,
                    str(dgl_logtable.data[m_idx * num_datasets + d_idx][0]) if model != 'PinSAGE' else str(
                        dgl_pinsage_logtable.data[d_idx][0]),
                    'X' if model == 'PinSAGE' else str(
                        pyg_logtable.data[m_idx * num_datasets + d_idx][0]),
                    str(sgnn_logtable.data[m_idx * num_datasets + d_idx][0]),
                    '{:s}({:s}S)'.format(str(fgnn_logtable.data[m_idx * num_datasets + d_idx][0]),
                                         fgnn_logtable.data_configs[m_idx * num_datasets + d_idx][0]['num_sample_worker']),
                    ' '.join(data_refs)
                ))

        print('Done')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Table 4 Tests Runner")
    argparser.add_argument('--num-epoch', type=int, default=NUM_EPOCH)
    argparser.add_argument('--mock', action='store_true', default=MOCK)
    argparser.add_argument(
        '--rerun-tests', action='store_true', default=RERUN_TESTS)
    args = argparser.parse_args()

    global_config(args)
    run_table4_tests()
