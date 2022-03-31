import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
from runner_helper2 import *

def get_dgl_logtable():
    return LogTable(
        num_row=8,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        devices='0',
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        devices='0 1',
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        devices='0 1 2',
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        devices='0 1 2 3',
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        devices='0 1 2 3 4',
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        devices='0 1 2 3 4 5',
    ).update_row_definition(
        row_id=6,
        col_range=[0, 0],
        devices='0 1 2 3 4 5 6',
    ).update_row_definition(
        row_id=7,
        col_range=[0, 0],
        devices='0 1 2 3 4 5 6 7',
    ).create()


def get_fgnn_logtable():
    return LogTable(
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


def get_sgnn_logtable():
    return LogTable(
        num_row=8,
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
