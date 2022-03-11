import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
from runner_helper2 import *

def get_fgnn_logtable():
    return LogTable(
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
