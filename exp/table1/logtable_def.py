"""
Log table definition
"""

from common.runner2 import LogTable


def get_dgl_logtable():
    return LogTable(
        num_row=2,
        num_col=4
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_col_definition(
        col_id=3,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 3],
        BOOL_use_gpu_sampling='no_use_gpu_sampling'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        BOOL_use_gpu_sampling='use_gpu_sampling'
    ).create()


def get_sgnn_logtable():
    return LogTable(
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
        cache_percentage=0.20
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
