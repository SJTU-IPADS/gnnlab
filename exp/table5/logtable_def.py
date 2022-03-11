import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
from runner_helper2 import *

def get_dgl_logtable():
    return LogTable(
        num_row=8,
        num_col=3
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 2],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).create()


def get_dgl_pinsage_logtable():
    return LogTable(
        num_row=4,
        num_col=3
    ).update_col_definition(
        col_id=0,
        definition='sample_time'
    ).update_col_definition(
        col_id=1,
        definition='copy_time'
    ).update_col_definition(
        col_id=2,
        definition='train_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 2],
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 2],
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 2],
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 2],
        dataset=Dataset.uk_2006_05
    ).create()


def get_pyg_logtable():
    return LogTable(
        num_row=8,
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
        app=App.gcn,
        dataset=Dataset.products,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.twitter,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=2,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.papers100M,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=3,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=4,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.products,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=5,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.twitter,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=6,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.papers100M,
        BOOL_pipelining='no_pipelining'
    ).update_row_definition(
        row_id=7,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05,
        BOOL_pipelining='no_pipelining'
    ).create()


def get_sgnn_logtable():
    return LogTable(
        num_row=12,
        num_col=9
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
        definition='epoch_time:copy_time'
    ).update_col_definition(
        col_id=4,
        definition='cache_percentage'
    ).update_col_definition(
        col_id=5,
        definition='cache_hit_rate'
    ).update_col_definition(
        col_id=6,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=7,
        definition='train_time'
    ).update_col_definition(
        col_id=8,
        definition='convert_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 8],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 8],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 8],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 8],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 8],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 8],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 8],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 8],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 8],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 8],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 8],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 8],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()

def get_fgnn_logtable():
    return LogTable(
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
        definition='cache_percentage'
    ).update_col_definition(
        col_id=6,
        definition='cache_hit_rate'
    ).update_col_definition(
        col_id=7,
        definition='epoch_time:train_total'
    ).update_col_definition(
        col_id=8,
        definition='train_time'
    ).update_col_definition(
        col_id=9,
        definition='convert_time'
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
