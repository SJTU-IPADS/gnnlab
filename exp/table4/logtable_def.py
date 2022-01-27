from common import *
from common.runner2 import LogTable


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
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).create()


def get_dgl_pinsage_logtable():
    return LogTable(
        num_row=4,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:total'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        dataset=Dataset.uk_2006_05
    ).create()


def get_pyg_logtable():
    return LogTable(
        num_row=8,
        num_col=1
    ).update_col_definition(
        col_id=0,
        definition='epoch_time'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 0],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 0],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).create()


def get_sgnn_logtable():
    return LogTable(
        num_row=12,
        num_col=2
    ).update_col_definition(
        col_id=0,
        definition='epoch_time:total'
    ).update_col_definition(
        col_id=1,
        definition='cache_percentage'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=6,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=7,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=8,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()


def get_fgnn_logtable():
    return LogTable(
        num_row=12,
        num_col=2
    ).update_col_definition(
        col_id=0,
        definition='pipeline_train_epoch_time'
    ).update_col_definition(
        col_id=1,
        definition='cache_percentage'
    ).update_row_definition(
        row_id=0,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=2,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=3,
        col_range=[0, 1],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=4,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.twitter,
    ).update_row_definition(
        row_id=6,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.papers100M,
    ).update_row_definition(
        row_id=7,
        col_range=[0, 1],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05,
    ).update_row_definition(
        row_id=8,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=10,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=11,
        col_range=[0, 1],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).create()
