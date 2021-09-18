from common import *
import datetime

here = os.path.abspath(os.path.dirname(__file__))


def breakdown_test():
    app_dir = os.path.join(here, '../dgl/multi_gpu')
    log_dir = os.path.join(
        here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    log_table = LogTable(
        num_row=12,
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
        dataset=Dataset.products
    ).update_row_definition(
        row_id=1,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=2,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=3,
        col_range=[0, 3],
        app=App.gcn,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=4,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=5,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=6,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=7,
        col_range=[0, 3],
        app=App.graphsage,
        dataset=Dataset.twitter
    ).update_row_definition(
        row_id=8,
        col_range=[0, 3],
        app=App.pinsage,
        dataset=Dataset.products
    ).update_row_definition(
        row_id=9,
        col_range=[0, 3],
        app=App.pinsage,
        dataset=Dataset.papers100M
    ).update_row_definition(
        row_id=10,
        col_range=[0, 3],
        app=App.pinsage,
        dataset=Dataset.uk_2006_05
    ).update_row_definition(
        row_id=11,
        col_range=[0, 3],
        app=App.pinsage,
        dataset=Dataset.twitter
    ).create()

    ConfigList(
    ).select(
        'app',
        [App.gcn, App.graphsage, App.pinsage]
    ).override(
        'num_epoch',
        [1]
    ).override(
        'num_sampling_worker',
        [16]
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
    ).run(
        appdir=app_dir,
        logdir=log_dir,
        mock=False
    ).parse_logs(
        logtable=log_table,
        logdir=log_dir
    )


def scalability_test():
    pass


if __name__ == '__main__':
    breakdown_test()
    scalability_test()
