from common import *
import datetime

here = os.path.abspath(os.path.dirname(__file__))

app_dir = os.path.join(here, '../dgl/multi_gpu')
log_dir = os.path.join(
    here, f'run-logs/logs_dgl_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

if __name__ == '__main__':
    ConfigList(
    ).select(
        'app',
        [App.gcn, App.graphsage, App.pinsage]
    ).combo(
        'app',
        [App.gcn, App.graphsage],
        'bool:use_gpu_sampling',
        ['use_gpu_sampling']
    ).override(
        'bool:pipelining',
        ['no_pipelining', 'pipelining']
    ).combo(
        'app',
        [App.pinsage],
        'bool:pipelining',
        ['pipelining', 'no_pipelining'],
    ).override(
        'devices',
        ['0 1'],
    ).override(
        'bool:validate_configs',
        ['validate_configs']
    ).run(appdir=app_dir, logdir=log_dir, mock=False)
