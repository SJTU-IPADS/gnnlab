import copy
import os
import re
from enum import Enum
from typing import Tuple
from collections import defaultdict
from tqdm import tqdm

from . import *

# find the most recent output dir


def find_recent_outdir(parent, prefix):
    folders = os.listdir(parent)
    folders = [folder for folder in folders if folder.startswith(prefix)]

    if len(folders) == 0:
        return None

    folders.sort()

    return folders[-1]


class LogTable:
    def __init__(self, num_row, num_col, **kwargs):
        '''
            kwargs here are overall performance
            row definition is a dict of params that indicate a cfg {'app': App.gcn, 'batch_size' : 8000 }
            col definition is a metric, like sampling_time, copy_time in logfile
        '''
        tmp_col_0 = [{} for _ in range(num_col)]
        tmp_col_1 = [None for _ in range(num_col)]

        self.num_row = num_row
        self.num_col = num_col
        self.default_param_definition = kwargs

        self.row_definitions = [copy.deepcopy(
            tmp_col_0) for _ in range(num_row)]
        self.col_definitions = [None for _ in range(num_col)]
        self.data = [copy.deepcopy(tmp_col_1) for _ in range(num_row)]
        self.data_configs = [copy.deepcopy(tmp_col_0) for _ in range(num_row)]
        self.data_refs = [set() for _ in range(num_row)]

        self.is_finalized = False

    def update_row_definition(self, row_id, col_range: Tuple[int, int], **kwargs):
        assert(row_id < self.num_row)
        for j in range(col_range[0], col_range[1] + 1):
            assert(j < self.num_col)
            self.row_definitions[row_id][j].update(
                self.default_param_definition)
            self.row_definitions[row_id][j].update(kwargs)

        return self

    def update_col_definition(self, col_id, definition):
        assert(col_id < self.num_col)
        self.col_definitions[col_id] = definition
        return self

    def create(self):
        for i in range(self.num_col):
            assert(self.col_definitions[i] != None)

        for i in range(self.num_row):
            for j in range(self.num_col):
                assert(self.row_definitions[i][j] != None)

        self.is_finalized = True
        return self


class RunStatus(Enum):
    Ok = 0
    NotOk = 1

    def __str__(self):
        if self is RunStatus.Ok:
            return '1'
        else:
            return '0'


class RunConfig:
    def __init__(self, app: App, **kwargs):
        self.configs = {}
        self.configs['app'] = app
        self.configs.update(kwargs)

        self.std_out_log = None
        self.std_err_log = None
        self.run_idx = -1

        self.is_log_parsed = False
        self.full_configs = defaultdict(lambda: None)
        self.test_results = defaultdict(lambda: 'X')

        self.status = RunStatus.NotOk

    def form_cmd(self, idx, appdir, logdir, durable_log=True):
        cmd_line = ''
        cmd_line += 'python ' + \
            os.path.join(appdir, f'train_{self.configs["app"]}.py')

        for k, v in self.configs.items():
            if k == 'app':
                continue

            if k.startswith('BOOL_'):
                param_val = v.replace('_', '-')
                cmd_line += f' --{param_val}'
            else:
                param_key = k.replace('_', '-')
                param_val = v
                cmd_line += f' --{param_key} {param_val}'

        if durable_log:
            std_out_log = os.path.join(logdir, f'test{idx}.log')
            std_err_log = os.path.join(logdir, f'test{idx}.err.log')

            cmd_line += f' > \"{std_out_log}\"'
            cmd_line += f' 2> \"{std_err_log}\"'

            self.std_out_log = std_out_log
            self.std_err_log = std_err_log

        return cmd_line

    def run(self, idx, appdir, logdir, mock=False, durable_log=True, callback=None):
        ret = 0

        if mock:
            print(self.form_cmd(idx, appdir, logdir, durable_log))
        else:
            os.system('mkdir -p {}'.format(logdir))
            ret = os.system(self.form_cmd(idx, appdir, logdir, durable_log))
            if callback != None:
                callback(self)

        self.run_idx = idx

        return ret

    def match(self, params):
        for k, v in params.items():
            if k not in self.configs.keys() or v != self.configs[k]:
                return False

        return True

    def parse_log(self, config_pattern=r'config:(.+)=(.+)\n', result_pattern=r'test_result:(.+)=(.+)\n'):

        if not self.is_log_parsed:
            try:
                with open(self.std_out_log, 'r', encoding='utf8') as f:
                    for line in f:
                        m1 = re.match(config_pattern, line)
                        m2 = re.match(result_pattern, line)

                        if m1:
                            key = m1.group(1)
                            value = m1.group(2)
                            self.full_configs[key] = value

                        if m2:
                            key = m2.group(1)
                            value = m2.group(2)
                            self.test_results[key] = value
            except:
                pass

            self.is_log_parsed = True


class ConfigList:
    def __init__(self, name='Unnamed test group'):
        self.name = name
        self.conf_list = [
            RunConfig(app=App.gcn,       dataset=Dataset.products),
            RunConfig(app=App.gcn,       dataset=Dataset.twitter),
            RunConfig(app=App.gcn,       dataset=Dataset.papers100M),
            RunConfig(app=App.gcn,       dataset=Dataset.uk_2006_05),

            RunConfig(app=App.graphsage, dataset=Dataset.products),
            RunConfig(app=App.graphsage, dataset=Dataset.twitter),
            RunConfig(app=App.graphsage, dataset=Dataset.papers100M),
            RunConfig(app=App.graphsage, dataset=Dataset.uk_2006_05),

            RunConfig(app=App.pinsage,   dataset=Dataset.products),
            RunConfig(app=App.pinsage,   dataset=Dataset.twitter),
            RunConfig(app=App.pinsage,   dataset=Dataset.papers100M),
            RunConfig(app=App.pinsage,   dataset=Dataset.uk_2006_05),
        ]
        self.fail_count = 0

    def select(self, key, val_indicator):
        '''
        filter config list by key and list of value
        available key: app, dataset, cache_policy, pipeline
        '''
        newlist = []
        for cfg in self.conf_list:
            if key in cfg.configs.keys() and cfg.configs[key] in val_indicator:
                newlist.append(cfg)
        self.conf_list = newlist
        return self

    def override(self, key, val_list):
        '''
        override config list by key and value.
        if len(val_list)>1, then config list is extended, example:
            [cfg1(batch_size=4000)].override('batch_size',[1000,8000])
         => [cfg1(batch_size=1000),cfg1(batch_size=8000)]
        available key: arch, logdir, cache_percent, cache_policy, batch_size
        '''
        if len(val_list) == 0:
            return self
        orig_list = self.conf_list
        self.conf_list = []
        for val in val_list:
            new_list = copy.deepcopy(orig_list)
            for cfg in new_list:
                cfg.configs[key] = val
            self.conf_list += new_list
        return self

    def combo(self, select_key, select_val_indicator, override_key, override_val_list):
        '''
        Combination of select and override
        Select some configs and override their values. The remaining configs keep no change
        '''

        if len(override_val_list) == 0:
            return self

        # tmp select
        orig_list = self.conf_list
        newlist = []
        self.conf_list = []
        for cfg in orig_list:
            if select_key in cfg.configs.keys() and cfg.configs[select_key] in select_val_indicator:
                newlist.append(cfg)
            else:
                self.conf_list.append(cfg)

        # apply override
        for val in override_val_list:
            newnew_list = copy.deepcopy(newlist)
            for cfg in newnew_list:
                cfg.configs[override_key] = val
            self.conf_list += newnew_list
        return self

    def _list_select(self, cfg, select_op, select_key_val_dict):
        if select_op == 'and':
            for key, vals in select_key_val_dict.items():
                if not key in cfg.configs.keys() or not cfg.configs[key] in vals:
                    return False
            return True
        else:
            for key, vals in select_key_val_dict.items():
                if key in cfg.configs.keys() and cfg.configs[key] in vals:
                    return True
            return False

    def multi_combo(self, select_op, select_key_val_dict, override_key, override_val_list):
        assert(select_op == 'and' or select_op == 'or')

        if len(override_val_list) == 0:
            return self

        # tmp select
        orig_list = self.conf_list
        newlist = []
        self.conf_list = []
        for cfg in orig_list:
            if self._list_select(cfg, select_op, select_key_val_dict):
                newlist.append(cfg)
            else:
                self.conf_list.append(cfg)

        # apply override
        for val in override_val_list:
            newnew_list = copy.deepcopy(newlist)
            for cfg in newnew_list:
                cfg.configs[override_key] = val
            self.conf_list += newnew_list
        return self

    def multi_combo_multi_override(self, select_op, select_key_val_dict, override_key_val_dict):
        assert(select_op == 'and' or select_op == 'or')

        if len(override_key_val_dict) == 0:
            return self

        # tmp select
        orig_list = self.conf_list
        newlist = []
        self.conf_list = []
        for cfg in orig_list:
            if self._list_select(cfg, select_op, select_key_val_dict):
                newlist.append(cfg)
            else:
                self.conf_list.append(cfg)

        # apply override
        for cfg in newlist:
            for override_key, override_value in override_key_val_dict.items():
                cfg.configs[override_key] = override_value

        self.conf_list += newlist
        return self

    def multi_combo_multi_override_list(self, select_op, select_key_val_dict, override_key_val_dict_list):
        assert(select_op == 'and' or select_op == 'or')

        if len(override_key_val_dict_list) == 0:
            return self

        # tmp select
        orig_list = self.conf_list
        newlist = []
        self.conf_list = []
        for cfg in orig_list:
            if self._list_select(cfg, select_op, select_key_val_dict):
                newlist.append(cfg)
            else:
                self.conf_list.append(cfg)

        # apply override
        for cfg in newlist:
            for override_key_val_dict in override_key_val_dict_list:
                new_cfg = copy.deepcopy(cfg)
                for override_key, override_value in override_key_val_dict.items():
                    new_cfg.configs[override_key] = override_value

                self.conf_list.append(new_cfg)

        return self

    def write_configs_book(self, logdir, mock=False):
        if mock:
            print(f'Test Group: {self.name}')
            for i, conf in enumerate(self.conf_list):
                print(f'Config{i}:')
                for k, v in conf.configs.items():
                    print(f'  {k}: {v}')
        else:
            os.system('mkdir -p {}'.format(logdir))
            with open(os.path.join(logdir, 'configs_book.txt'), 'w', encoding='utf8') as f:
                f.write(f'Test Group: {self.name}' + '\n')
                for i, conf in enumerate(self.conf_list):
                    f.write(f'Config{i}:' + '\n')
                    for k, v in conf.configs.items():
                        f.write(f'  {k}: {v}' + '\n')

    def write_run_status(self, logdir, mock):
        if mock:
            for i, conf in enumerate(self.conf_list):
                print(f'Test{i}={conf.status}')
        else:
            os.system('mkdir -p {}'.format(logdir))
            with open(os.path.join(logdir, 'run_status.txt'), 'w', encoding='utf8') as f:
                for i, conf in enumerate(self.conf_list):
                    f.write(f'Test{i}={conf.status}' + '\n')

    def load_run_status(self, logdir):
        pattern = r'Test(.+)=(.+)\n'
        try:
            with open(os.path.join(logdir, 'run_status.txt'), 'r', encoding='utf8') as f:
                for line in f:
                    m = re.match(pattern, line)
                    if m:
                        idx = int(m.group(1))
                        status = m.group(2)
                        self.conf_list[idx].status = RunStatus.Ok if status == '1' else RunStatus.NotOk
        except:
            pass

    def match(self, params):
        ret = []

        for conf in self.conf_list:
            if conf.match(params):
                ret.append(conf)

        return ret

    def run(self, appdir, logdir, mock=False, durable_log=True, callback=None):
        self.write_configs_book(logdir, mock)
        self.load_run_status(logdir)

        error_count = 0
        for i, conf in enumerate(tqdm(self.conf_list, desc=self.name)):
            if conf.status != RunStatus.Ok:
                conf: RunConfig
                ret = conf.run(i, appdir, logdir,
                               mock, durable_log, callback)
                error_count += (ret > 0)
                if ret == 0:
                    conf.status = RunStatus.Ok
            else:
                ret = conf.form_cmd(i, appdir, logdir, durable_log)

        self.write_run_status(logdir, mock)
        return self

    def parse_logs(self, logtable, logdir, left_wrap=' ', right_wrap=' ', sep=' '):
        assert(logtable.is_finalized)
        with open(os.path.join(logdir, 'test_result.txt'), 'w', encoding='utf8') as f:
            for i in range(logtable.num_row):
                for j in range(logtable.num_col):
                    row_def = logtable.row_definitions[i][j]
                    col_def = logtable.col_definitions[j]

                    configs = self.match(row_def)
                    assert(len(configs) == 1)

                    conf = configs[0]
                    conf.parse_log()

                    logtable.data[i][j] = conf.test_results[col_def]
                    logtable.data_configs[i][j] = conf.full_configs
                    logtable.data_refs[i].add(
                        os.sep.join(os.path.normpath(conf.std_out_log).split(os.sep)[-2:]))

                    f.write('{:}{:}{:}{:}'.format('' if j == 0 else sep,
                                                    left_wrap, logtable.data[i][j], right_wrap))
                f.write('  # {:s}\n'.format(
                    ' '.join(logtable.data_refs[i])))

        return self

    def parse_logs_no_output(self, logtable):
        assert(logtable.is_finalized)
        for i in range(logtable.num_row):
            for j in range(logtable.num_col):
                row_def = logtable.row_definitions[i][j]
                col_def = logtable.col_definitions[j]

                configs = self.match(row_def)
                assert(len(configs) == 1)

                conf = configs[0]
                conf.parse_log()

                logtable.data[i][j] = conf.test_results[col_def]
                logtable.data_configs[i][j] = conf.full_configs
                logtable.data_refs[i].add(conf.std_out_log)

        return self
