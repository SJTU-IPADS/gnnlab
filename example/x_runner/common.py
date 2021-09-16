from enum import Enum
import copy
import os


class App(Enum):
    gcn = 0
    graphsage = 1
    pinsage = 2

    def __str__(self):
        return self.name


class Dataset(Enum):
    reddit = 0
    products = 1
    papers100M = 2
    friendster = 3
    papers100M_300 = 4
    uk_2006_05 = 5
    twitter = 6
    sk_2005 = 7

    def __str__(self):
        if self is Dataset.friendster:
            return 'com-friendster'
        elif self is Dataset.uk_2006_05:
            return 'uk-2006-05'
        elif self is Dataset.sk_2005:
            return 'sk-2005'
        return self.name


class RunConfig:
    def __init__(self, app: App, **kwargs):
        self.configs = {}
        self.configs['app'] = app
        self.configs.update(kwargs)

    def form_cmd(self, idx, appdir, logdir, durable_log=True):
        cmd_line = ''
        cmd_line += 'python ' + \
            os.path.join(appdir, f'train_{self.configs["app"]}.py')

        for k, v in self.configs.items():
            if k == 'app':
                continue

            if k.startswith('bool:'):
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

        return ret


class ConfigList:
    def __init__(self):
        self.conf_list = [
            RunConfig(app=App.gcn,       dataset=Dataset.reddit),
            RunConfig(app=App.gcn,       dataset=Dataset.products),
            RunConfig(app=App.gcn,       dataset=Dataset.papers100M),
            RunConfig(app=App.gcn,       dataset=Dataset.friendster),

            RunConfig(app=App.graphsage, dataset=Dataset.reddit),
            RunConfig(app=App.graphsage, dataset=Dataset.products),
            RunConfig(app=App.graphsage, dataset=Dataset.papers100M),
            RunConfig(app=App.graphsage, dataset=Dataset.friendster),

            RunConfig(app=App.pinsage,   dataset=Dataset.reddit),
            RunConfig(app=App.pinsage,   dataset=Dataset.products),
            RunConfig(app=App.pinsage,   dataset=Dataset.papers100M),
            RunConfig(app=App.pinsage,   dataset=Dataset.friendster)
        ]

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

    def write_configs_book(self, logdir, mock=False):
        os.system('mkdir -p {}'.format(logdir))
        if mock:
            for i, conf in enumerate(self.conf_list):
                print(f'Config{i}:')
                for k, v in conf.configs.items():
                    print(f'  {k}: {v}')
        else:
            with open(os.path.join(logdir, 'configs_book.txt'), 'w', encoding='utf8') as f:
                for i, conf in enumerate(self.conf_list):
                    f.write(f'Config{i}:' + '\n')
                    for k, v in conf.configs.items():
                        f.write(f'  {k}: {v}' + '\n')

    def run(self, appdir, logdir, mock=False, durable_log=True, callback=None):
        self.write_configs_book(logdir, mock)

        error_count = 0
        for i, conf in enumerate(self.conf_list):
            print(
                f'Running config [{i + 1}/{len(self.conf_list)}], run_fails={error_count}')
            conf: RunConfig
            ret = conf.run(i, appdir, logdir,
                           mock, durable_log, callback)
            error_count += (ret > 0)
