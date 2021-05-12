import os
import subprocess
import time
import copy
import pandas as pd
import re
import signal
import sys

TIMESTAMP = time.strftime('%Y%m%dT%H%M%S')
OUTPUT_FILE = 'evaluation{:s}.csv'.format(TIMESTAMP)
ENV = os.environ.copy()
ENV['MKL_SERVICE_FORCE_INTEL'] = '1'


def back_track(elements, nums, first, length, ret):
    if first == length:
        for i in range(1, length):
            if nums[i] < nums[i - 1]:
                return

        ret.append(copy.deepcopy(nums))
        return

    for num in elements:
        nums[first] = num
        back_track(elements, nums, first + 1, length, ret)


def generate_permutation(elements, min_len, max_len):
    ret = []
    for length in range(min_len, max_len + 1):
        nums = [0 for _ in range(length)]
        back_track(elements, nums, 0, length, ret)
    return ret


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.output_data = []

    def run(self):
        here = os.path.dirname(os.path.abspath(__file__))
        print(here)
        app_path = '{:s}/samgraph/train_graphsage.py'.format(here)
        for batch_size in self.config['batch_size_candidates']:
            for num_hidden in self.config['num_hidden_candidates']:
                for fanout in self.config['fanout_candidates']:
                    print("Running test with batch", batch_size,
                          "num_hidden", num_hidden, "fanout", ' '.join(f'{f}' for f in fanout), '...')
                    try:
                        process = subprocess.run(
                            ['python', app_path, '--parse-args', '--report-last', '--num-epoch', '1', '--batch-size',
                                str(batch_size), '--num-hidden', str(num_hidden), '--fanout'] + [str(f) for f in fanout],
                            capture_output=True,
                            env=ENV
                        )
                        process.check_returncode()
                        output = process.stdout.decode('utf-8')
                        data = self.parse_result(output)

                        self.output_data.append({
                            'batch_size': batch_size,
                            'num_hidden': num_hidden,
                            'fanout': ' '.join(f'{f}' for f in fanout),
                            **data
                        })
                    except subprocess.CalledProcessError as e:
                        print(e)
                        error = process.stderr.decode('utf-8')
                        print(error)

        self.write_file()

    def parse_result(self, output):
        data = {}
        data['num_samples'] = re.search(
            r'Samples ([0-9]+)', output).group(1)
        data['total_time'] = re.search(
            r'Time ([0-9]+\.[0-9]+)', output).group(1)
        data['sample_time'] = re.search(
            r'] sample ([0-9]+\.[0-9]+)', output).group(1)
        data['copy_time'] = re.search(
            r'copy ([0-9]+\.[0-9]+)', output).group(1)
        data['train_time'] = re.search(
            r'Train Time ([0-9]+\.[0-9]+)', output).group(1)
        data['shuffle_time'] = re.search(
            r'shuffle ([0-9]+\.[0-9]+)', output).group(1)
        data['core_sample_time'] = re.search(
            r'core sample ([0-9]+\.[0-9]+)', output).group(1)
        data['id_remap_time'] = re.search(
            r'id remap ([0-9]+\.[0-9]+)', output).group(1)
        data['graph_copy_time'] = re.search(
            r'graph copy ([0-9]+\.[0-9]+)', output).group(1)
        data['id_copy_time'] = re.search(
            r'id copy ([0-9]+\.[0-9]+)', output).group(1)
        data['extract'] = re.search(
            r'extract ([0-9]+\.[0-9]+)', output).group(1)
        data['feat_copy_time'] = re.search(
            r'feat copy ([0-9]+\.[0-9]+)', output).group(1)
        data['convert_time'] = re.search(
            r'Convert Time ([0-9]+\.[0-9]+)', output).group(1)
        return data

    def write_file(self):
        df = pd.DataFrame(self.output_data)
        df.to_csv(OUTPUT_FILE)


def get_config():
    config = {}

    config['batch_size_candidates'] = [8192]
    config['num_hidden_candidates'] = [256]

    fanout_candidates = generate_permutation(
        elements=[20, 15, 10, 5], min_len=1, max_len=4)
    fanout_candidates.sort(key=lambda x: (len(x), x))
    for fanout in fanout_candidates:
        fanout.reverse()
    config['fanout_candidates'] = fanout_candidates

    return config


config = get_config()
runner = Runner(config)


def signal_handler(sig, frame):
    print("Recv SIGINT signal. Saving data...")
    runner.write_file()
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    print(config)
    runner.run()
