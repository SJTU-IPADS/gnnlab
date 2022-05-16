from ctypes.wintypes import PINT
import os, sys
import subprocess

class TestCase:
    def __init__(self, dataset, percentage, policy, ctx) -> None:
        self.dataset = dataset
        self.percentage = percentage
        self.policy = policy
        self.ctx = ctx

    def __iter__(self):
        for ds in self.dataset:
            for percentage in self.percentage:
                for policy in self.policy:
                    for ctx in self.ctx:
                        yield {
                            'DATA_SET'   : ds,
                            'UM_PERCENT' : percentage,
                            'UM_POLICY'  : policy,
                            'UM_CTX'     : ctx
                        }

    def __len__(self):
        return len(self.dataset) * len(self.percentage) * len(self.policy) * len(self.ctx)

cases = TestCase(
    dataset=[
        # 'reddit', 
        'papers100M',
        'uk-2006-05',
        'com-friendster',
    ], 
    percentage=['0', '0.25', '0.50', '0.75', '1'],
    policy=[
        'default', 
        # 'presample',
    ],
    ctx=[
        'cuda:1 cpu', 
        'cuda:1 cuda:2'
    ]
)

def test_env(case:dict):
    env = dict(os.environ)
    comm_env = {
        'CUDA_DEVICE_ORDER'    : 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES' : '1, 2, 3'
    }
    return {**env, **comm_env, **case}

if __name__ == '__main__':
    for i, case in enumerate(cases):
        env = test_env(case)
        print(f'TESTING: {100 * i / len(cases)}%', flush=True)
        print('\t', case, flush=True)
        subprocess.run(args=[
            'bash',
            'example/samgraph/unified_memory/single_sampler_um_multi_gpu.sh',
            '-log'
        ], env=env)


