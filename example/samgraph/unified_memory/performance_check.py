import os
import subprocess



class TestCase:
    def __init__(self, dataset, mm_type) -> None:
        self.dataset = dataset
        self.mm_type = mm_type

    def __iter__(self):
        for ds in self.dataset:
            for mm in self.mm_type:
                yield {
                    'DATA_SET' : ds,
                    'MM_TYPE'  : mm
                }


def test_env(case:dict):
    env = dict(os.environ)
    comm_env = {
        'CUDA_DEVICE_ORDER'    : 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES' : '1, 2, 3'
    }
    return {**env, **comm_env, **case}

cases = TestCase(
    dataset=[
        'papers100M',
    ],
    mm_type=[
        'P2P',
        'MAPPED_MM',
    ]
)

if __name__ == '__main__':
    for case in cases:
        env = test_env(case)
        print(case)
        subprocess.run(args=[
            'bash',
            'example/samgraph/unified_memory/performance_check.sh',
        ], env=env)