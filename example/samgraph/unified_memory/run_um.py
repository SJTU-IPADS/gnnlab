import os
import subprocess
from test_cases import *


# papers100M ~7G 
# reddit     ~1G 
# friendster ~16G

# cases = [
#     {
#         OCP_MEM   : str(i * 1024 ** 3),
#         DATA_SET  : 'com-friendster',
#         SAMPLER   : 'gpu',
#         UM        : '1',
#         UM_IN_CPU : '0',
#         UM_FACTOR : str(16 / (29.5 - i))
#     } for i in [13, 17, 21, 25, 29]
# ][:1]

# cases = Papers100M.um_test_gpu_not_use_um + Papers100M.um_test_normal_cases + Papers100M.um_test_graph_in_cpu
# cases = Papers100M.um_test_graph_in_cpu
# cases = Papers100M.um_test_cpu #+ Papers100M.um_test_graph_in_cpu
# cases = Papers100M.um_test_normal_cases[-1:]
# cases = Papers100M.um_test_graph_in_gpu

# cases = Friendster.um_test_gpu_not_use_um + Friendster.um_test_normal_cases + Friendster.um_test_graph_in_cpu 
# cases = Friendster.um_test_cpu
# cases = Friendster.um_test_graph_in_gpu

# cases = Papers100M.um_test_normal_cases + Friendster.um_test_normal_cases
cases = [Papers100M.um_test_normal_cases[0], Papers100M.um_test_normal_cases[2]]
# cases = Papers100M.um_test_gpu_not_use_um

def um_test_env(case:dict):
    env = dict(os.environ)
    comm_env = {
        'CUDA_DEVICE_ORDER'    : 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES' : '1, 2'
    }
    return {**env, **comm_env, **case}


def add_um_policy(cases, policies):
    res = []
    for policy in policies:
        for case in cases:
            res.append({
                **case, 
                'UM_POLICY' : policy
            })
    return res


if __name__ == '__main__':
    policies = []
    policies += ['default']
    # policies += ['degree']
    # policies += ['trainset']
    policies += ['presample']
    for case in add_um_policy(cases, policies):
        subprocess.run('nvidia-smi')
        subprocess.run(args=[
            'bash',
            'example/samgraph/unified_memory/single.sh',
            '-log'
        ], env=um_test_env(case))