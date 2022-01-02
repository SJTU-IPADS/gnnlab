import os
import subprocess

OCP_MEM   = 'OCP_MEM'
DATA_SET  = 'DATA_SET'
SAMPLER   = 'SAMPLER'
UM        = 'UM'
UM_IN_CPU = 'UM_IN_CPU'
UM_FACTOR = 'UM_FACTOR'

#            GPU            CPU
# papers100M ~7G ~800M(UM)  ~3G   ~9.5G(UM)
# reddit     ~1G ~800M(UM)  ~3.2G ~3.7G(UM)

cases = [
    {
        OCP_MEM   : str(i * 1024 ** 3),
        DATA_SET  : 'papers100M',
        SAMPLER   : 'gpu',
        UM        : '1',
        UM_IN_CPU : '1',
        UM_FACTOR : str(9 / (29.5 - i))
    } for i in [0]
]

def um_test_env(case:dict):
    env = dict(os.environ)
    return {**env, **case}

def mem_eator(size):
    return subprocess.Popen(
        ['mem_eator/build/mem_eator.o', str(size)], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == '__main__':
    for case in cases:
        eator = mem_eator(int(case[OCP_MEM]))
        for stdout_line in iter(eator.stdout.readline, ""):
            line = stdout_line.decode('utf-8')
            print(line)
            if 'mem eator' in line:
                break
        subprocess.run('nvidia-smi')
        subprocess.run(args=[
            'bash',
            'example/samgraph/unified_memory/single.sh',
            '-log'
        ], env=um_test_env(case))

        eator.kill()