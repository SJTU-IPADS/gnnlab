OCP_MEM    = 'OCP_MEM'
DATA_SET   = 'DATA_SET'
SAMPLER    = 'SAMPLER'
UM         = 'UM'
UM_IN_CPU  = 'UM_IN_CPU'
UM_FACTOR  = 'UM_FACTOR'
UM_POLICY  = 'UM_POLICY'
UM_PERCENT = 'UM_PERCENT'


class Papers100M:
    um_test_normal_cases = [
        {
            DATA_SET   : 'papers100M',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i),            
        } for i in [1, 0.75, 0.50, 0.25, 0.10, 0.05]
    ]

    um_test_graph_in_gpu = [
        {
            DATA_SET   : 'papers100M',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i)
        } for i in [1]
    ]

    um_test_graph_in_cpu = [
        {
            DATA_SET   : 'papers100M',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i)
        } for i in [0]
    ]

    um_test_gpu_not_use_um = [
        {
            DATA_SET   : 'papers100M',
            SAMPLER    : 'gpu',
            UM         : '0',
            UM_PERCENT : str(i)
        } for i in [0]
    ]

    um_test_cpu = [
        {
            DATA_SET   : 'papers100M',
            SAMPLER    : 'cpu',
            UM         : '0',
            UM_PERCENT : str(i)
        } for i in [0]
    ]


class Friendster:
    um_test_normal_cases = [
        {
            DATA_SET   : 'com-friendster',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i)
        } for i in [1, 0.75, 0.50, 0.25, 0.10, 0.05]
    ]

    um_test_graph_in_gpu = [
        {
            DATA_SET   : 'com-friendster',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i)
        } for i in [1]
    ]

    um_test_graph_in_cpu = [
        {
            DATA_SET   : 'com-friendster',
            SAMPLER    : 'gpu',
            UM         : '1',
            UM_PERCENT : str(i)
        } for i in [0]
    ]

    um_test_gpu_not_use_um = [
        {
            DATA_SET   : 'com-friendster',
            SAMPLER    : 'gpu',
            UM         : '0',
            UM_PERCENT : str(i)
        } for i in [0]
    ]

    um_test_cpu = [
        {
            OCP_MEM    : str(i * 1024 ** 3),
            DATA_SET   : 'com-friendster',
            SAMPLER    : 'cpu',
            UM         : '0',
            UM_PERCENT : str(i)
        } for i in [0]
    ]