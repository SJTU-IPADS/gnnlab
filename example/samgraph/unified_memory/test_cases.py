OCP_MEM   = 'OCP_MEM'
DATA_SET  = 'DATA_SET'
SAMPLER   = 'SAMPLER'
UM        = 'UM'
UM_IN_CPU = 'UM_IN_CPU'
UM_FACTOR = 'UM_FACTOR'

um_test_normal_cases = [
    {
        OCP_MEM   : str(i * 1024 ** 3),
        DATA_SET  : 'papers100M',
        SAMPLER   : 'gpu',
        UM        : '1',
        UM_IN_CPU : '0',
        UM_FACTOR : str(9 / (29.5 - i))
    } for i in [20, 23, 26, 29]
]

um_test_graph_in_gpu = [
    {
        OCP_MEM   : str(i * 1024 ** 3),
        DATA_SET  : 'papers100M',
        SAMPLER   : 'gpu',
        UM        : '1',
        UM_IN_CPU : '0',
        UM_FACTOR : str(9 / (29.5 - i))
    } for i in [20]
]

um_test_graph_in_cpu = [
    {
        OCP_MEM   : str(i * 1024 ** 3),
        DATA_SET  : 'papers100M',
        SAMPLER   : 'gpu',
        UM        : '1',
        UM_IN_CPU : '1',
        UM_FACTOR : str(9 / (29.5 - i))
    } for i in [0]
]

um_test_gpu_not_use_um = [
    {
        OCP_MEM   : str(i * 1024 ** 3),
        DATA_SET  : 'papers100M',
        SAMPLER   : 'gpu',
        UM        : '0',
        UM_IN_CPU : '0',
        UM_FACTOR : str(9 / (29.5 - i))
    } for i in [0]
]