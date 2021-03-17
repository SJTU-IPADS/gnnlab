from ctypes import *
import os
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
subprocess.run(
            ['g++', '-shared' , '-fPIC', '-o' ,'ctypes_param.so', 'ctypes_param.cc'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,)
target = os.path.join(here, 'ctypes_param.so')
mod = CDLL(os.path.join(here, 'ctypes_param.so'))
big_num = mod.big_num

big_int = 28147497671065
big_num.restype = c_ulonglong
ret = mod.big_num(c_ulonglong(big_int))
print(ret)
big_num.argtypes = (c_ulonglong,)
ret = big_num(big_int)
print(ret)