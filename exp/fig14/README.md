# Figure 14:  GCN breakdown Test

The goal of this experiment is to show the scalability breakdown of FGNN on GCN model.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of GPU. ②Number of CPU threads ③Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **FGNN:**  Modify  `L63(#CPU threads), L73-L145(#GPU, #Cache percentage)` in `run.py`.



## Run Command


```sh
> python run.py
```



There are several command line arguments:

- `--num-epoch`: Number of epochs to run per test case.  The default value is set to 3 for fast run. In the paper, we set it to 10.
- `--mock`: Show the run command for each test case but not actually run it.
- `--rerun-tests` Rerun the most recently tests. Sometimes not all the test cases run successfully(e.g. cache percentage is too large and leads to OOM). You can adjust the configurations and rerun the tests again. The `--rerun-tests` option only reruns those failed test cases.



```sh
> python run.py --help
usage: Table 1 Runner [-h] [--num-epoch NUM_EPOCH] [--mock] [--rerun-tests]

optional arguments:
  -h, --help            show this help message and exit
  --num-epoch NUM_EPOCH
                        Number of epochs to run per test case
  --mock                Show the run command for each test case but not actually run it
  --rerun-tests         Rerun the most recently tests
```





## Output Example

`python run.py` will create a new folder(e.g. `output_2022-01-29_21-30-39/`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_21-30-39/ -L 1
output_2022-01-29_21-30-39/
├── fig14.eps
├── fig14.res
└── logs_fgnn

1 directory, 2 files
```



```sh
> cat output_2022-01-29_21-30-39/fig14.res
" " ? ? ? ?
"1S 1T" 1.01    0.50    4.03    4.11    # logs_fgnn/test18.log logs_fgnn/test0.log
"1S 2T" 1.00    0.35    1.99    2.20    # logs_fgnn/test1.log logs_fgnn/test19.log
"1S 3T" 1.00    0.24    1.37    1.47    # logs_fgnn/test2.log logs_fgnn/test20.log
"1S 4T" 1.00    0.18    1.00    1.13    # logs_fgnn/test3.log logs_fgnn/test21.log
"1S 5T" 1.00    0.16    0.86    1.08    # logs_fgnn/test4.log logs_fgnn/test22.log
"1S 6T" 1.02    0.15    0.70    1.06    # logs_fgnn/test5.log logs_fgnn/test23.log
"1S 7T" 1.00    0.12    0.61    1.06    # logs_fgnn/test6.log logs_fgnn/test24.log
" " ? ? ? ?
"2S 1T" 0.52    0.50    3.98    4.12    # logs_fgnn/test7.log logs_fgnn/test25.log
"2S 2T" 0.52    0.34    2.01    2.14    # logs_fgnn/test26.log logs_fgnn/test8.log
"2S 3T" 0.52    0.24    1.34    1.48    # logs_fgnn/test27.log logs_fgnn/test9.log
"2S 4T" 0.54    0.20    1.12    1.13    # logs_fgnn/test28.log logs_fgnn/test10.log
"2S 5T" 0.51    0.16    0.85    0.94    # logs_fgnn/test29.log logs_fgnn/test11.log
"2S 6T" 0.53    0.14    0.70    0.80    # logs_fgnn/test30.log logs_fgnn/test12.log
" " ? ? ? ?
"3S 1T" 0.35    0.48    4.04    4.13    # logs_fgnn/test31.log logs_fgnn/test13.log
"3S 2T" 0.39    0.34    2.02    2.15    # logs_fgnn/test14.log logs_fgnn/test32.log
"3S 3T" 0.35    0.24    1.34    1.47    # logs_fgnn/test33.log logs_fgnn/test15.log
"3S 4T" 0.36    0.19    1.03    1.11    # logs_fgnn/test16.log logs_fgnn/test34.log
"3S 5T" 0.35    0.16    0.83    0.93    # logs_fgnn/test17.log logs_fgnn/test35.log
```





## FAQ