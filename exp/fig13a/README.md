# Figure 13a:  GCN Scalability Test

The goal of this experiment is to show the scalability performance of DGL and FGNN on GCN model.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of GPU. ②Number of CPU threads ③Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **DGL:** Modify `L64-L65(#GPU)` in `run.py`.
  - **FGNN:**  Modify  `L106(#CPU threads), L116-L151(#GPU, #Cache percentage)` in `run.py`.



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

`python run.py` will create a new folder(e.g. `output_2022-01-29_20-10-39`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_20-10-39 -L 1
output_2022-01-29_20-10-39
├── fig13a.eps             # Output figure
├── fig13a-full.res        # Output data with data source
├── fig13a.res             # Output data
├── logs_dgl
└── logs_fgnn

2 directories, 3 files
```



```sh
> cat output_2022-01-29_20-10-39/fig13a.res
"GPUs"  "DGL"   "1S"    "2S"    "3S"
1       18.51   -       -       -
2       9.79    4.11    -       -
3       7.19    2.14    4.08    -
4       6.00    1.49    2.19    4.14
5       5.28    1.19    1.46    2.13
6       4.79    1.06    1.11    1.45
7       4.48    1.05    0.93    1.11
8       4.03    1.04    0.81    0.91
```





## FAQ