# Figure 13b:  PinSAGE Scalability Test

The goal of this experiment is to show the scalability performance of DGL and FGNN on PinSAGE model.

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

`python run.py` will create a new folder(e.g. `output_2022-01-29_14-04-31`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_14-04-31
output_2022-01-29_14-04-31
├── logs_dgl                   # log folder for dgl test cases
│   ├── configs_book.txt       # detail configurations for each test cases
│   ├── run_status.txt
│   ├── test0.err.log
│   ├── test0.log
│   ├── test1.err.log
│   ├── test1.log
│   └── test_result.txt
├── logs_sgnn                  # log folder for dgl test cases
│   ├── configs_book.txt
│   ├── run_status.txt
│   ├── test0.err.log
│   ├── test0.log
│   ├── test1.err.log
│   ├── test1.log
│   ├── test2.err.log
│   ├── test2.log
│   ├── test3.err.log
│   ├── test3.log
│   └── test_result.txt
└── table1.dat				  # output table data
```



```sh
> cat output_2022-01-29_14-04-31/table1.dat
GNN Systems               Sample  Extract  Train  Total    #
DGL                         5.24    11.94   4.00  21.64    # logs_dgl/test1.log
 w/ GPU-base Sampling       1.21    18.48   4.04  23.81    # logs_dgl/test0.log
SGNN                        2.90     5.64   4.02  12.56    # logs_sgnn/test3.log
 w/ GPU-base Caching        2.85     1.81   4.00   8.66    # logs_sgnn/test2.log
 w/ GPU-base Sampling       0.71     5.53   4.06  10.36    # logs_sgnn/test1.log
 w/ Both                    0.70     3.64   3.94   8.33    # logs_sgnn/test0.log
```





## FAQ