# Table 1: Motivation Test

The goal of this experiment is to show the speedup performance and the memory contention of the optimizations(i.e. GPU sampling and GPU caching) for the sample-based GNN training.

`run.py` is the runner script for table1 while `logtable_def.py` define log parsing rules for table1.



## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage (0.2 means caching the features for the 20% vertex out of total).
  -  Modify `L112, L117` in `run.py` and `L68, L78` in `logtable_def.py`
  -  `arch0` means SGNN w/CPU sampling(default cache pct. is 0.2) while `arch2` means SGNN w/GPU sampling(default cache pct. is  0.07)
  -  Cache percentage 0 means not utilizing the cache mechanism.



## Run Command


```sh
> python run.py
```



There are serveral command line arguments:

- `--num-epoch`: Number of epochs to run per test case.  The default value is set to 3 for fast run. In the paper, we set it to 10.
- `--mock`: Show the run command for each test case but not actually run it
- `--rerun-tests` Rerun the most recently tests. Sometimes not all the test cases run successfully(e.g. cache percentage is too large and leads to OOM). You can adjust the configurations and rerun the tests agains. The `--rerun-tests` option only reruns those failed test cases.



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
