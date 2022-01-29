# Table 1: Motivation Test

The goal of this experiment is to show the speedup performance and the memory contention of the two optimizations(i.e. GPU sampling and GPU caching) for the sample-based GNN training.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage (0.2 means caching the features for the 20% vertex out of total) and the number of CPU working threads.
  - Cache percentage
    - **SGNN**: Modify `L116, L121` in `run.py` and `L68, L78` in `logtable_def.py`
    - `arch0` means SGNN w/CPU sampling(default cache pct. is 0.2) while `arch2` means SGNN w/GPU sampling(default cache pct. is  0.07)
    - Cache percentage 0 means not utilizing the cache mechanism.

  - GPU working threads
    - **DGL**: Modify `L58` in `run.py`
    - **SGNN**: Modify `L105` in `run.py`




## Run Command


```sh
> python run.py
```



There are several command line arguments:

- `--num-epoch`: Number of epochs to run per test case.  The default value is set to 3 for fast run. In the paper, we set it to 10.
- `--mock`: Show the run command for each test case but not actually run it
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

`python run.py` will create a new folder(e.g. `output_2022-01-29_17-19-44`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_17-19-44
output_2022-01-29_17-19-44
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
> cat output_2022-01-29_17-19-44/table1.dat
GNN Systems               Sample  Extract  Train  Total    #
DGL                         4.40    13.63   4.10  22.51    # logs_dgl/test1.log
 w/ GPU-base Sampling       1.28    13.61   4.12  19.08    # logs_dgl/test0.log
SGNN                        3.26     5.96   4.14  13.37    # logs_sgnn/test3.log
 w/ GPU-base Caching        3.05     1.95   4.07   8.99    # logs_sgnn/test2.log
 w/ GPU-base Sampling       0.72     5.88   4.07  10.70    # logs_sgnn/test1.log
 w/ Both                    0.72     3.90   3.98   8.61    # logs_sgnn/test0.log
```





## FAQ
