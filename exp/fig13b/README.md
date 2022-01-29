# Figure 13b:  PinSAGE Scalability Test

The goal of this experiment is to show the scalability performance of DGL and FGNN on PinSAGE model.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of GPU. ②Number of CPU threads ③Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **DGL:** Modify `L61(#GPU)` in `run.py`.
  - **FGNN:**  Modify  `L99(#CPU threads), L104-L139(#GPU, #Cache percentage)` in `run.py`.



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

`python run.py` will create a new folder(e.g. `output_2022-01-29_20-45-14`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_20-45-14 -L 1
output_2022-01-29_20-45-14
├── fig13b.eps           # Output figure
├── fig13b-full.res      # Output data with comments
├── fig13b.res           # Output data
├── logs_dgl
└── logs_fgnn

2 directories, 3 files
```



```sh
> cat output_2022-01-29_20-45-14/fig13b.res
"GPUs"  "DGL"   "1S"    "2S"    "3S"
1       13.14   -       -       -
2       6.92    6.25    -       -
3       4.90    3.22    6.22    -
4       3.89    2.21    3.26    6.34
5       3.31    1.66    2.21    3.29
6       3.03    1.38    1.67    2.26
7       2.81    1.18    1.39    1.65
8       2.57    1.02    1.17    1.37
```





## FAQ