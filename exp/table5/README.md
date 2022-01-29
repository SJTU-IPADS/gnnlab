# Table 5:  Performance Breakdown Test

The goal of this experiment is to show the performance breakdown of DGL, PyG, and FGNN.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **2x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- Minimum hardware configuration: At least 2 GPUs.
- For other hardware configurations, you may need to modify the ①Number  of CPU threads. ②Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **PyG:** Modify `L151(#CPU threads)` in `run.py`.
  - **FGNN:**  Modify  `L193(#CPU threads), L207-L266(#Cache percentage)` in `run.py`.



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

`python run.py` will create a new folder(e.g. `output_2022-01-29_18-54-46`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_18-54-46 -L 1
output_2022-01-29_18-54-46
├── logs_dgl
├── logs_dgl_pinsage
├── logs_fgnn
├── logs_pyg
├── table5.dat               # output table data
└── table5-full.dat          # output table data with comments

4 directories, 2 files
```



```sh
> cat output_2022-01-29_18-54-46/table5.dat
--------------------------------------------------------------------------------------------------------------------------------------
          |            DGL             |            PyG             |                             FGNN
--------------------------------------------------------------------------------------------------------------------------------------
 GNN | DS | Sample   Extract    Train  | Sample   Extract    Train  |      Sample = S + M + C          Extract (Ratio, Hit%)    Train
--------------------------------------------------------------------------------------------------------------------------------------
 GCN | PR |  0.37      3.48      1.28  |  6.28      3.70      2.28  | 0.40  = 0.30  + 0.01  + 0.09      0.16   (1.00%, 1.00%)    1.23
 GCN | TW |  0.76     11.71      1.55  | 10.44     11.97      2.60  | 0.38  = 0.26  + 0.03  + 0.08      0.82   (0.25%, 0.89%)    1.55
 GCN | PA |  1.28     13.31      4.16  |  8.97     14.30      6.53  | 1.00  = 0.70  + 0.11  + 0.19      0.49   (0.21%, 0.99%)    3.93
 GCN | UK |     X         X         X  | 14.93     20.68      4.98  | 0.57  = 0.38  + 0.04  + 0.15      2.75   (0.14%, 0.70%)    3.09
--------------------------------------------------------------------------------------------------------------------------------------
 GSG | PR |  0.14      2.38      0.24  |  3.12      2.45      0.24  | 0.21  = 0.15  + 0.01  + 0.05      0.10   (1.00%, 1.00%)    0.26
 GSG | TW |  0.38      5.87      0.46  |  7.54      6.11      0.35  | 0.16  = 0.11  + 0.02  + 0.04      0.49   (0.32%, 0.89%)    0.44
 GSG | PA |  0.56      7.50      1.31  |  3.98      8.10      0.93  | 0.46  = 0.31  + 0.06  + 0.09      0.28   (0.25%, 0.99%)    1.14
 GSG | UK |     X         X         X  | 11.86     10.73      0.87  | 0.26  = 0.18  + 0.02  + 0.07      1.39   (0.18%, 0.72%)    1.04
--------------------------------------------------------------------------------------------------------------------------------------
 PSG | PR |  0.17      1.90      1.83  |   X         X         X    | 0.21  = 0.16  + 0.01  + 0.04      0.08   (1.00%, 1.00%)    1.79
 PSG | TW |  0.23      6.06      2.67  |   X         X         X    | 0.28  = 0.21  + 0.02  + 0.05      0.61   (0.26%, 0.86%)    2.57
 PSG | PA |  0.53      6.12      6.30  |   X         X         X    | 0.62  = 0.47  + 0.05  + 0.10      0.33   (0.22%, 0.97%)    6.22
 PSG | UK |     X         X         X  |   X         X         X    | 0.67  = 0.49  + 0.03  + 0.14      2.99   (0.13%, 0.57%)    7.12
--------------------------------------------------------------------------------------------------------------------------------------
```





## FAQ