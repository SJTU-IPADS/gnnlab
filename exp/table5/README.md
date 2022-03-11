# Table 5:  Performance Breakdown Test

The goal of this experiment is to show the performance breakdown of DGL, SGNN, and FGNN.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **2x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- Minimum hardware configuration: At least 2 GPUs.
- For other hardware configurations, you may need to modify the ①Number  of CPU threads. ②Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **SGNN:** Modify `L197(#CPU threads), L216-L271(#Cache percentage` in `run.py`.
  - **FGNN:**  Modify  `L310(#CPU threads), L328-L383(#Cache percentage)` in `run.py`.



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
├── logs_sgnn
├── table5.dat               # output table data
└── table5-full.dat          # output table data with comments

4 directories, 2 files
```



```sh
> cat output_2022-01-29_18-54-46/table5.dat
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
          |            DGL             |                           SGNN                           |                             FGNN
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
 GNN | DS | Sample   Extract    Train  |    Sample = S + M        Extract (Ratio, Hit%)    Train  |      Sample = S + M + C          Extract (Ratio, Hit%)    Train
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
 GCN | PR |  0.39      3.22      1.30  | 0.32  = 0.31  + 0.01      0.05   (1.00%, 1.00%)    1.30  | 0.39  = 0.30  + 0.01  + 0.08      0.15   (1.00%, 1.00%)    1.30
 GCN | TW |  0.90     11.04      1.64  | 0.31  = 0.28  + 0.03      3.51   (0.01%, 0.32%)    1.70  | 0.36  = 0.26  + 0.03  + 0.07      0.81   (0.25%, 0.89%)    1.73
 GCN | PA |  1.67     12.73      4.21  | 0.84  = 0.73  + 0.11      0.32   (0.07%, 0.98%)    4.48  | 0.95  = 0.68  + 0.11  + 0.17      0.50   (0.21%, 0.99%)    4.46
 GCN | UK |     X         X         X  |    X  =    X  +    X         X   (   X%,    X%)       X  | 0.55  = 0.38  + 0.03  + 0.13      3.06   (0.14%, 0.70%)    3.57
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
 GSG | PR |  0.15      2.47      0.24  | 0.16  = 0.15  + 0.01      0.03   (1.00%, 1.00%)    0.28  | 0.20  = 0.15  + 0.01  + 0.04      0.08   (1.00%, 1.00%)    0.29
 GSG | TW |  0.43      5.61      0.49  | 0.13  = 0.11  + 0.02      0.63   (0.15%, 0.82%)    0.54  | 0.15  = 0.11  + 0.01  + 0.03      0.48   (0.32%, 0.89%)    0.52
 GSG | PA |  0.64      7.07      1.40  | 0.39  = 0.33  + 0.06      0.19   (0.11%, 0.98%)    1.50  | 0.45  = 0.31  + 0.06  + 0.08      0.35   (0.25%, 0.99%)    1.47
 GSG | UK |     X         X         X  | 0.19  = 0.19  + 0.00      4.31   (0.00%, 0.00%)    1.26  | 0.26  = 0.18  + 0.02  + 0.06      1.35   (0.18%, 0.72%)    1.27
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
 PSG | PR |  0.48      1.97      1.91  | 0.18  = 0.17  + 0.01      0.03   (1.00%, 1.00%)    1.97  | 0.20  = 0.15  + 0.01  + 0.04      0.07   (1.00%, 1.00%)    1.93
 PSG | TW |  0.95      6.16      2.87  | 0.25  = 0.23  + 0.02      1.19   (0.04%, 0.62%)    2.94  | 0.28  = 0.21  + 0.02  + 0.05      0.58   (0.26%, 0.86%)    2.86
 PSG | PA |  2.46      5.95      6.30  | 0.57  = 0.52  + 0.05      0.22   (0.06%, 0.97%)    6.80  | 0.62  = 0.48  + 0.05  + 0.09      0.34   (0.22%, 0.97%)    6.94
 PSG | UK |     X         X         X  |    X  =    X  +    X         X   (   X%,    X%)       X  | 0.63  = 0.47  + 0.03  + 0.13      3.39   (0.13%, 0.57%)    8.00
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
```





## FAQ