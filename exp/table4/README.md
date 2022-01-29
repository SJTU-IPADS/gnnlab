# Table 4: Overall Performance Test

The goal of this experiment is to show the overall end2end performance of DGL, PyG, SGNN and FGNN.

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of GPU ②Number  of CPU threads. ③Number of vertex(in percentage, 0<=pct. <=1) to be cached.
  - **DGL:** Modify `L69(#GPU), L110(#GPU)` in `run.py`. 
  - **PyG:** Modify `L142(#CPU threads), L145(#GPU)` in `run.py`.
  - **SGNN:** Modify `L184(#CPU threads), L203(#GPU), L204-L254(#Cache Percentage)` in `run.py`.
  - **FGNN:**  Modify  `L290(#CPU threads), L304-L351(#CPU, #Cache percentage)` in `run.py`.



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

`python run.py` will create a new folder(e.g. `output_2022-01-29_18-17-21`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree -L 1 output_2022-01-29_18-17-21
output_2022-01-29_18-17-21
├── logs_dgl
├── logs_dgl_pinsage
├── logs_fgnn
├── logs_pyg
├── logs_sgnn
├── table4.dat                # output table data
└── table4-full.dat           # output table data with data source

5 directories, 2 files
```



```sh
> cat output_2022-01-29_18-17-21/table4.dat
GNN Models       Dataset     DGL    PyG   SGNN       FGNN
GCN              PR         1.18  11.28   0.23   0.33(3S)
GCN              TW         3.40  11.48   1.35   0.41(2S)
GCN              PA         4.07  14.58   2.75   0.86(2S)
GCN              UK            X  13.30      X   1.20(2S)
GraphSAGE        PR         0.71   7.45   0.07   0.11(4S)
GraphSAGE        TW         1.63   7.50   0.32   0.17(2S)
GraphSAGE        PA         2.17   8.64   0.95   0.28(2S)
GraphSAGE        UK            X   8.95   1.59   0.51(1S)
PinSAGE          PR         0.80      X   0.31   0.40(1S)
PinSAGE          TW         2.01      X   0.81   0.51(1S)
PinSAGE          PA         2.56      X   1.72   1.03(1S)
PinSAGE          UK            X      X      X   1.43(1S)
```




## FAQ