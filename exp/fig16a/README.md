# Figure 15:  Convergence Test

The goal of this experiment is to give a comparison of training time for GraphSAGE to the same accuracy target between FGNN and DGL on products and papers100M.

- `run.sh` is the runner script.

## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Total Number of Epochs ②Number of GPU. ③Number of vertex(in percentage, 0<=pct. <=1) to be cached.
  - **DGL**: Modify the arguments `num-epoch`, `devices` of L16 and L23 in run.sh.
  - **FGNN**: Modify the arguments `num-epoch`, `num-sample-worker` and `num-train-worker`, `cache-percentage` of L17 and L24 in run.sh.

## Run Command


```sh
> bash run.sh
```

## Output Example

`bash run.sh` will create a new folder(e.g. `run_logs/acc_test/one/2022-01-29_15-28-45/`) to store log files.

```sh
> tree -L 4 .
.
├── acc_one.res          # the results of all tests
├── acc-timeline-pa.plt  # drawing script of fig15b
├── acc-timeline-pr.plt  # drawing script of fig15a
├── fig15a.eps           # fig15a
├── fig15b.eps           # fig15b
├── parse_acc.py         # the script to parse the log files
├── run_logs
│   └── acc_test
│       └── one
│           └── 2022-01-29_15-28-45 # running log files
└── run.sh               # the main running script
```



```sh
> cat acc_one.res
system  dataset batch_size      time    acc
dgl     papers  8000            2.10    1.11
dgl     papers  8000            7.08    20.86
dgl     papers  8000            11.85   33.28
# ...
dgl     papers  8000            780.10  56.27
dgl     papers  8000            783.71  56.55
fgnn    papers  8000            1.86    4.19
fgnn    papers  8000            2.10    25.99
fgnn    papers  8000            2.17    26.37
# ...

```




## FAQ
