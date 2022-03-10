# Figure 16b:  Performance Test on a Single GPU

The goal of this experiment is to get the end-to-end performance between DGL and FGNN over a single GPU.

- `run.sh` is the runner script.

## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of vertex(in percentage, 0<=pct. <=1) to be cached.
  - **FGNN**: Modify the arguments `cache-percentage` of L28-L38 in run.sh.

## Run Command


```sh
> bash run.sh
```

## Output Example

`bash run.sh` will create a new folder(e.g. `run-logs/single/2022-01-30_02-23-22`) to store log files.

```sh
> tree -L 3 .
.
├── fig16b.dat          # the results of this test
├── fig16b.eps          # fig16b
├── fig16b.plt          # drawing script of fig16b
├── run-logs
│   └── single
│       └── 2022-01-30_02-23-22 # running log files
└── run.sh              # the main running script

```



```sh
> cat fig16b.dat
dataset dgl     fgnn    app
PR      4.38    1.61    GCN
TW      11.57   2.65    GCN
PA      15.72   4.91    GCN
PR      2.25    .47     GraphSAGE
# ...

```

## FAQ
