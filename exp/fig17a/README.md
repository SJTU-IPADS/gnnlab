# Figure 17a:  Dynamic Swtching Test

The goal of this experiment is to get the runtime of one epoch in GNNLab w/ and w/o dynamic switching for training PinSAGE on Papers100M.

- `run.sh` is the runner script.

## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of vertex(in percentage, 0<=pct. <=1) to be cached. ②Number of GPU.
  - **Original GNNLab with async training**: Modify the arguments `cache-percentage`, `num-sample-worker` and `num-train-worker` of L11-L17 in run.sh.
  - **GNNLab with dynamic switching**: Modify the arguments `cache-percentage`(cache percentage for trainer) and `switch-cache-percentage`(cache percentage for switcher), `num-sample-worker` and `num-train-worker` of L20-L26 in run.sh.

## Run Command


```sh
> bash run.sh
```

## Output Example

`bash run.sh` will create a new folder(e.g. `run-logs/switch/2022-01-30_12-51-05/`) to store log files.

```sh
> tree -L 3 .
.
├── fig17a.dat          # the results of this test
├── fig17a.eps          # fig16a
├── fig17a.plt          # drawing script of fig16a
├── run-logs
│   └── switch
│       └── 2022-01-30_12-51-05 # running log files
└── run.sh              # the main running script

```



```sh
> cat fig17a.dat
Config  "w/o DS"        "w/ DS"
"1S 1T"  6.48           3.8678
"1S 2T"  3.29           2.5552
# ...

```

## FAQ
