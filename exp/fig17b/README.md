# Figure 17b:  Performance Test on a Single GPU

The goal of this experiment is to get the end-to-end performance between DGL, T<sub>SOTA</sub> and GNNLab over a single GPU.

- `run.sh` is the runner script.

## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of vertex(in percentage, 0<=pct. <=1) to be cached.
  - **GNNLab**: Modify the arguments `cache-percentage` of L27-L37 in run.sh.
  - **T<sub>SOTA</sub>**: Modify the arguments `cache-percentage` of L40-L50 in run.sh.

## Run Command


```sh
> bash run.sh
```

## Output Example

`bash run.sh` will create a new folder(e.g. `run-logs/single/2022-01-30_02-23-22`) to store log files.

```sh
> tree -L 3 .
.
├── fig17b.dat          # the results of this test
├── fig17b.eps          # fig17b
├── fig17b.plt          # drawing script of fig17b
├── run-logs
│   └── single
│       └── 2022-01-30_02-23-22 # running log files
└── run.sh              # the main running script

```



```sh
> cat fig17b.dat
dataset	dgl	fgnn	sgnn	app
PR	4.47	1.62	1.52	GCN
TW	11.74	2.72	4.14	GCN
PA	16.17	4.97	9.42	GCN
PR	2.29	.47	0.43	GraphSAGE
# ...

```

## FAQ
