# Figure 14b:  GCN Scalability Test

The goal of this experiment is to show the scalability performance of DGL, SGNN and FGNN on GCN model.

Dataset: twitter

- `run.py` is the runner script.
- `logtable_def.py` defines log parsing rules.



## Hardware Requirements

- Paper's configurations: **8x16GB** NVIDIA V100 GPUs, **2x24** cores Intel 8163 CPU
- For other hardware configurations, you may need to modify the ①Number of GPU. ②Number of CPU threads ③Number of vertex (in percentage, 0<=pct. <=1) to be cached.
  - **DGL:** Modify `L66-L67(#GPU)` in `run.py`.
  - **FGNN:**  Modify  `L108(#CPU threads), L118-L153(#GPU, #Cache percentage)` in `run.py`.
  - **SGNN:**  Modify  `L187(#Cache percentage), L190(#GPU)` in `run.py`.



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

`python run.py` will create a new folder(e.g. `output_2022-01-29_20-10-39`) as result.

`python run.py --rerun-tests`  does not create a new folder and reuse the last created folder.

```sh
> tree output_2022-01-29_20-10-39 -L 1
output_2022-01-29_20-10-39
├── fig14b.eps             # Output figure
├── fig14b-full.res        # Output data with comments
├── fig14b.res             # Output data
├── logs_dgl
├── logs_fgnn
└── logs_sgnn

3 directories, 3 files
```



```sh
> cat output_2022-01-29_20-10-39/fig14b.res
"GPUs"	"DGL"	"SGNN"	"1S"	"2S"	"3S"
1   11.86	5.47	-	-	-
2	7.07	4.26	1.69	-	-
3	5.65	3.20	0.95	1.70	-
4	4.88	2.65	0.71	0.97	1.69
...
```





## FAQ
