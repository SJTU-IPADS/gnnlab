# Experiments

### Table of Contents 
  - [Overview](#overview)
  - [Paper's Hardware Configurations](#papers-hardware-configurations)
  - [Run A Single Experiment](#run-a-single-experiment)
  - [Run All Experiments](#run-all-experiments)
  - [Example output.](#example-output)
  - [Expected Running Time](#expected-running-time)
  - [Clean Experiment Logs](#clean-experiment-logs)
  - [FAQ](#faq)


## Overview
Our experiments have been automated by scripts (`run.py`). Each figure or table in our paper is treated as one experiment and is associated with a subdirectory in `fgnn-artifacts/exp`. The script will automatically run the experiment, save the logs into files, and parse the output data from the files.

```bash
> tree -L 2 fgnn-artifacts/exp
fgnn-artifacts/exp
├── fig4a
├── fig4b
├── ...
├── table1
│   ├── README.md
│   ├── run.py
├── ...
├── Makefile
```
## Paper's Hardware Configurations
- 8 * NVIDIA V100 GPUs (**16GB** of memory each)
- 2 * Intel Xeon Platinum 8163 CPUs (24 cores each)
- 512GB RAM

**Note: If you have a different hardware environment, you need to goto the subdirectories (i.e., `figXX` or `tableXX`), follow the instructions to modify some script configurations(e.g. smaller cache ratio), and then run the experiment**


## Run A Single Experiment

The following commands are used to run a certain experiment(e.g. table1).

```bash
cd fgnn-artifacts/exp
make table1.run
```

Moreover, the user can also goto a subdirectory (e.g., `fgnn-artifacts/exp/table1`) and then follow the instruction (`README.md`) to run the experiment.


## Run All Experiments

The following commands are used to run all experiments. Note that running all experiments may take several hours. This [table](exp/README.md#expected-running-time) lists the expected running time for each experiment.

```bash
cd fgnn-artifacts/exp
make all
```

## Example output.

The experiment output files are in the subdirectories (`figXX/run-logs` or `figXX/output_XX`). The output files include log files for each testcase, parsed data, and eps-format figures.

```bash
> cat output_2022-01-29_17-19-44/table1.dat
GNN Systems               Sample  Extract  Train  Total    #
DGL                         4.40    13.63   4.10  22.51    # logs_dgl/test1.log
 w/ GPU-base Sampling       1.28    13.61   4.12  19.08    # logs_dgl/test0.log
SGNN                        3.26     5.96   4.14  13.37    # logs_sgnn/test3.log
 w/ GPU-base Caching        3.05     1.95   4.07   8.99    # logs_sgnn/test2.log
 w/ GPU-base Sampling       0.72     5.88   4.07  10.70    # logs_sgnn/test1.log
 w/ Both                    0.72     3.90   3.98   8.61    # logs_sgnn/test0.log
```

## Expected Running Time

| Experiment | Number of Test Cases | Expected Run Time |
|:----------:|:--------------------:|:-----------------:|
|    Fig4a   |       31 tests       |      40 mins      |
|    Fig4b   |       31 tests       |      40 mins      |
|    Fig5a   |       31 tests       |      40 mins      |
|    Fig5b   |       36 tests       |      40 mins      |
|    Fig10   |       48 tests       |      50 mins      |
|   Fig11a   |       73 tests       |      80 mins      |
|   Fig11b   |       94 tests       |      120 mins     |
|   Fig11c   |       94 tests       |      120 mins     |
|    Fig12   |       33 tests       |      30 mins      |
|    Fig13   |       33 tests       |      30 mins      |
|   Fig14a   |       34 tests       |      32 mins      |
|   Fig14b   |       34 tests       |      23 mins      |
|    Fig15   |       36 tests       |      40 mins      |
|   Fig16a   |        3 tests       |      25 mins      |
|   Fig17a   |       14 tests       |      20 mins      |
|   Fig17b   |       27 tests       |      28 mins      |
|   Table1   |       12 tests       |      10 mins      |
|   Table2   |       12 tests       |      15 mins      |
|   Table4   |       44 tests       |      45 mins      |
|   Table5   |       32 tests       |      35 mins      |
|   Table6   |        4 tests       |       5 mins      |

## Clean Experiment Logs

```bash
make clean
```



## FAQ

**The data mismatch the paper data.**

- **Do not run `nvidia-smi` command during the tests. `nvidia-smi` command harm the performance severely.**
