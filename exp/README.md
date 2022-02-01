# Experiments

## Paper's Hardware Configurations
- 2 * 24 cores Intel Xeon Platinum 8163 CPUs
- 512GB RAM
- 8 * 16GB NVIDIA Tesla V100 GPUs

**Note: If you don't have the same hardware environments, you should go into subdirectories(i.e. figXX or tableXX), follow the instructions to modify some script configurations, and then run the experiment**



## AE Environment
- 2 * 24 cores Intel Xeon Platinum 8163 CPUs

- 512GB RAM

- 8 * 32GB NVIDIA Tesla V100 GPUs

  


## Run All Experiments

Running all the experiments will take hours. 
We don't recommend running  all the experiments at one time.


```bash
make all
```

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
|   Fig13a   |       26 tests       |      25 mins      |
|   Fig13b   |       33 tests       |      30 mins      |
|    Fig14   |       36 tests       |      37 mins      |
|    Fig15   |        4 tests       |      30 mins      |
|   Fig16a   |       14 tests       |      20 mins      |
|   Fig16b   |       18 tests       |      25 mins      |
|   Table1   |       12 tests       |       7 mins      |
|   Table2   |       12 tests       |      15 mins      |
|   Table4   |       44 tests       |      44 mins      |
|   Table5   |       32 tests       |      36 mins      |
|   Table6   |        4 tests       |       5 mins      |


## Run One Experiment

`make figXX.run`(e.g. `make fig4a.run`) or `make  tableXX.run` (e.g. `make table1.run`)

or `cd figXX/tableXX` and follow the instruction to run the experiment.

**Each experiment takes about 10-60 minutes.**

## Experiment Output

The experiment output files are in the subdirectories(`figXX/run-logs` or `figXX/output_XXXXXXX`).



## Clean Run Logs

```bash
make clean
```



## FAQ

####  The paper reported OOM in some test cases(UK dataset). However those test cases run successfully  in AE environment.

In the FGNN, all tests were run in 16GB V100 machine. In the AE environment, each GPU has 32GB RAM. All 16GB V100 machines have been occupied.



#### AE data mismatch the paper data.

- Check if someone else is running the test at the same time. Use `nvidia-smi` and `htop` command.
- **Do not run `nvidia-smi` command during the tests. `nvidia-smi` command harm the performance severely.**