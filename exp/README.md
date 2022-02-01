# Experiments

## Default Hardware Configurations
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


```sh
make all
```



## Run One Experiment

`make figXX.run`(e.g. `make fig4a.run`) or `make  tableXX.run` (e.g. `make table1.run`)

or `cd figXX/tableXX` and follow the instruction to run the experiment.



## Experiment Output

The experiment output files are in the subdirectories(`figXX/run-logs` or `figXX/output_XXXXXXX`).



## Clean Run Logs

```sh
make clean
```



## FAQ

####  The paper reported OOM in some test cases(UK dataset). However those test cases run successfully  in AE environment.

In the FGNN, all tests were run in 16GB V100 machine. In the AE environment, each GPU has 32GB RAM. All 16GB V100 machines have been occupied.



#### AE data mismatch the paper data.

- Check if someone else is running the test at the same time. Use `nvidia-smi` and `htop` command.
- **Do not run `nvidia-smi` command during the tests. `nvidia-smi` command harm the performance severely.**