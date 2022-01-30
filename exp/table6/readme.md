# Table 6: Init cost

The goal of this experiment is to show that the init cost of presample is small.

Before running this test, please generate feature for twitter and uk-2006-05 dataset:
```bash
dd if=/dev/zero of=/graph-learning/samgraph/twitter/feat.bin count=41652230 bs=1024
dd if=/dev/zero of=/graph-learning/samgraph/uk-2006-05/feat.bin count=77741046 bs=1024
```
This is necessary since these dataset does not provides feature, while this tests requires measuring time for loading feature data from disk. Please make sure that you have enough disk space(120GB) to hold them.

This test also requires page cache to be cleaned before each run. Normally, this can be achieved by:
```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

In our provided machine, this can be achieved by `sudo /opt/clean_page_cache/run.sh` without password.

`runner.sh` handles page cache properly, then calls `runner.py`. It may ask you for sudo password.
`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs

## Run Command

```sh
> bash ./runner.sh
> python parser.py
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

## Output Example

```sh
> cat table6.dat
                                    0            1            2            3
cache_policy              presample_1  presample_1  presample_1  presample_1
cache_percentage                100.0         24.0         20.0         13.0
dataset_short                      PR           TW           PA           UK
sample_type                    kKHop2       kKHop2       kKHop2       kKHop2
app                               gcn          gcn          gcn          gcn
Disk to DRAM                    12.92        51.59       553.23       104.25
DRAM to GPU-mem                  0.82         9.18         10.1        10.67
Load graph topology              0.11         1.17         1.42         2.49
Load feature cache               0.71         8.01         8.68         8.18
Pre-sampling for PreSC#1         0.43          0.7         1.98         1.15

```

## FAQ