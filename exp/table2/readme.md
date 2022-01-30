# Table 2: Similarity of hot nodes across epochs

The goal of this experiment is to show that, the hottest nodes in different epoch is similar. The definition of similarity is in paper.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs

## Run Command

```sh
> python runner.py
> python parser.py
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

## Output Example

```sh
> cat table2.dat
                            node_access:epoch_similarity                                 
dataset_short                                         PR         TW         PA         UK
sample_type         app                                                                  
kKHop2              gcn                        73.924075  78.884412  91.303635  77.457764
kRandomWalk         pinsage                    78.214917  72.702608  87.112816  64.021383
kWeightedKHopPrefix gcn                        77.701915  66.630311  89.575640  72.996494

```

## FAQ