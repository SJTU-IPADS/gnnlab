# Figure 5: Gap between degree-based policy and optimal.

The goal of this experiment is to show the gap between degree-based policy and optimal.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig5a.eps`.

degree-based results is done by running same tests like figure 4a, while the results of optimal requires an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`)

Fig 5a & 5b is similar, while 5a uses 3hop neighbour sampling on papers100M dataset, and 5b uses 3hop weighted sampling on twitter dataset.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L29` in `runner.py`. `percent_gen(0, 30, 1)` means run test from cache ratio 0% to 30% with step=1%.

## Run Command

```sh
> python runner.py
> python parser.py
> gnuplot plot.plt
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

The number of epochs to run is set to 3 for fast reproduce. You may change line containing `.override('epoch', [3])` to change the numer of epochs.


## Output Example

`python runner.py` will redirect all logs to `run-logs` directory. An short example of the `data.dat` looks like this:
```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent	batch_miss_nbytes	batch_feat_nbytes
degree	0.0	PA	kKHop2	gcn	0.000	0.000	290971183.4702	290971183.4702
degree	5.0	PA	kKHop2	gcn	28.930	97.166	206796937.3245	290974516.5563
degree	10.0	PA	kKHop2	gcn	49.570	100.000	146735258.2781	290975610.0662
degree	15.0	PA	kKHop2	gcn	63.960	100.000	104867731.4967	290967316.3444
degree	20.0	PA	kKHop2	gcn	75.110	100.000	72411585.2715	290979756.9272
degree	25.0	PA	kKHop2	gcn	82.900	100.000	49750253.351	290937805.1391
degree	30.0	PA	kKHop2	gcn	88.400	100.000	33741050.9139	290962416.7417

```

## FAQ