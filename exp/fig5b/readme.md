# Figure 5: Gap between degree-based policy and optimal.

The goal of this experiment is to show the gap between degree-based policy and optimal.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig5b.eps`.

degree-based results is done by running same tests like figure 4a, while the results of optimal requires an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`)

Fig 5a & 5b is similar, while 5a uses 3hop neighbour sampling on papers100M dataset, and 5b uses 3hop weighted sampling on twitter dataset.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L29` in `runner.py`. `percent_gen(0, 35, 1)` means run test from cache ratio 0% to 35% with step=1%.

## Run Command

```sh
> python runner.py
> python parser.py
> gnuplot plot.plt
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

## Output Example

`python runner.py` will redirect all logs to `run-logs` directory. An short example of the `data.dat` looks like this:
```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent	batch_miss_nbytes	batch_feat_nbytes
degree	0.0	TW	kWeightedKHopPrefix	gcn	0.000	0.000	344889073.5094	344889073.5094
degree	5.0	TW	kWeightedKHopPrefix	gcn	42.920	59.066	196767260.9811	344712153.3585
degree	10.0	TW	kWeightedKHopPrefix	gcn	48.380	74.328	177905789.5849	344619288.1509
degree	15.0	TW	kWeightedKHopPrefix	gcn	50.960	84.666	169084155.1698	344796208.3019
degree	20.0	TW	kWeightedKHopPrefix	gcn	52.450	91.498	163933106.717	344790730.8679
degree	25.0	TW	kWeightedKHopPrefix	gcn	53.580	95.388	160116504.1509	344915755.4717
degree	30.0	TW	kWeightedKHopPrefix	gcn	54.520	99.279	156894604.0755	344983271.8491
degree	35.0	TW	kWeightedKHopPrefix	gcn	55.300	100.000	154118105.3585	344808979.3208


```

## FAQ