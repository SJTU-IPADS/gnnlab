# Figure 11a: Impact of presample epochs.

The goal of this experiment is to show that the performance of presample is close to optimal, while requires only 1~2 epoch of sample results.
This test is similar to figure 5b, while adds extra tests of presample under different sample epochs.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig11a.eps`.

The results of optimal is produced by an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`). This is already included in `runner.py`.

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
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent
degree	0.0	TW	kWeightedKHopPrefix	gcn	0.000	0.000
degree	5.0	TW	kWeightedKHopPrefix	gcn	42.920	59.050
degree	10.0	TW	kWeightedKHopPrefix	gcn	48.390	74.315
degree	15.0	TW	kWeightedKHopPrefix	gcn	50.960	84.654
degree	20.0	TW	kWeightedKHopPrefix	gcn	52.460	91.492
degree	25.0	TW	kWeightedKHopPrefix	gcn	53.590	95.381
degree	30.0	TW	kWeightedKHopPrefix	gcn	54.540	99.271
degree	35.0	TW	kWeightedKHopPrefix	gcn	55.320	100.000
presample_1	0.0	TW	kWeightedKHopPrefix	gcn	0.000	0.000
presample_1	5.0	TW	kWeightedKHopPrefix	gcn	52.350	59.050
presample_1	10.0	TW	kWeightedKHopPrefix	gcn	63.990	74.315
presample_1	15.0	TW	kWeightedKHopPrefix	gcn	73.140	84.654
presample_1	20.0	TW	kWeightedKHopPrefix	gcn	80.260	91.492
presample_1	25.0	TW	kWeightedKHopPrefix	gcn	81.720	95.381
presample_1	30.0	TW	kWeightedKHopPrefix	gcn	82.980	99.271
presample_1	35.0	TW	kWeightedKHopPrefix	gcn	84.020	100.000
presample_2	0.0	TW	kWeightedKHopPrefix	gcn	0.000	0.000
presample_2	5.0	TW	kWeightedKHopPrefix	gcn	54.910	59.050
presample_2	10.0	TW	kWeightedKHopPrefix	gcn	67.680	74.315
presample_2	15.0	TW	kWeightedKHopPrefix	gcn	77.520	84.654
presample_2	20.0	TW	kWeightedKHopPrefix	gcn	82.120	91.492
presample_2	25.0	TW	kWeightedKHopPrefix	gcn	86.790	95.381
presample_2	30.0	TW	kWeightedKHopPrefix	gcn	88.610	99.271
presample_2	35.0	TW	kWeightedKHopPrefix	gcn	89.350	100.000
```

## FAQ